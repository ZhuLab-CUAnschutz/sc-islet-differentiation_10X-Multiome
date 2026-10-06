#!/usr/bin/env python
"""Validate frozen inputs, CREsted API, model shape, and the known WT prediction."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from common import one_hot, read_single_fasta, sequence_sha256, write_json


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def validate_sequence() -> tuple[str, str]:
    name, sequence = read_single_fasta(config.FASTA_PATH)
    if len(sequence) != config.INPUT_LEN:
        raise ValueError(f"Expected {config.INPUT_LEN} bp, found {len(sequence)}")
    invalid = sorted(set(sequence).difference("ACGT"))
    if invalid:
        raise ValueError(f"Non-ACGT bases in construct: {invalid}")
    digest = sequence_sha256(sequence)
    if digest != config.REFERENCE_SEQUENCE_SHA256:
        raise ValueError(
            f"Construct sequence SHA256 changed: {digest}; "
            f"expected {config.REFERENCE_SEQUENCE_SHA256}"
        )
    if config.NO_MUTATION_FLANKS != (507, 507):
        raise ValueError(f"Unexpected protected flanks: {config.NO_MUTATION_FLANKS}")
    if config.ENHANCER_END - config.ENHANCER_START != 1100:
        raise ValueError("Expected a 1,100-bp mutable enhancer")
    return name, sequence


def validate_reference_table() -> pd.DataFrame:
    table = pd.read_csv(config.REFERENCE_PREDICTIONS)
    required = {"celltype", "construct"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing reference columns: {required.difference(table.columns)}")
    observed = tuple(table["celltype"])
    if observed != config.MODEL_CLASSES:
        raise ValueError(
            "Reference prediction row order does not match MODEL_CLASSES; "
            "do not proceed because model outputs could be mislabeled"
        )
    return table


def validate_model(sequence: str, reference: pd.DataFrame) -> dict:
    import keras

    from crested_compat import load_design_api

    api = load_design_api()
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    if input_shape[1:] != (config.INPUT_LEN, 4):
        raise ValueError(f"Unexpected model input shape: {input_shape}")
    if output_shape[-1] != len(config.MODEL_CLASSES):
        raise ValueError(f"Unexpected model output shape: {output_shape}")

    prediction = np.asarray(model.predict(one_hot(sequence), verbose=0))[0]
    expected = reference["construct"].to_numpy(dtype=float)
    absolute_difference = np.abs(prediction - expected)
    allowed = np.maximum(0.5, 0.01 * np.abs(expected))
    failing = absolute_difference > allowed
    if np.any(failing):
        rows = [
            (
                config.MODEL_CLASSES[index],
                float(expected[index]),
                float(prediction[index]),
                float(absolute_difference[index]),
            )
            for index in np.flatnonzero(failing)
        ]
        raise ValueError(
            "WT predictions do not reproduce the June reference within 1%/0.5 units: "
            + repr(rows)
        )

    return {
        "model_path": str(config.MODEL_PATH),
        "model_input_shape": input_shape,
        "model_output_shape": output_shape,
        "max_reference_absolute_difference": float(absolute_difference.max()),
        "crested_version": api.version,
        "crested_api_style": api.api_style,
        "crested_ise_signature": api.signature,
        "keras_version": keras.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Validate only files and sequence coordinates; useful off-cluster.",
    )
    args = parser.parse_args()

    name, sequence = validate_sequence()
    reference = validate_reference_table()
    report = {
        "status": "passed",
        "fasta_name": name,
        "sequence_length": len(sequence),
        "sequence_sha256": sequence_sha256(sequence),
        "enhancer_interval_0_based_half_open": [
            config.ENHANCER_START,
            config.ENHANCER_END,
        ],
        "mutable_bases": config.ENHANCER_END - config.ENHANCER_START,
        "model_classes": list(config.MODEL_CLASSES),
        "python_packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "keras", "tensorflow", "crested")
        },
    }
    if not args.skip_model:
        report.update(validate_model(sequence, reference))
        config.RESULTS.mkdir(parents=True, exist_ok=True)
        write_json(report, config.PREFLIGHT_JSON)

    print("SOX32 preflight passed")
    print(f"construct={len(sequence)} bp; mutable enhancer=507:1607 (1,100 bp)")
    if args.skip_model:
        print("model/API checks skipped")
    else:
        print(
            f"model={report['model_input_shape']} -> {report['model_output_shape']}; "
            f"CREsted={report['crested_version']}"
        )


if __name__ == "__main__":
    main()

