#!/usr/bin/env python
"""Evolve the tested SOX32 construct under three DE-focused objectives."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from common import (
    diff_mutations,
    file_sha256,
    one_hot,
    read_single_fasta,
    reconstruct_sequences,
    sequence_sha256,
    write_json,
)
from crested_compat import load_design_api
from objectives import activity_selector, specificity_selector


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def normalize_intermediate(intermediate: object) -> dict:
    if isinstance(intermediate, dict):
        return intermediate
    if isinstance(intermediate, (list, tuple)) and len(intermediate) == 1:
        info = intermediate[0]
        if isinstance(info, dict):
            return info
    raise TypeError(f"Unexpected CREsted intermediate result: {type(intermediate)}")


def run_one_objective(
    objective: str,
    model: object,
    sequence: str,
    baseline_prediction: np.ndarray,
    api: object,
) -> tuple[list[str], np.ndarray, list[tuple[int, str]]]:
    optimizer = None
    extra_kwargs: dict[str, float] = {}
    if objective == "de_activity":
        optimizer = api.EnhancerOptimizer(optimize_func=activity_selector)
    elif objective == "crested_default":
        optimizer = None
    elif objective == "de_specificity":
        optimizer = api.EnhancerOptimizer(optimize_func=specificity_selector)
        extra_kwargs = {
            "mean_weight": config.SPECIFICITY_MEAN_WEIGHT,
            "max_weight": config.SPECIFICITY_MAX_WEIGHT,
            "target_floor": (
                float(baseline_prediction[config.TARGET_INDEX])
                * config.SPECIFICITY_DE_FLOOR_FRACTION
            ),
            "floor_penalty": config.SPECIFICITY_FLOOR_PENALTY,
        }
    else:
        raise ValueError(f"Unknown objective: {objective}")

    intermediate, designed = api.in_silico_evolution(
        n_mutations=config.N_MUTATIONS,
        target=config.TARGET_INDEX,
        model=model,
        n_sequences=1,
        return_intermediate=True,
        no_mutation_flanks=config.NO_MUTATION_FLANKS,
        enhancer_optimizer=optimizer,
        starting_sequences=[sequence],
        **extra_kwargs,
    )
    info = normalize_intermediate(intermediate)
    changes = [(int(position), str(alt)) for position, alt in info["changes"]]
    sequences = reconstruct_sequences(str(info["initial_sequence"]), changes)
    predictions = np.asarray(info["predictions"], dtype=np.float32)

    expected_steps = config.N_MUTATIONS + 1
    if len(sequences) != expected_steps or predictions.shape != (
        expected_steps,
        len(config.MODEL_CLASSES),
    ):
        raise ValueError(
            f"{objective}: unexpected trajectory shapes "
            f"{len(sequences)=}, {predictions.shape=}"
        )
    if str(designed[0]) != sequences[-1]:
        raise ValueError(f"{objective}: reconstructed and returned final sequences differ")
    for step, candidate in enumerate(sequences):
        if len(candidate) != config.INPUT_LEN:
            raise ValueError(f"{objective} step {step}: sequence length changed")
        if candidate[: config.ENHANCER_START] != sequence[: config.ENHANCER_START]:
            raise ValueError(f"{objective} step {step}: left protected flank changed")
        if candidate[config.ENHANCER_END :] != sequence[config.ENHANCER_END :]:
            raise ValueError(f"{objective} step {step}: right protected flank changed")
        for position, _, _ in diff_mutations(sequence, candidate):
            if not config.ENHANCER_START <= position < config.ENHANCER_END:
                raise ValueError(f"{objective} step {step}: mutation outside enhancer")
    return sequences, predictions, changes


def main() -> None:
    import keras

    keras.utils.set_random_seed(config.SEED)
    api = load_design_api()
    _, sequence = read_single_fasta(config.FASTA_PATH)
    if sequence_sha256(sequence) != config.REFERENCE_SEQUENCE_SHA256:
        raise ValueError("Frozen input sequence failed its SHA256 check")
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")

    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    baseline_prediction = np.asarray(model.predict(one_hot(sequence), verbose=0))[0]
    if baseline_prediction.shape != (len(config.MODEL_CLASSES),):
        raise ValueError(f"Unexpected model output: {baseline_prediction.shape}")

    all_sequences: list[list[str]] = []
    all_predictions: list[np.ndarray] = []
    all_change_positions: list[list[int]] = []
    all_change_alts: list[list[str]] = []

    for objective in config.OBJECTIVES:
        print(f"Starting objective: {objective}", flush=True)
        sequences, predictions, changes = run_one_objective(
            objective, model, sequence, baseline_prediction, api
        )
        all_sequences.append(sequences)
        all_predictions.append(predictions)
        all_change_positions.append([int(position) for position, _ in changes])
        all_change_alts.append([str(alt) for _, alt in changes])
        print(
            f"Finished {objective}: "
            f"DE {predictions[0, config.TARGET_INDEX]:.3f} -> "
            f"{predictions[-1, config.TARGET_INDEX]:.3f}; "
            f"net substitutions={len(diff_mutations(sequence, sequences[-1]))}",
            flush=True,
        )

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.EVOLUTION_NPZ,
        objectives=np.asarray(config.OBJECTIVES),
        classes=np.asarray(config.MODEL_CLASSES),
        sequences=np.asarray(all_sequences),
        predictions=np.stack(all_predictions),
        change_positions=np.asarray(all_change_positions, dtype=np.int32),
        change_alts=np.asarray(all_change_alts),
        enhancer_interval=np.asarray(
            [config.ENHANCER_START, config.ENHANCER_END], dtype=np.int32
        ),
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "model_path": str(config.MODEL_PATH),
        "model_size_bytes": config.MODEL_PATH.stat().st_size,
        "model_sha256": file_sha256(config.MODEL_PATH),
        "sequence_sha256": sequence_sha256(sequence),
        "crested_version": api.version,
        "crested_api_style": api.api_style,
        "crested_ise_signature": api.signature,
        "package_versions": {
            name: package_version(name)
            for name in ("numpy", "keras", "tensorflow", "crested")
        },
        "target_class": config.TARGET_CLASS,
        "model_classes": list(config.MODEL_CLASSES),
        "objectives": list(config.OBJECTIVES),
        "n_mutations": config.N_MUTATIONS,
        "mutation_budgets": list(config.MUTATION_BUDGETS),
        "enhancer_interval_0_based_half_open": [
            config.ENHANCER_START,
            config.ENHANCER_END,
        ],
        "no_mutation_flanks": list(config.NO_MUTATION_FLANKS),
        "specificity_parameters": {
            "mean_weight": config.SPECIFICITY_MEAN_WEIGHT,
            "max_weight": config.SPECIFICITY_MAX_WEIGHT,
            "de_floor_fraction_of_wt": config.SPECIFICITY_DE_FLOOR_FRACTION,
            "floor_penalty": config.SPECIFICITY_FLOOR_PENALTY,
        },
        "seed": config.SEED,
    }
    write_json(metadata, config.RUN_METADATA_JSON)
    print(f"Wrote {config.EVOLUTION_NPZ}", flush=True)


if __name__ == "__main__":
    main()
