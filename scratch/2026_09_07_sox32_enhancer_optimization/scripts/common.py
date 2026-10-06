"""Small sequence and serialization helpers shared by the experiment scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_single_fasta(path: str | Path) -> tuple[str, str]:
    """Read exactly one FASTA record and return its name and uppercase sequence."""
    name = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    raise ValueError(f"Expected one FASTA record in {path}")
                name = line[1:].strip().split()[0]
            else:
                if name is None:
                    raise ValueError(f"Sequence encountered before FASTA header in {path}")
                chunks.append(line)
    if name is None or not chunks:
        raise ValueError(f"No FASTA record found in {path}")
    return name, "".join(chunks).upper()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("ascii")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_hot(sequence: str) -> np.ndarray:
    """Return a one-sequence A/C/G/T one-hot batch with float32 dtype."""
    array = np.zeros((1, len(sequence), 4), dtype=np.float32)
    for position, base in enumerate(sequence.upper()):
        if base not in BASE_TO_INDEX:
            raise ValueError(f"Non-ACGT base {base!r} at position {position}")
        array[0, position, BASE_TO_INDEX[base]] = 1.0
    return array


def reconstruct_sequences(
    initial_sequence: str,
    changes: Iterable[tuple[int, str]],
) -> list[str]:
    """Reconstruct step 0 through step N from CREsted's change records."""
    current = initial_sequence
    sequences = [current]
    for position, replacement in changes:
        position = int(position)
        replacement = str(replacement)
        if position == -1:
            continue
        current = current[:position] + replacement + current[position + len(replacement) :]
        sequences.append(current)
    return sequences


def diff_mutations(reference: str, sequence: str) -> list[tuple[int, str, str]]:
    if len(reference) != len(sequence):
        raise ValueError("Reference and candidate lengths differ")
    return [
        (position, ref_base, alt_base)
        for position, (ref_base, alt_base) in enumerate(zip(reference, sequence))
        if ref_base != alt_base
    ]


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path, width: int = 80) -> None:
    with Path(path).open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start : start + width] + "\n")


def write_json(payload: dict, path: str | Path) -> None:
    with Path(path).open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

