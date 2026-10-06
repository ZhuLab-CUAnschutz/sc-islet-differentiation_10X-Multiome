#!/usr/bin/env python
"""Cluster-side path and geometry checks for higher-order marginalization."""

import argparse
import os
import sys

import numpy as np


# ── constants ────────────────────────────────────────────────────────────────

CELL_TYPES = [
    "DE", "ENP_phase1", "FB_FLT1", "PFG1", "PFG2", "PGT1", "PGT2", "PGT3",
    "PP1", "PP2", "SC_delta_GHRL", "early_ENP", "early_SC_EC",
    "early_SC_alpha", "early_SC_beta", "exocrine", "late_ENP", "late_SC_EC",
    "late_SC_alpha", "late_SC_beta", "liver", "proliferating_endocrine",
]

REQUIRED_CWM_KEYS = ["ST18", "ST36_sub1", "ST66", "J_FOXH1"]

EXPECTED_ROWS = {
    "single_motifs.tsv": 4,
    "pair_arrangements.tsv": 10,
    "triples.tsv": 12,
    "null_triples.tsv": 42,
}

SEQ_LEN = 2114


# ── helpers ──────────────────────────────────────────────────────────────────

def cwm_consensus_ohe(cwm, imp_frac=0.1):
    """Return the trimmed width of a CWM after importance-fraction trimming."""
    W4 = cwm if cwm.shape[1] == 4 else cwm.T
    imp = np.abs(W4).sum(1)
    if imp.max() <= 0:
        keep = np.arange(len(W4))
    else:
        keep = np.where(imp >= imp_frac * imp.max())[0]
    core = W4[keep.min():keep.max() + 1]
    return core.shape[0]


def count_tsv_rows(path):
    """Return the number of data rows (excluding header) in a TSV."""
    with open(path) as fh:
        lines = [l for l in fh if l.strip()]
    return len(lines) - 1  # subtract header


def load_tsv(path):
    """Load a TSV into a list of dicts keyed by header columns."""
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        vals = line.split("\t")
        rows.append(dict(zip(header, vals)))
    return rows


# ── checks ───────────────────────────────────────────────────────────────────

def check_models(models_root, cts, failures):
    """Check that model and background files exist for each cell type."""
    for ct in cts:
        model = os.path.join(
            models_root, "models", ct, "fold_0", "chrombpnet", "0.5",
            "models", "chrombpnet_nobias.h5",
        )
        bg = os.path.join(
            models_root, "models", ct, "fold_0", "chrombpnet", "0.5",
            "auxiliary", "filtered.nonpeaks.bed",
        )
        if os.path.isfile(model):
            print(f"PASS  model exists: {ct}")
        else:
            print(f"FAIL  model missing: {model}")
            failures.append(f"model missing: {ct}")
        if os.path.isfile(bg):
            print(f"PASS  background exists: {ct}")
        else:
            print(f"FAIL  background missing: {bg}")
            failures.append(f"background missing: {ct}")


def check_genome(genome, failures):
    """Check genome fasta and .fai index."""
    if os.path.isfile(genome):
        print(f"PASS  genome fasta exists: {genome}")
    else:
        print(f"FAIL  genome fasta missing: {genome}")
        failures.append("genome fasta missing")

    fai = genome + ".fai"
    if os.path.isfile(fai):
        print(f"PASS  genome index exists: {fai}")
    else:
        print(f"FAIL  genome index missing: {fai}")
        failures.append("genome .fai index missing")


def check_cwm(npz_path, failures):
    """Check CWM npz exists and contains required keys."""
    if not os.path.isfile(npz_path):
        print(f"FAIL  CWM npz missing: {npz_path}")
        failures.append("CWM npz missing")
        return None

    print(f"PASS  CWM npz exists: {npz_path}")
    data = np.load(npz_path)
    widths = {}
    for key in REQUIRED_CWM_KEYS:
        if key in data:
            w = cwm_consensus_ohe(data[key])
            widths[key] = w
            print(f"PASS  CWM key '{key}' present (trimmed width={w})")
        else:
            print(f"FAIL  CWM key '{key}' missing from {npz_path}")
            failures.append(f"CWM key '{key}' missing")
    return widths


def check_input_tsvs(inputs_dir, failures):
    """Check TSV files exist and have expected row counts."""
    for fname, expected in EXPECTED_ROWS.items():
        path = os.path.join(inputs_dir, fname)
        if not os.path.isfile(path):
            print(f"FAIL  input TSV missing: {path}")
            failures.append(f"input TSV missing: {fname}")
            continue
        n = count_tsv_rows(path)
        if n == expected:
            print(f"PASS  {fname}: {n} rows (expected {expected})")
        else:
            print(f"FAIL  {fname}: {n} rows (expected {expected})")
            failures.append(f"{fname} row count: got {n}, expected {expected}")


def check_pair_widths(inputs_dir, cwm_widths, failures):
    """Check that wa and wb in pair_arrangements match motif widths in npz."""
    path = os.path.join(inputs_dir, "pair_arrangements.tsv")
    if not os.path.isfile(path) or cwm_widths is None:
        return

    rows = load_tsv(path)
    for i, row in enumerate(rows):
        idA = row.get("idA", "")
        idB = row.get("idB", "")
        wa = int(row.get("wa", -1))
        wb = int(row.get("wb", -1))

        if idA in cwm_widths:
            expected_wa = cwm_widths[idA]
            if wa == expected_wa:
                print(f"PASS  pair {i}: wa={wa} matches {idA} width")
            else:
                print(f"FAIL  pair {i}: wa={wa} but {idA} width={expected_wa}")
                failures.append(
                    f"pair {i} wa mismatch: {wa} vs {idA} width {expected_wa}"
                )

        if idB in cwm_widths:
            expected_wb = cwm_widths[idB]
            if wb == expected_wb:
                print(f"PASS  pair {i}: wb={wb} matches {idB} width")
            else:
                print(f"FAIL  pair {i}: wb={wb} but {idB} width={expected_wb}")
                failures.append(
                    f"pair {i} wb mismatch: {wb} vs {idB} width {expected_wb}"
                )


def check_pair_spans(inputs_dir, failures):
    """Check all pair spans (wa + gap + wb) fit in SEQ_LEN bp."""
    path = os.path.join(inputs_dir, "pair_arrangements.tsv")
    if not os.path.isfile(path):
        return

    rows = load_tsv(path)
    for i, row in enumerate(rows):
        wa = int(row.get("wa", 0))
        wb = int(row.get("wb", 0))
        gap = int(row.get("opt_gap", 0))
        span = wa + gap + wb
        if span <= SEQ_LEN:
            print(f"PASS  pair {i}: span={span} fits in {SEQ_LEN} bp")
        else:
            print(f"FAIL  pair {i}: span={span} exceeds {SEQ_LEN} bp")
            failures.append(f"pair {i} span {span} > {SEQ_LEN}")


def check_triples_vs_pairs(inputs_dir, failures):
    """Check that anchor_orient and anchor_gap in triples match pairs."""
    pairs_path = os.path.join(inputs_dir, "pair_arrangements.tsv")
    if not os.path.isfile(pairs_path):
        return

    pairs = load_tsv(pairs_path)
    pair_lookup = {}
    for row in pairs:
        key = (row.get("idA", ""), row.get("idB", ""))
        gap = int(row.get("opt_gap", 0))
        orient = row.get("opt_orient", "")
        pair_lookup.setdefault(key, []).append({"gap": gap, "orient": orient})

    for tsv_name in ["triples.tsv", "null_triples.tsv"]:
        triples_path = os.path.join(inputs_dir, tsv_name)
        if not os.path.isfile(triples_path):
            continue

        triples = load_tsv(triples_path)
        for i, row in enumerate(triples):
            idA = row.get("idA", "")
            idB = row.get("idB", "")
            anchor_gap = int(row.get("anchor_gap", -999))

            key = (idA, idB)
            if key not in pair_lookup:
                print(
                    f"FAIL  {tsv_name} row {i}: anchor pair "
                    f"({idA}, {idB}) not in pair_arrangements"
                )
                failures.append(
                    f"{tsv_name} row {i}: anchor pair not in pair_arrangements"
                )
                continue

            matched = any(p["gap"] == anchor_gap for p in pair_lookup[key])

            if matched:
                print(
                    f"PASS  {tsv_name} row {i}: anchor_gap={anchor_gap} "
                    f"matches pair ({idA}, {idB})"
                )
            else:
                print(
                    f"FAIL  {tsv_name} row {i}: anchor_gap={anchor_gap} "
                    f"not found for pair ({idA}, {idB})"
                )
                failures.append(
                    f"{tsv_name} row {i}: anchor_gap mismatch"
                )


def check_triple_composite_span(inputs_dir, cwm_widths, failures, maxdist=200):
    """Warn about max composite span and gap positions skipped at flanks."""
    pairs_path = os.path.join(inputs_dir, "pair_arrangements.tsv")
    if not os.path.isfile(pairs_path) or cwm_widths is None:
        return

    pairs = load_tsv(pairs_path)
    pair_wa_wb = {}
    for row in pairs:
        key = (row.get("idA", ""), row.get("idB", ""))
        pair_wa_wb.setdefault(key, {
            "wa": int(row.get("wa", 0)),
            "wb": int(row.get("wb", 0)),
        })

    for tsv_name in ["triples.tsv", "null_triples.tsv"]:
        triples_path = os.path.join(inputs_dir, tsv_name)
        if not os.path.isfile(triples_path):
            continue

        triples = load_tsv(triples_path)
        for i, row in enumerate(triples):
            idA = row.get("idA", "")
            idB = row.get("idB", "")
            idC = row.get("idC", "")
            anchor_gap = int(row.get("anchor_gap", 0))

            key = (idA, idB)
            if key not in pair_wa_wb:
                continue

            wa = pair_wa_wb[key]["wa"]
            wb = pair_wa_wb[key]["wb"]
            wc = cwm_widths.get(idC, 0)
            if wc == 0:
                continue

            composite = wa + anchor_gap + wb + maxdist + wc
            headroom = SEQ_LEN - composite
            if headroom < 0:
                print(
                    f"FAIL  {tsv_name} row {i}: composite span {composite} "
                    f"exceeds {SEQ_LEN} bp (headroom={headroom})"
                )
                failures.append(
                    f"{tsv_name} row {i}: composite span {composite} > {SEQ_LEN}"
                )
            else:
                skipped_per_flank = max(0, maxdist - headroom // 2)
                print(
                    f"WARN  {tsv_name} row {i}: composite={composite} "
                    f"(wa={wa}+gap={anchor_gap}+wb={wb}+maxdist={maxdist}"
                    f"+wc={wc}), headroom={headroom}, "
                    f"~{skipped_per_flank} gap positions skipped per flank"
                )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preflight checks for higher-order marginalization."
    )
    parser.add_argument(
        "--models_root", required=True,
        help="Root directory for single-task models.",
    )
    parser.add_argument(
        "--genome", required=True,
        help="Path to hg38 genome fasta.",
    )
    parser.add_argument(
        "--all-cts", action="store_true",
        help="Check model paths for all 22 cell types (default: DE only).",
    )
    args = parser.parse_args()

    # inputs/ is relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_dir = os.path.join(os.path.dirname(script_dir), "inputs")

    failures = []

    print("=" * 60)
    print("PREFLIGHT: higher-order marginalization")
    print("=" * 60)

    # 1. Model paths
    print("\n--- Model paths ---")
    cts = CELL_TYPES if args.all_cts else ["DE"]
    check_models(args.models_root, cts, failures)

    # 2. Genome fasta
    print("\n--- Genome ---")
    check_genome(args.genome, failures)

    # 3. CWM npz
    print("\n--- CWM npz ---")
    npz_path = os.path.join(inputs_dir, "motifs_de4.npz")
    cwm_widths = check_cwm(npz_path, failures)

    # 4. Input TSVs
    print("\n--- Input TSVs ---")
    check_input_tsvs(inputs_dir, failures)

    # 5. Pair widths vs CWM widths
    print("\n--- Pair widths vs CWM ---")
    check_pair_widths(inputs_dir, cwm_widths, failures)

    # 6. Triples vs pairs consistency
    print("\n--- Triples vs pairs ---")
    check_triples_vs_pairs(inputs_dir, failures)

    # 7. Pair spans fit in sequence length
    print("\n--- Pair span check ---")
    check_pair_spans(inputs_dir, failures)

    # 8. Triple composite span warnings
    print("\n--- Triple composite span ---")
    check_triple_composite_span(inputs_dir, cwm_widths, failures)

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"PREFLIGHT FAILED  ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED  (all checks ok)")
        sys.exit(0)


if __name__ == "__main__":
    main()
