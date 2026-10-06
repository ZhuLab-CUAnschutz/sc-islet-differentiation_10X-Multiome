#!/usr/bin/env python3
"""
build_inputs.py
Generate all input TSVs for the DE higher-order synergy analysis.

Reads from the prior de_synergy run and produces files format-compatible
with marginalize_tracks.py:
  - inputs/single_motifs.tsv
  - inputs/pair_arrangements.tsv
  - inputs/triples.tsv
  - inputs/null_triples.tsv

Usage:
    python scripts/build_inputs.py
"""

import os
import itertools

import numpy as np
import pandas as pd

# ── paths relative to this script ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # 2026_09_20_de_higher_order/

CALLS_LONG = os.path.join(
    SCRIPT_DIR, "..", "..", "2026_09_06_de_synergy", "tables", "calls_long.tsv"
)
MOTIFS_DE4 = os.path.join(
    SCRIPT_DIR, "..", "..", "2026_09_06_de_synergy", "inputs", "motifs_de4.npz"
)
CWMS_NPZ = os.path.join(PROJECT_DIR, "inputs", "cwms_v1.1.npz")
CATALOG_META = os.path.join(PROJECT_DIR, "config", "catalog_v1.1_metadata.tsv")
OUT_DIR = os.path.join(PROJECT_DIR, "inputs")

# ── the 4 DE motifs (set1 catalog IDs) ─────────────────────────────────────
DE_MOTIFS = [
    ("ST18", "EOMES"),
    ("ST36_sub1", "MIXL1"),
    ("ST66", "SOX17"),
    ("J_FOXH1", "FOXH1"),
]
DE_IDS = [m[0] for m in DE_MOTIFS]
DE_TFS = {m[0]: m[1] for m in DE_MOTIFS}

SET1_LABEL = "set1_catalog+JFOXH1"


# ── helpers ────────────────────────────────────────────────────────────────
def _pair_key(a, b):
    """Canonical unordered pair key."""
    return frozenset((a, b))


# ── main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Read motif widths from motifs_de4.npz ───────────────────────────
    with np.load(MOTIFS_DE4) as npz:
        motif_widths = {k: npz[k].shape[0] for k in npz.files}
    print("Motif widths:", motif_widths)

    # ── 2. Read set1 rows from calls_long.tsv ──────────────────────────────
    calls = pd.read_csv(CALLS_LONG, sep="\t")
    set1 = calls[calls["set"] == SET1_LABEL].copy()
    print(f"Set1 rows: {len(set1)}  (6 heterotypic + 4 homodimers expected)")

    # Build lookup by unordered pair of IDs
    pair_lookup = {}
    for _, row in set1.iterrows():
        key = _pair_key(row["idA"], row["idB"])
        pair_lookup[key] = row

    # ── A. single_motifs.tsv ───────────────────────────────────────────────
    # Categories from catalog metadata; FOXH1 has no catalog entry
    CATEGORIES = {"ST18": "Base", "ST36_sub1": "Base", "ST66": "Partial", "J_FOXH1": "JASPAR (no catalog)"}
    singles = pd.DataFrame(
        [
            {
                "short_id": sid,
                "curator_tf": tf,
                "curator_category": CATEGORIES.get(sid, ""),
                "width": motif_widths[sid],
            }
            for sid, tf in DE_MOTIFS
        ]
    )
    singles_path = os.path.join(OUT_DIR, "single_motifs.tsv")
    singles.to_csv(singles_path, sep="\t", index=False)
    print(f"Wrote {singles_path}  ({len(singles)} rows)")

    # ── B. pair_arrangements.tsv ───────────────────────────────────────────
    pair_cols = [
        "idA", "idB", "tfA", "tfB", "kind", "ct",
        "opt_orient", "opt_gap", "opt_center_center",
        "wa", "wb", "dS", "dJ_opt", "delta", "maxZ", "call",
    ]
    pairs = set1[pair_cols].copy().reset_index(drop=True)
    # Rename to match prez plot_tracks.py convention (tf_A, tf_B with underscores)
    pairs = pairs.rename(columns={"tfA": "tf_A", "tfB": "tf_B"})
    pairs_path = os.path.join(OUT_DIR, "pair_arrangements.tsv")
    pairs.to_csv(pairs_path, sep="\t", index=False)
    print(f"Wrote {pairs_path}  ({len(pairs)} rows)")

    # ── C. triples.tsv ────────────────────────────────────────────────────
    # 4 triples = C(4,3), for each triple 3 anchor-pair choices
    triple_rows = []
    for combo in itertools.combinations(DE_IDS, 3):
        # combo is a 3-tuple of short_ids, e.g. ("ST18", "ST36_sub1", "ST66")
        for i in range(3):
            # anchor pair = the two that are NOT at index i
            anchor = [combo[j] for j in range(3) if j != i]
            slid = combo[i]
            idA, idB = anchor[0], anchor[1]

            # look up the pair row (order-agnostic)
            key = _pair_key(idA, idB)
            row = pair_lookup[key]

            # Use the orientation from calls_long.tsv as-is.
            # Ensure idA/idB match the calls_long.tsv ordering.
            actual_idA, actual_idB = row["idA"], row["idB"]

            triple_rows.append(
                {
                    "idA": actual_idA,
                    "idB": actual_idB,
                    "idC": slid,
                    "tfA": DE_TFS[actual_idA],
                    "tfB": DE_TFS[actual_idB],
                    "tfC": DE_TFS[slid],
                    "anchor_orient": row["opt_orient"],
                    "anchor_gap": row["opt_gap"],
                    "anchor_cc": row["opt_center_center"],
                    "wa": motif_widths[actual_idA],
                    "wb": motif_widths[actual_idB],
                    "wc": motif_widths[slid],
                }
            )

    triples = pd.DataFrame(triple_rows)
    triples_path = os.path.join(OUT_DIR, "triples.tsv")
    triples.to_csv(triples_path, sep="\t", index=False)
    print(f"Wrote {triples_path}  ({len(triples)} rows)")

    # ── D. null_triples.tsv ───────────────────────────────────────────────
    # Select 7 lineage-mismatched null motifs from cwms_v1.1.npz

    # Read catalog metadata to find DE-related short_ids to exclude
    if os.path.exists(CATALOG_META):
        meta = pd.read_csv(CATALOG_META, sep="\t")
    else:
        # Fall back to the prez copy
        fallback = os.path.join(
            SCRIPT_DIR, "..", "..", "2026_08_08_prez", "config",
            "catalog_v1.1_metadata.tsv"
        )
        meta = pd.read_csv(fallback, sep="\t")
        print(f"  (using fallback catalog metadata from {fallback})")

    # Exclude any short_id whose curator_tf is one of the 4 DE TFs
    de_tf_set = {"EOMES", "MIXL1", "SOX17", "FOXH1"}
    exclude_ids = set(
        meta.loc[meta["curator_tf"].isin(de_tf_set), "short_id"].tolist()
    )
    # Also explicitly exclude JASPAR DE keys if present in cwms
    exclude_ids.update({"J_EOMES", "J_MIXL1", "J_SOX17", "J_FOXH1"})
    # And the set1 catalog DE IDs themselves (should already be caught)
    exclude_ids.update({"ST18", "ST36_sub1", "ST66"})

    with np.load(CWMS_NPZ) as npz:
        cwm_widths = {k: npz[k].shape[0] for k in npz.files}

    eligible = {
        k: w for k, w in cwm_widths.items() if k not in exclude_ids
    }
    print(f"Eligible null motifs: {len(eligible)} (after excluding {len(exclude_ids)} DE-related)")

    # Pick 7 with varying widths: sort by width, then pick evenly spaced
    eligible_sorted = sorted(eligible.items(), key=lambda x: (x[1], x[0]))
    n_eligible = len(eligible_sorted)
    indices = [int(round(i * (n_eligible - 1) / 6)) for i in range(7)]
    null_motifs = [eligible_sorted[i] for i in indices]
    print("Selected null motifs:")
    for sid, w in null_motifs:
        # look up curator_tf from metadata if available
        tf_row = meta.loc[meta["short_id"] == sid, "curator_tf"]
        tf_name = tf_row.values[0] if len(tf_row) > 0 else sid
        print(f"  {sid} (w={w}, tf={tf_name})")

    # 6 heterotypic pairs x 7 null motifs = 42 rows
    hetero_set1 = set1[set1["kind"] == "heterotypic"]
    null_rows = []
    for _, prow in hetero_set1.iterrows():
        for null_id, null_w in null_motifs:
            tf_row = meta.loc[meta["short_id"] == null_id, "curator_tf"]
            null_tf = tf_row.values[0] if len(tf_row) > 0 else null_id

            null_rows.append(
                {
                    "idA": prow["idA"],
                    "idB": prow["idB"],
                    "idC": null_id,
                    "tfA": prow["tfA"],
                    "tfB": prow["tfB"],
                    "tfC": null_tf,
                    "anchor_orient": prow["opt_orient"],
                    "anchor_gap": prow["opt_gap"],
                    "anchor_cc": prow["opt_center_center"],
                    "wa": int(prow["wa"]),
                    "wb": int(prow["wb"]),
                    "wc": null_w,
                    "note": f"null: {null_tf} ({null_id}) is lineage-mismatched",
                }
            )

    null_triples = pd.DataFrame(null_rows)
    null_triples_path = os.path.join(OUT_DIR, "null_triples.tsv")
    null_triples.to_csv(null_triples_path, sep="\t", index=False)
    print(f"Wrote {null_triples_path}  ({len(null_triples)} rows)")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  single_motifs.tsv  : {len(singles):>3} rows  (4 DE TFs)")
    print(f"  pair_arrangements.tsv: {len(pairs):>3} rows  (6 hetero + 4 homo)")
    print(f"  triples.tsv        : {len(triples):>3} rows  (4 triples x 3 anchors)")
    print(f"  null_triples.tsv   : {len(null_triples):>3} rows  (6 pairs x 7 nulls)")


if __name__ == "__main__":
    main()
