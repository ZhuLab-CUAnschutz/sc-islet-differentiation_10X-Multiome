"""Stage 3.6: rank variants as contrast-panel examples ("winners" first, diverse).

A good example has: a large effect (|logfc| in its top cell type), the variant in
an accessible region there (high predicted counts, a real peak), and a clear
profile-shape change (high JSD, a proxy for a gained/lost motif). We also want
variety: not all the same motif (cap per disrupted TF, tighter for CTCF), and a
mix of specificity scopes (cell-type, lineage, endocrine, broad), not only single
-cell-type effects.

Per variant: top cell type = argmax |logfc|; scope from the set of significant
cell types (config `significance`): cell_type (<=2 sig), lineage:X (all sig share
one lineage), endocrine / non_endocrine (all sig share endocrine status), else
broad. Disrupted TF = the most frequent TF among the variant's called hits.

Selection: keep variants significant in their top cell type AND accessible there
(predicted counts >= per-trait median). Order by |logfc| then JSD, then greedily
pick with a per-TF cap (CTCF/CTCFL tighter) and a cap on cell_type-scope share so
lineage/endocrine/broad examples are represented.

Usage:
    python src/3_report/6_rank_examples.py --config config/config.yaml [--max-examples 80]
"""

import argparse

import numpy as np
import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging

log = setup_logging(__file__)
CTCF_TFS = {"CTCF", "CTCFL"}


def scope_of(sig_cts, md) -> str:
    """Classify specificity scope from the significant cell types.

    cell_type = one cell type; lineage:X = all significant cell types share lineage
    X (e.g. early+late beta); endocrine / non_endocrine = share endocrine status
    across lineages; broad = spans both. Endocrine sub-lineages have only two cell
    types, so 'same lineage' (not a <=2 count) is what makes something lineage-scoped.
    """
    sub = md.loc[[c for c in sig_cts if c in md.index]]
    if len(sub) <= 1:
        return "cell_type"
    lineages, endos = set(sub["lineage"]), set(sub["endocrine"])
    if len(lineages) == 1:
        return f"lineage:{next(iter(lineages))}"
    if endos == {"endocrine"}:
        return "endocrine"
    if endos == {"non_endocrine"}:
        return "non_endocrine"
    return "broad"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores-dir", default="results/chrombpnet")
    ap.add_argument("--max-examples", type=int, default=80)
    ap.add_argument("--tf-cap", type=int, default=6)
    ap.add_argument("--ctcf-cap", type=int, default=2)
    ap.add_argument("--max-celltype-frac", type=float, default=0.55)
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    cell_types = metadata.cell_types(cfg)
    md = metadata.load_cell_type_metadata(cfg)
    threshold = cfg["significance"]["pval_threshold"]
    scores_dir = io.sandbox_path(cfg, args.scores_dir)
    cols = ("logfc", "abs_logfc_x_jsd.pval", "allele1_pred_counts", "allele2_pred_counts", "jsd")
    logfc, pval, a1c, a2c, jsd = io.build_score_matrices(scores_dir, cell_types, cols=cols)
    credible = io.load_credible_sets(cfg)
    hit_map = io.load_hit_map(cfg)

    rows = []
    for trait in cfg["traits"]:
        key = trait["key"]
        ids = [v for v in credible[credible["trait"] == key]["id"].unique() if v in logfc.index]
        sub = logfc.loc[ids]
        top_ct = sub.abs().idxmax(axis=1)
        idx = [sub.columns.get_loc(c) for c in top_ct]
        r = np.arange(len(sub))
        peak = np.maximum(a1c.loc[ids].values[r, idx], a2c.loc[ids].values[r, idx])
        sig_mask = pval.loc[ids] <= threshold
        scopes = [scope_of(list(sig_mask.columns[sig_mask.loc[v]]), md) for v in ids]
        # Cell types by descending |effect| per variant: the primary/displayed hit motif
        # is the one called in the strongest cell types, not a global vote across all 22.
        ct_order = {v: list(sub.loc[v].abs().sort_values(ascending=False).index) for v in ids}
        prim = {v: io.primary_hit_for(hit_map.get(v), ct_order[v]) for v in ids}
        df = pd.DataFrame({
            "trait": key, "variant_id": ids, "top_cell_type": top_ct.values,
            "logfc": sub.values[r, idx], "abs_logfc": np.abs(sub.values[r, idx]),
            "peak_pred_counts": peak, "jsd": jsd.loc[ids].values[r, idx],
            "pval": pval.loc[ids].values[r, idx], "n_sig_ct": sig_mask.sum(axis=1).values,
            "scope": scopes,
        })
        df = df[(df["pval"] <= threshold) & (df["peak_pred_counts"] >= df["peak_pred_counts"].median())]
        df["disrupts_hit"] = df["variant_id"].isin(hit_map)
        df["primary_motif"] = df["variant_id"].map(lambda v: prim[v][0])  # ST##:TF at top ct
        df["primary_tf"] = df["variant_id"].map(lambda v: prim[v][1])  # curator TF at top ct
        df["hit_celltypes"] = df["variant_id"].map(
            lambda v: ";".join(hit_map[v]["cts"]) if v in hit_map else "")
        df["hit_motifs"] = df["variant_id"].map(
            lambda v: ";".join(io.order_hits_by_ct(hit_map.get(v), ct_order[v])))
        rows.append(df)

    allc = pd.concat(rows, ignore_index=True).sort_values(
        ["abs_logfc", "jsd"], ascending=False).reset_index(drop=True)

    # Stratified selection: quotas per specificity scope so the set spans cell-type,
    # lineage, and endocrine specificity (not all broad); within each, biggest effects
    # first with a per-disrupted-TF cap (CTCF tighter) for motif variety.
    def bucket(scope):
        return "lineage" if scope.startswith("lineage") else scope
    frac = {"cell_type": 0.35, "lineage": 0.30, "endocrine": 0.20, "non_endocrine": 0.06, "broad": 0.09}
    quota = {b: int(round(f * args.max_examples)) for b, f in frac.items()}
    filled = {b: 0 for b in frac}
    picked, seen, tf_count = [], set(), {}

    def tf_ok(tf):
        cap = args.ctcf_cap if tf in CTCF_TFS else args.tf_cap
        return tf_count.get(tf, 0) < cap

    for _, row in allc.iterrows():  # effect-sorted; fill per-scope quotas
        b = bucket(row["scope"])
        if filled[b] >= quota[b] or not tf_ok(row["primary_tf"]):
            continue
        picked.append(row); seen.add(row["variant_id"])
        filled[b] += 1; tf_count[row["primary_tf"]] = tf_count.get(row["primary_tf"], 0) + 1
    for _, row in allc.iterrows():  # backfill to max_examples by effect (still TF-capped)
        if len(picked) >= args.max_examples:
            break
        if row["variant_id"] in seen or not tf_ok(row["primary_tf"]):
            continue
        picked.append(row); seen.add(row["variant_id"])
        tf_count[row["primary_tf"]] = tf_count.get(row["primary_tf"], 0) + 1

    sel = pd.DataFrame(picked).sort_values(["abs_logfc", "jsd"], ascending=False).reset_index(drop=True)
    sel.insert(0, "rank", np.arange(1, len(sel) + 1))

    out = io.sandbox_path(cfg, "results/variants")
    allc.round(4).to_csv(out / "example_candidates.tsv", sep="\t", index=False)
    sel.round(4).to_csv(out / "example_ranking.tsv", sep="\t", index=False)
    log.info("selected %d/%d; scopes=%s; top TFs=%s",
             len(sel), len(allc), dict(sel["scope"].apply(lambda s: s.split(":")[0]).value_counts()),
             dict(sel["primary_tf"].value_counts().head(8)))
    io.write_provenance(cfg, "3_report_6_rank_examples", {
        "max_examples": args.max_examples, "tf_cap": args.tf_cap, "ctcf_cap": args.ctcf_cap,
        "max_celltype_frac": args.max_celltype_frac, "n_candidates": int(len(allc))})


if __name__ == "__main__":
    main()
