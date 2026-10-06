"""Stage 3.8: curated-union selection of the most interesting variants for the browser.

Pools candidates across all traits and picks a diverse, non-CTCF set. Selection leads
with disruptions/additions of lineage-specific TF binding sites, then fills with high-PIP
causal variants, the most cell-type-specific, and the largest effects. Excludes CTCF/CTCFL;
caps per disrupted TF for motif variety. The selected set is then ranked by plain effect
size (|logfc|, largest first).

Per variant: top cell type = argmax |logfc|; direction = loss (logfc<0, likely a
disrupted motif) or gain (logfc>0, a created motif); primary_tf = dominant TF among
its called hits; scope from the significant cell types; PIP from the credible set.
Gate: significant in top ct and accessible there (peak predicted counts >= per-trait
median). Writes results/variants/browser_variants.tsv.

Usage:
    python src/3_report/8_select_browser_variants.py --config config/config.yaml [--n 50]
"""

import argparse

import numpy as np
import pandas as pd

from common import io, metadata
from common.logging_setup import setup_logging

log = setup_logging(__file__)
CTCF_TFS = {"CTCF", "CTCFL"}

# Lineage / developmental islet TFs (v1.1 curator_tf spellings), to prioritise examples.
LINEAGE_TFS = {
    "SOX9", "SOX17", "HNF4G", "FEV", "NEUROD1", "MAFB", "HNF1A", "FOXA2", "EOMES",
    "GATA4/6", "RFX6", "RFX3", "OTX2", "PAX6", "ISL1", "LMX1A", "NKX6-1", "NKX2-2",
    "PDX1", "ONECUT1", "ONECUT2", "FOXC1", "FOXK1", "FOXO3", "FOXJ3", "GRHL2",
    "MIXL1", "MEIS1", "MECOM", "CREB5", "RARB", "SALL4", "SIX4",
}


def gini(x):
    a = np.sort(np.abs(x)); n = len(a); s = a.sum()
    return 0.0 if s == 0 else (2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * s)


def scope_of(sig_cts, md):
    sub = md.loc[[c for c in sig_cts if c in md.index]]
    if len(sub) <= 1:
        return "cell_type"
    lin, endo = set(sub["lineage"]), set(sub["endocrine"])
    if len(lin) == 1:
        return f"lineage:{next(iter(lin))}"
    if endo == {"endocrine"}:
        return "endocrine"
    if endo == {"non_endocrine"}:
        return "non_endocrine"
    return "broad"


def build_candidates(cfg, md, cell_types, threshold):
    cols = ("logfc", "abs_logfc_x_jsd.pval", "allele1_pred_counts", "allele2_pred_counts", "jsd")
    logfc, pval, a1c, a2c, jsd = io.build_score_matrices(
        io.sandbox_path(cfg, "results/chrombpnet"), cell_types, cols=cols)
    credible = io.load_credible_sets(cfg)
    hit_map = io.load_hit_map(cfg)
    rows = []
    for trait in cfg["traits"]:
        key = trait["key"]
        tsub = credible[credible["trait"] == key][["id", "PIP"]].drop_duplicates("id").set_index("id")
        ids = [v for v in tsub.index if v in logfc.index]
        sub = logfc.loc[ids]
        top_ct = sub.abs().idxmax(axis=1)
        idx = [sub.columns.get_loc(c) for c in top_ct]; r = np.arange(len(sub))
        peak = np.maximum(a1c.loc[ids].values[r, idx], a2c.loc[ids].values[r, idx])
        sig = pval.loc[ids] <= threshold
        # Cell types ranked by |effect| per variant, so the primary/displayed hit motif
        # is the one called in the strongest cell types, not a global vote across all 22.
        ct_order = {v: list(sub.loc[v].abs().sort_values(ascending=False).index) for v in ids}
        prim = {v: io.primary_hit_for(hit_map.get(v), ct_order[v]) for v in ids}
        df = pd.DataFrame({
            "trait": key, "variant_id": ids, "top_cell_type": top_ct.values,
            "logfc": sub.values[r, idx], "abs_logfc": np.abs(sub.values[r, idx]),
            "peak_pred_counts": peak, "jsd": jsd.loc[ids].values[r, idx],
            "pval": pval.loc[ids].values[r, idx], "PIP": tsub.loc[ids, "PIP"].values,
            "gini": sub.apply(lambda x: gini(x.values), axis=1).values,
            "scope": [scope_of(list(sig.columns[sig.loc[v]]), md) for v in ids],
        })
        df = df[(df["pval"] <= threshold) & (df["peak_pred_counts"] >= df["peak_pred_counts"].median())]
        df["direction"] = np.where(df["logfc"] < 0, "loss", "gain")
        df["primary_motif"] = df["variant_id"].map(lambda v: prim[v][0])  # ST##:TF at top ct
        df["primary_tf"] = df["variant_id"].map(lambda v: prim[v][1])  # curator TF at top ct
        df["hit_motifs"] = df["variant_id"].map(
            lambda v: ";".join(io.order_hits_by_ct(hit_map.get(v), ct_order[v])))
        df["disrupts_hit"] = df["variant_id"].isin(hit_map)
        df["is_lineage_tf"] = df["primary_tf"].isin(LINEAGE_TFS)
        rows.append(df)
    allc = pd.concat(rows, ignore_index=True)
    return allc[~allc["primary_tf"].isin(CTCF_TFS)]  # non-CTCF only


def curated_union(allc, n, per_tf_cap=5):
    picked, seen, tf_count = [], set(), {}

    def take(pool, k):
        for _, row in pool.iterrows():
            if len(picked) >= n or k <= 0:
                break
            v, tf = row["variant_id"], row["primary_tf"]
            if v in seen or tf_count.get(tf, 0) >= per_tf_cap:
                continue
            picked.append(row); seen.add(v); tf_count[tf] = tf_count.get(tf, 0) + 1; k -= 1

    # Exclude 'broad' scope entirely: those are ubiquitous-factor (often CTCF) sites,
    # not the cell-type/lineage-specific examples we want to feature.
    specific = allc[allc["scope"] != "broad"]
    # 1) lineage-TF disruptions/additions, biggest effect first (the headliners)
    take(specific[specific["is_lineage_tf"]].sort_values("abs_logfc", ascending=False), n // 2)
    # 2) high-PIP causal
    take(specific.sort_values(["PIP", "abs_logfc"], ascending=False), n // 5)
    # 3) most cell-type/group-specific
    take(specific.sort_values("gini", ascending=False), n // 5)
    # 4) largest specific effects to fill
    take(specific.sort_values("abs_logfc", ascending=False), n)
    sel = pd.DataFrame(picked)
    # rank by plain effect size (|logfc|), largest first
    sel = sel.sort_values("abs_logfc", ascending=False).reset_index(drop=True)
    sel.insert(0, "rank", np.arange(1, len(sel) + 1))
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--per-tf-cap", type=int, default=5)
    args = ap.parse_args()

    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    cell_types = metadata.cell_types(cfg)
    threshold = cfg["significance"]["pval_threshold"]

    allc = build_candidates(cfg, md, cell_types, threshold)
    named = [v for v in cfg["report"]["named_loci"]]
    sel = curated_union(allc, args.n, args.per_tf_cap)
    out = io.sandbox_path(cfg, "results/variants/browser_variants.tsv")
    sel.round(4).to_csv(out, sep="\t", index=False)
    log.info("selected %d browser variants; lineage-TF=%d; scopes=%s; top TFs=%s; loss/gain=%s",
             len(sel), int(sel["is_lineage_tf"].sum()),
             dict(sel["scope"].apply(lambda s: s.split(":")[0]).value_counts()),
             dict(sel["primary_tf"].value_counts().head(10)),
             dict(sel["direction"].value_counts()))
    io.write_provenance(cfg, "3_report_8_select_browser_variants",
                        {"n": args.n, "per_tf_cap": args.per_tf_cap, "criteria": "curated_union_non_ctcf"})


if __name__ == "__main__":
    main()
