#!/usr/bin/env python
"""Downstream synergy pipeline: aggregate -> in-matrix empirical-null soft/hard calls -> filter ->
cluster (all + endocrine/non-endocrine). Runs on whatever chunk summaries exist (partial or full),
so it can be prototyped on completed chunks while the sweep is still running.

Inputs : <results>/chunk_*/summary__all.tsv  (one per completed chunk; default $SYNERGY_DIR/sweep)
         catalog v1.1 metadata.tsv            (short_id, curator_category, curator_tf)
         inputs/frac_pos.tsv                  (from extract_frac_pos.py; optional)
Outputs: tables/agg_long.tsv, tables/calls_long.tsv, tables/pair_annot.tsv,
         tables/cluster_labels.tsv, figures/synergy_clustermap_rowz.{png,pdf}
"""
import os, glob, argparse
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SF = os.environ.get("SYNERGY_DIR", os.path.join(REPO, "results", "6_synergy"))  # synergy work dir
CONFIG = os.path.join(REPO, "config")

ALPHA = 0.05            # BH q cutoff for a synergy call
CENTRAL_LO, CENTRAL_HI = 2.5, 97.5   # central-bulk percentiles used to fit the per-CT null

COMPOSITE_CATS = {"Heterocomposite","Homocomposite","Base obligate homodimer"}


def load_ct_facets(config_dir):
    """Canonical cell-type metadata from the repo config/.
    Returns dev_order (by display_order), endocrine set, and per-CT + facet color maps."""
    ct = pd.read_csv(os.path.join(config_dir, "cell_type_metadata.tsv"), sep="\t").sort_values("display_order")
    dev_order = ct.cell_type.tolist()
    endocrine = set(ct.loc[ct.endocrine == "endocrine", "cell_type"])
    ct_color = dict(zip(ct.cell_type, ct.color))
    lineage_of = dict(zip(ct.cell_type, ct.lineage))
    lin = pd.read_csv(os.path.join(config_dir, "lineage_metadata.tsv"), sep="\t")
    lin_color = dict(zip(lin.lineage, lin.color))
    endo = pd.read_csv(os.path.join(config_dir, "endocrine_metadata.tsv"), sep="\t")
    endo_color = dict(zip(endo.endocrine, endo.color))
    return {"dev_order": dev_order, "endocrine": endocrine, "ct_color": ct_color,
            "lineage_of": lineage_of, "lin_color": lin_color, "endo_color": endo_color}


def bh_qvalues(p):
    """Benjamini-Hochberg q-values over a 1D array of p (NaNs preserved)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]; n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]        # enforce monotonicity
    qv = np.empty(n); qv[order] = np.clip(ranked, 0, 1)
    q[ok] = qv
    return q


def right_tail_p(col):
    """Per-CT right-tail p vs a robust null fit on the central bulk of this CT's values."""
    x = col.values.astype(float)
    finite = x[np.isfinite(x)]
    lo, hi = np.percentile(finite, [CENTRAL_LO, CENTRAL_HI])
    bulk = finite[(finite >= lo) & (finite <= hi)]
    mu, sd = np.median(bulk), (bulk.std(ddof=1) or 1e-9)
    return pd.Series(norm.sf((x - mu) / sd), index=col.index)


def aggregate(results_dir, meta_path, fracpos_path=None):
    files = sorted(glob.glob(os.path.join(results_dir, "chunk_*", "summary__all.tsv")))
    assert files, f"no summaries under {results_dir}"
    long = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    long["pair"] = long.idA + "__" + long.idB
    long = long.drop_duplicates(["pair", "ct"])
    # frac_pos = fraction of backgrounds where joint > additive (within-pair consistency; from npz).
    if fracpos_path and os.path.exists(fracpos_path):
        fp = pd.read_csv(fracpos_path, sep="\t")[["idA", "idB", "ct", "frac_pos"]]
        long = long.merge(fp, on=["idA", "idB", "ct"], how="left")
        print(f"[agg] merged frac_pos ({long.frac_pos.notna().sum()}/{len(long)} rows have it)")
    else:
        long["frac_pos"] = np.nan
        print("[agg] frac_pos table absent -> column is NaN")
    print(f"[agg] {len(files)} chunks -> {long.pair.nunique()} pairs x {long.ct.nunique()} CT "
          f"({len(long)} pair-CT rows)")

    meta = pd.read_csv(meta_path, sep="\t").set_index("short_id")
    cat = meta["curator_category"].to_dict(); tf = meta["curator_tf"].to_dict()
    annot = (long[["pair","idA","idB"]].drop_duplicates("pair").set_index("pair"))
    annot["tf_A"] = annot.idA.map(tf); annot["tf_B"] = annot.idB.map(tf)
    annot["category_A"] = annot.idA.map(cat); annot["category_B"] = annot.idB.map(cat)
    annot["is_homodimer"] = annot.idA == annot.idB
    annot["involves_composite"] = (annot.category_A.isin(COMPOSITE_CATS)
                                   | annot.category_B.isin(COMPOSITE_CATS))
    return long, annot


def call_synergy(long):
    """In-matrix empirical null (per-CT central-bulk fit) + BH across all pair-CT tests."""
    d = long.copy()
    d["p_delta"] = d.groupby("ct")["delta"].transform(right_tail_p)
    d["p_maxz"]  = d.groupby("ct")["maxZ"].transform(right_tail_p)
    d["q_delta"] = bh_qvalues(d["p_delta"].values)
    d["q_maxz"]  = bh_qvalues(d["p_maxz"].values)
    d["q_wilcox"] = bh_qvalues(d["wilcoxon_p"].values)   # computed for the record; NOT a gate
    # GATE = empirical-null Delta only. (The wilcoxon-at-argmax saturates -> non-binding; the
    # real within-pair consistency check is frac_pos below, reported now, not yet gated.)
    d["synergistic"] = d.q_delta < ALPHA
    d["hard"] = d.synergistic & (d.q_maxz < ALPHA)       # peaky arrangement = hard subset
    d["soft"] = d.synergistic & ~d.hard
    d["call"] = np.where(d.hard, "hard", np.where(d.soft, "soft", "none"))
    print(f"[call] pair-CT synergistic={int(d.synergistic.sum())} "
          f"(hard={int(d.hard.sum())} soft={int(d.soft.sum())}) of {len(d)} tests | "
          f"gate q_delta<{ALPHA}")
    return d


def cluster_rows(mat, kmax=15):
    """Ward on row-z of mat (pairs x CT); pick k by silhouette. Returns labels, order, k."""
    z = mat.sub(mat.mean(1), axis=0).div(mat.std(1).replace(0, 1), axis=0).fillna(0)
    if len(z) < 4:
        return pd.Series(1, index=z.index), list(z.index), 1, z
    L = linkage(z.values, method="ward")
    best_k, best_s = 2, -1
    for k in range(2, min(kmax, len(z) - 1) + 1):
        lab = fcluster(L, k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(z.values, lab)
        if s > best_s:
            best_k, best_s = k, s
    labels = pd.Series(fcluster(L, best_k, criterion="maxclust"), index=z.index)
    from scipy.cluster.hierarchy import leaves_list
    order = [z.index[i] for i in leaves_list(L)]
    print(f"[cluster] {len(z)} pairs, k={best_k} (silhouette={best_s:.3f})")
    return labels, order, best_k, z


def orientation_qc(calls, annot, out):
    """Distributions of the winning orientation/gap among synergistic calls, plus a homodimer
    under-calling check (homodimers search 3 orientations vs 4 for distinct -> less argmax inflation
    -> may be judged conservatively against the mostly-4-orientation null)."""
    c = calls.merge(annot.reset_index()[["pair","is_homodimer"]], on="pair")
    syn = c[c.synergistic]
    # orientation counts by call tier
    ot = (syn.groupby(["opt_orient","call"]).size().unstack(fill_value=0)
             .reindex(["FF","FR","RF","RR"]).fillna(0).astype(int))
    # gap summary
    gstat = syn.opt_gap.describe()[["mean","50%","min","max"]]
    # homodimer vs distinct synergy RATE (fraction of pair-CT tests called synergistic)
    rate = c.groupby("is_homodimer").synergistic.mean()
    n_by = c.groupby("is_homodimer").size()
    ot.to_csv(os.path.join(out,"tables","orientation_qc.tsv"), sep="\t")
    print(f"[orient-qc] winning orientation among synergistic:\n{ot.to_string()}")
    print(f"[orient-qc] opt_gap: mean={gstat['mean']:.1f} median={gstat['50%']:.0f} "
          f"min={gstat['min']:.0f} max={gstat['max']:.0f}")
    print(f"[orient-qc] synergy RATE  homodimer={rate.get(True,float('nan')):.3f} "
          f"(n={int(n_by.get(True,0))})  distinct={rate.get(False,float('nan')):.3f} "
          f"(n={int(n_by.get(False,0))})")
    # frac_pos: within-pair consistency (reported, NOT gated). Show what a future >=0.75 gate would drop.
    if syn.frac_pos.notna().any():
        fpv = syn.frac_pos.dropna()
        would_drop = int((fpv < 0.75).sum())
        print(f"[frac-qc] frac_pos among synergistic: median={fpv.median():.2f} "
              f"min={fpv.min():.2f}  |  a future >=0.75 gate would drop {would_drop}/{len(fpv)} calls")

    npanel = 3 if syn.frac_pos.notna().any() else 2
    fig, ax = plt.subplots(1, npanel, figsize=(4.6*npanel, 3.6))
    ot.plot(kind="bar", stacked=True, ax=ax[0],
            color={"hard":"#b2182b","soft":"#ef8a62","none":"#dddddd"})
    ax[0].set_title("winning orientation (synergistic)")
    ax[0].set_xlabel("orientation"); ax[0].set_ylabel("# pair-CT calls")
    ax[1].hist(syn.opt_gap.values, bins=range(0, 52, 2), color="#4d4d4d")
    ax[1].set_title("optimal gap (synergistic)")
    ax[1].set_xlabel("gap (bp)"); ax[1].set_ylabel("# pair-CT calls")
    if npanel == 3:
        ax[2].hist(syn.frac_pos.dropna().values, bins=[i/20 for i in range(21)], color="#2166ac")
        ax[2].axvline(0.75, color="#b2182b", ls="--", lw=1)
        ax[2].set_title("frac_pos (synergistic; 0.75 guide)")
        ax[2].set_xlabel("frac backgrounds joint>additive"); ax[2].set_ylabel("# pair-CT calls")
    fig.tight_layout()
    fig.savefig(os.path.join(out,"figures","orientation_gap_qc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ot


def main(a):
    for sub in ["tables", "figures"]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    ct = load_ct_facets(a.config)
    DEV_ORDER, ENDOCRINE = ct["dev_order"], ct["endocrine"]
    long, annot = aggregate(a.results, a.meta, a.fracpos)
    calls = call_synergy(long)
    orientation_qc(calls, annot, a.out)

    # wide delta matrix (pairs x CT), ordered columns
    delta = calls.pivot(index="pair", columns="ct", values="delta")
    cols = [c for c in DEV_ORDER if c in delta.columns] + \
           [c for c in delta.columns if c not in DEV_ORDER]
    delta = delta[cols]

    # filter: synergistic (soft or hard) in >=1 CT
    syn_any = calls.groupby("pair").synergistic.any()
    keep = syn_any[syn_any].index
    print(f"[filter] {len(keep)}/{delta.shape[0]} pairs synergistic in >=1 CT")
    dsel = delta.loc[delta.index.intersection(keep)]

    labels, order, k, z = cluster_rows(dsel)

    # endocrine / non-endocrine tag per pair (called in which block)
    endo_cols = [c for c in dsel.columns if c in ENDOCRINE]
    nonendo_cols = [c for c in dsel.columns if c not in ENDOCRINE]
    syn_wide = calls.pivot(index="pair", columns="ct", values="synergistic").reindex(dsel.index).fillna(False)
    in_endo = syn_wide[endo_cols].any(axis=1); in_non = syn_wide[nonendo_cols].any(axis=1)
    tag = np.where(in_endo & in_non, "shared",
          np.where(in_endo, "endocrine", np.where(in_non, "non_endocrine", "none")))
    tag = pd.Series(tag, index=dsel.index)
    print(f"[split] endocrine={int((tag=='endocrine').sum())} "
          f"non_endocrine={int((tag=='non_endocrine').sum())} shared={int((tag=='shared').sum())}")

    # ---- figure: row-z clustermap, canonical developmental column order + colors ----
    row_colors = tag.map({"endocrine": ct["endo_color"]["endocrine"],
                          "non_endocrine": ct["endo_color"]["non_endocrine"], "shared": "#b8b8b8"})
    col_bar = pd.DataFrame({
        "endocrine": [ct["endo_color"]["endocrine" if c in ENDOCRINE else "non_endocrine"] for c in z.columns],
        "lineage":   [ct["lin_color"].get(ct["lineage_of"].get(c), "#dddddd") for c in z.columns],
    }, index=z.columns)
    g = sns.clustermap(z, col_cluster=False, row_cluster=(len(z) >= 4),
                       cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
                       row_colors=row_colors, col_colors=col_bar,
                       figsize=(9, max(6, min(0.06 * len(z) + 3, 22))),
                       yticklabels=False, cbar_pos=(0.02, 0.83, 0.03, 0.12))
    g.ax_heatmap.set_xlabel("cell type (canonical developmental order)")
    g.ax_heatmap.set_ylabel(f"{len(z)} synergistic pairs (row-z of Δ*)")
    g.fig.suptitle(f"Synergy row-z(Δ*)  |  {long.pair.nunique()} pairs profiled, "
                   f"{len(keep)} synergistic in ≥1 CT  |  k={k}", y=1.02, fontsize=10)
    for ext in ("png", "pdf"):
        g.fig.savefig(os.path.join(a.out, "figures", f"synergy_clustermap_rowz.{ext}"),
                      dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # ---- within-block reclustering: endocrine block (11 CT) and non-endocrine block (11 CT) ----
    # For each cell-type block, take the synergistic pairs that fire in that block and recluster them
    # on the row-z of Δ across ONLY that block's cell types -> block-specific syntax substructure.
    block_labels = []
    for bname, bcols, bmask in [("endocrine", endo_cols, in_endo), ("non_endocrine", nonendo_cols, in_non)]:
        rows_b = dsel.index[bmask.values]
        mat_b = dsel.loc[rows_b, bcols]
        lab_b, ord_b, kb, zb = cluster_rows(mat_b)
        print(f"[block:{bname}] {len(rows_b)} pairs synergistic in this block, {len(bcols)} CT, k={kb}")
        rc = tag.reindex(zb.index).map({"endocrine": ct["endo_color"]["endocrine"],
              "non_endocrine": ct["endo_color"]["non_endocrine"], "shared": "#b8b8b8"})
        cbar = pd.Series([ct["lin_color"].get(ct["lineage_of"].get(c), "#dddddd") for c in zb.columns],
                         index=zb.columns, name="lineage")
        gb = sns.clustermap(zb, col_cluster=False, row_cluster=(len(zb) >= 4),
                            cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
                            row_colors=rc, col_colors=cbar,
                            figsize=(6.2, max(6, min(0.06 * len(zb) + 3, 22))),
                            yticklabels=False, cbar_pos=(0.02, 0.83, 0.03, 0.12))
        gb.ax_heatmap.set_xlabel(f"{bname} cell types")
        gb.ax_heatmap.set_ylabel(f"{len(zb)} pairs synergistic in ≥1 {bname} CT (row-z of Δ within block)")
        gb.fig.suptitle(f"{bname} block  |  {len(zb)} pairs  |  k={kb}", y=1.02, fontsize=10)
        for ext in ("png", "pdf"):
            gb.fig.savefig(os.path.join(a.out, "figures", f"synergy_clustermap_{bname}.{ext}"),
                           dpi=150, bbox_inches="tight")
        plt.close(gb.fig)
        block_labels.append(pd.DataFrame({"pair": zb.index, "block": bname,
                                          "block_cluster": lab_b.reindex(zb.index).values,
                                          "endo_tag": tag.reindex(zb.index).values}))
    pd.concat(block_labels).to_csv(os.path.join(a.out, "tables", "block_cluster_labels.tsv"),
                                   sep="\t", index=False)

    # ---- write tables ----
    long_out = calls.merge(annot.reset_index()[["pair","tf_A","tf_B","category_A","category_B",
                                                 "is_homodimer","involves_composite"]], on="pair")
    long_out.to_csv(os.path.join(a.out, "tables", "calls_long.tsv"), sep="\t", index=False)
    annot.to_csv(os.path.join(a.out, "tables", "pair_annot.tsv"), sep="\t")
    delta.to_csv(os.path.join(a.out, "tables", "delta_wide.tsv"), sep="\t")
    pd.DataFrame({"pair": dsel.index, "cluster": labels.reindex(dsel.index).values,
                  "endo_tag": tag.values,
                  "involves_composite": annot.reindex(dsel.index).involves_composite.values,
                  "is_homodimer": annot.reindex(dsel.index).is_homodimer.values}
                 ).to_csv(os.path.join(a.out, "tables", "cluster_labels.tsv"), sep="\t", index=False)
    print(f"[done] wrote tables/ + figures/ under {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(SF, "sweep"))
    p.add_argument("--meta", default=os.path.join(REPO, "results/3_single_task_models/motifs/v1.1/metadata.tsv"))
    p.add_argument("--config", default=CONFIG, help="dir with cell_type_metadata.tsv + facet metadata")
    p.add_argument("--fracpos", default=os.path.join(SF, "inputs", "frac_pos.tsv"),
                   help="TSV idA,idB,ct,frac_pos from npz (within-pair consistency); optional")
    p.add_argument("--out", default=SF)
    main(p.parse_args())
