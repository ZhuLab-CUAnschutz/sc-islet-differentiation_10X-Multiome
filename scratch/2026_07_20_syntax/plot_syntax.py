#!/usr/bin/env python
"""Aggregate pairwise-marginalization results and reproduce paper Fig-4-style panels + summary tables.
Reads results/<sub>/summary__<CT>.tsv and per-pair NPZs. Empirical null (lineage-mismatched pairs)
calibrates synergy thresholds/FDR for the all-pairs regime. CT-specificity z-scored EXCLUDING the
primary (max) CT to avoid selection bias.  Env: eugene_tools (mpl/pandas/numpy).  CPU only.
"""
import os, glob, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def load_summaries(res, sub):
    fs = sorted(glob.glob(f"{res}/{sub}/summary__*.tsv"))
    dfs = [pd.read_csv(f, sep="\t") for f in fs if os.path.getsize(f) > 0]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main(a):
    os.makedirs(a.fig, exist_ok=True)
    cand = load_summaries(a.res, "candidate")
    null = load_summaries(a.res, "null")
    man = pd.read_csv(a.manifest, sep="\t")[["idA", "idB", "tfA", "tfB", "group", "note"]]
    cand = cand.merge(man, on=["idA", "idB"], how="left")
    cand["pair"] = cand["tfA"].fillna(cand["idA"]) + "×" + cand["tfB"].fillna(cand["idB"])

    # ---- empirical null thresholds (all-pairs / single-fold regime) : 95th pct of lineage-mismatched null ----
    d95 = float(np.nanpercentile(null["delta"], 95)) if len(null) else 0.15
    z95 = float(np.nanpercentile(null["maxZ"], 95)) if len(null) else 4.0
    delta_thresh = max(0.15, d95)      # paper's 0.15 OR empirical 95th pct of null, whichever stricter (~5% FDR)
    z_thresh = max(4.0, z95)
    cand["syn"] = (cand["delta"] > delta_thresh) & (cand["wilcoxon_p"] < 0.001)
    cand["hard_cal"] = (cand["maxZ"] > z_thresh) & cand["syn"]
    cand.to_csv(f"{a.fig}/candidate_calls.tsv", sep="\t", index=False)
    with open(f"{a.fig}/null_calibration.txt", "w") as fh:
        fh.write(f"null pairs: {len(null)}  delta 95pct={d95:.3f}  maxZ 95pct={z95:.2f}\n")
        fh.write(f"applied delta_thresh={delta_thresh:.3f}  z_thresh={z_thresh:.2f}\n")

    # ================= per-CT synergy landscape (hero CTs) =================
    hero_cts = ["early_SC_beta", "late_SC_beta", "early_SC_EC", "late_SC_EC"]
    # ---- Fig-4i: dJ_opt vs dS scatter (one hero CT) ----
    for ct in [c for c in hero_cts if c in cand["ct"].unique()][:1]:
        sub = cand[cand.ct == ct]
        fig, ax = plt.subplots(figsize=(5.2, 5))
        lim = [min(sub.dS.min(), sub.dJ_opt.min()) - .1, max(sub.dS.max(), sub.dJ_opt.max()) + .1]
        ax.plot(lim, lim, "k--", lw=.8, label="additive (ΔJ=ΔS)")
        for g, c in [("beta", "#c0392b"), ("EC", "#2980b9"), ("cross", "#8e44ad"),
                     ("homo", "#27ae60"), ("composite", "#e67e22"), ("endocrine", "#16a085")]:
            s = sub[sub.group == g]
            if len(s): ax.scatter(s.dS, s.dJ_opt, s=60, c=c, label=g, edgecolor="k", lw=.4, zorder=3)
        for _, r in sub[sub.syn].iterrows():
            ax.annotate(r["pair"], (r.dS, r.dJ_opt), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Σ independent effects  ΔS = ΔA+ΔB (log counts)")
        ax.set_ylabel("joint effect ΔJ at optimal arrangement")
        ax.set_title(f"Motif-pair synergy — {ct}\n(above diagonal = synergistic)")
        ax.legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(f"{a.fig}/fig_scatter_{ct}.png", dpi=150); plt.close(fig)

    # ---- Fig-4j: hard/soft/none barplot (hero CT) ----
    for ct in [c for c in hero_cts if c in cand["ct"].unique()][:1]:
        sub = cand[cand.ct == ct]
        cats = {"hard": int(sub.hard_cal.sum()),
                "soft only": int((sub.soft & ~sub.hard_cal & sub.syn).sum()),
                "syn (no syntax)": int((sub.syn & ~sub.soft & ~sub.hard_cal).sum()),
                "none": int((~sub.syn).sum())}
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(cats.keys(), cats.values(), color=["#c0392b", "#e67e22", "#95a5a6", "#dfe6e9"], edgecolor="k")
        ax.set_ylabel("# motif pairs"); ax.set_title(f"Syntax classes — {ct}")
        plt.xticks(rotation=20, ha="right"); fig.tight_layout(); fig.savefig(f"{a.fig}/fig_syntax_classes_{ct}.png", dpi=150); plt.close(fig)

    # ---- Fig-4k: ΔJ vs distance curves for representative synergistic pairs (best orientation row) ----
    reps = cand[(cand.ct == "early_SC_beta") & cand.syn].sort_values("delta", ascending=False).head(6)
    if len(reps):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for ax, (_, r) in zip(axes.ravel(), reps.iterrows()):
            npz = f"{a.res}/candidate/{r.idA}__{r.idB}__{r.ct}.npz"
            if not os.path.exists(npz): ax.axis("off"); continue
            z = np.load(npz, allow_pickle=True); dJ = z["dJ"]; gaps = z["gaps"]
            oi = list(z["orients"]).index(str(z["opt_orient"])) if "opt_orient" in z else int(np.nanargmax(np.nanmax(dJ, 1)))
            ax.plot(gaps, dJ[oi], "-", color="#c0392b", lw=1.4, label="ΔJ (best orient)")
            ax.axhline(r.dS, ls="--", color="k", lw=.8, label="ΔS additive")
            ax.axvline(r.opt_gap, ls=":", color="#2980b9", lw=.8)
            ax.set_title(f"{r['pair']}  ({r.opt_orient}@{int(r.opt_gap)}bp)", fontsize=8)
            ax.set_xlabel("inner-edge gap (bp)"); ax.set_ylabel("Δ log counts")
            ax.legend(fontsize=6)
        fig.suptitle("Distance-dependent joint effect (early_SC_beta)"); fig.tight_layout()
        fig.savefig(f"{a.fig}/fig_distance_curves.png", dpi=150); plt.close(fig)

    # ---- Fig-4h: optimal spacing histogram for hard-syntax pairs (all CTs) ----
    hard = cand[cand.hard_cal]
    if len(hard):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(hard.opt_center_center.dropna(), bins=range(0, 60, 3), color="#c0392b", edgecolor="k")
        ax.set_xlabel("center-to-center distance at optimal (bp)"); ax.set_ylabel("# hard-syntax pair·CT")
        ax.set_title("Preferred spacing (hard syntax)"); fig.tight_layout()
        fig.savefig(f"{a.fig}/fig_spacing_hist.png", dpi=150); plt.close(fig)

    # ================= CT-specificity heatmap (the hero result) =================
    piv = cand.pivot_table(index="pair", columns="ct", values="delta")
    order = [c for c in ["DE","PGT1","PGT2","PGT3","PFG1","PFG2","PP1","PP2","ENP_phase1","early_ENP","late_ENP",
             "early_SC_beta","late_SC_beta","early_SC_alpha","late_SC_alpha","early_SC_EC","late_SC_EC",
             "SC_delta_GHRL","proliferating_endocrine","exocrine","liver","FB_FLT1"] if c in piv.columns]
    piv = piv[order]
    # z-score across CTs EXCLUDING each row's own max CT (de-bias, fix M4)
    zrows = []
    for _, row in piv.iterrows():
        v = row.values.astype(float); mx = np.nanargmax(v)
        mask = np.ones_like(v, bool); mask[mx] = False
        mu, sd = np.nanmean(v[mask]), np.nanstd(v[mask]) + 1e-9
        zrows.append((v - mu) / sd)
    zpiv = pd.DataFrame(zrows, index=piv.index, columns=piv.columns)
    zpiv.to_csv(f"{a.fig}/ct_specificity_zmatrix.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(11, max(4, .34 * len(zpiv))))
    im = ax.imshow(zpiv.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(zpiv))); ax.set_yticklabels(zpiv.index, fontsize=7)
    ax.set_title("Cell-type specificity of pairwise synergy  (z of Δ=ΔJ−ΔS across CTs, excl. own max)")
    fig.colorbar(im, ax=ax, shrink=.6, label="z (synergy vs other CTs)")
    fig.tight_layout(); fig.savefig(f"{a.fig}/fig_ct_specificity_heatmap.png", dpi=150); plt.close(fig)

    # summary table
    tab = cand.sort_values(["group", "delta"], ascending=[True, False])[
        ["pair", "ct", "group", "dJ_opt", "dS", "delta", "opt_orient", "opt_gap", "opt_center_center",
         "maxZ", "hard_cal", "soft", "syn", "wilcoxon_p"]]
    tab.to_csv(f"{a.fig}/syntax_summary_table.tsv", sep="\t", index=False)
    print(f"wrote figures + tables to {a.fig}")
    print(f"null-calibrated delta_thresh={delta_thresh:.3f} z_thresh={z_thresh:.2f}")
    print(f"candidate pair·CT rows: {len(cand)}; synergistic: {int(cand.syn.sum())}; hard: {int(cand.hard_cal.sum())}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True); p.add_argument("--fig", required=True)
    p.add_argument("--manifest", required=True)
    main(p.parse_args())
