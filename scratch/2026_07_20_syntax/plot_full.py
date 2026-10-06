#!/usr/bin/env python
"""Genome-wide syntax census over the full keep-set pair sweep (2211 pairs, primary CT).
Reproduces paper Fig-4 i (ΔJ vs ΔS scatter), j (hard/soft/none census), h (spacing histogram).
Synergy/hard thresholds calibrated from the lineage-mismatched empirical null. Env: eugene_tools."""
import glob, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def load(sub, res):
    fs = glob.glob(f"{res}/{sub}/summary__*.tsv")
    return pd.concat([pd.read_csv(f, sep="\t") for f in fs if __import__("os").path.getsize(f) > 0], ignore_index=True) if fs else pd.DataFrame()

def main(a):
    full = load("full", a.res); null = load("null", a.res)
    tfmap = pd.read_csv(a.tfmap, sep="\t").dropna(subset=["curator_tf"]).set_index("short_id")["curator_tf"].to_dict()
    full["pair"] = full["idA"].map(lambda s: tfmap.get(s, s)) + "×" + full["idB"].map(lambda s: tfmap.get(s, s))
    d_thr = max(0.15, float(np.nanpercentile(null["delta"], 95)))
    z_thr = max(4.0, float(np.nanpercentile(null["maxZ"], 95)))
    full["syn"] = (full["delta"] > d_thr) & (full["wilcoxon_p"] < 0.001)
    full["hard"] = (full["maxZ"] > z_thr) & full["syn"]
    full["soft_cal"] = full["soft"] & full["syn"] & ~full["hard"]
    full.to_csv(f"{a.fig}/full_calls.tsv", sep="\t", index=False)

    # Fig-4i scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = [np.nanmin(full.dS) - .2, np.nanmax(full.dJ_opt) + .2]
    ax.plot(lim, lim, "k--", lw=.8, zorder=1, label="additive")
    ax.scatter(full.dS, full.dJ_opt, s=8, c="#b0bec5", alpha=.5, label="not synergistic", zorder=2)
    s = full[full.syn]; ax.scatter(s.dS, s.dJ_opt, s=14, c="#e67e22", alpha=.8, label=f"synergistic (Δ>{d_thr:.2f})", zorder=3)
    h = full[full.hard]; ax.scatter(h.dS, h.dJ_opt, s=20, c="#c0392b", edgecolor="k", lw=.3, label=f"hard (Z>{z_thr:.1f})", zorder=4)
    ax.set_xlabel("Σ independent effects  ΔS = ΔA+ΔB"); ax.set_ylabel("joint effect ΔJ (optimal arrangement)")
    ax.set_title(f"All keep-set pairs (n={len(full)}, primary CT)\nnull-calibrated synergy")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{a.fig}/fig_full_scatter.png", dpi=150); plt.close(fig)

    # Fig-4j census
    cats = {"hard": int(full.hard.sum()), "soft": int(full.soft_cal.sum()),
            "syn (no syntax pref)": int((full.syn & ~full.hard & ~full.soft_cal).sum()),
            "none": int((~full.syn).sum())}
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.bar(cats.keys(), cats.values(), color=["#c0392b", "#e67e22", "#95a5a6", "#dfe6e9"], edgecolor="k")
    for i, v in enumerate(cats.values()): ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("# motif pairs"); ax.set_title(f"Syntax census — {len(full)} pairs")
    plt.xticks(rotation=20, ha="right"); fig.tight_layout(); fig.savefig(f"{a.fig}/fig_full_census.png", dpi=150); plt.close(fig)

    # Fig-4h spacing histogram (hard-syntax preferred center-to-center)
    hard = full[full.hard]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(hard.opt_center_center.dropna(), bins=range(0, 80, 3), color="#c0392b", edgecolor="k")
    ax.set_xlabel("center-to-center distance at optimal (bp)"); ax.set_ylabel("# hard-syntax pairs")
    ax.set_title(f"Preferred spacing of hard-syntax pairs (n={len(hard)})")
    fig.tight_layout(); fig.savefig(f"{a.fig}/fig_full_spacing.png", dpi=150); plt.close(fig)

    # top synergistic table
    top = full.sort_values("delta", ascending=False).head(60)[
        ["pair", "ct", "dJ_opt", "dS", "delta", "opt_orient", "opt_gap", "opt_center_center", "maxZ", "hard", "soft_cal", "wilcoxon_p"]]
    top.to_csv(f"{a.fig}/full_top_synergy.tsv", sep="\t", index=False)
    print(f"FULL: {len(full)} pairs | thresholds Δ>{d_thr:.3f} Z>{z_thr:.2f}")
    print(f"  synergistic: {int(full.syn.sum())} ({100*full.syn.mean():.0f}%) | hard: {cats['hard']} | soft: {cats['soft']}")
    print("  top synergy:"); print(top.head(15).to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--res", required=True); p.add_argument("--fig", required=True)
    p.add_argument("--tfmap", required=True); main(p.parse_args())