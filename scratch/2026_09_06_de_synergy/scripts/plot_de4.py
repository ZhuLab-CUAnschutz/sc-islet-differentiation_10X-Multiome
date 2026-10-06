#!/usr/bin/env python
"""Figures for the DE 4-TF synergy run. Reads tables/ + results/de4/*.npz. PNG + PDF into figures/."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); WD = os.path.join(HERE, "..")
RES, TAB, FIG = (os.path.join(WD, d) for d in ("results/de4", "tables", "figures")); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left"})
SETC = {"set1_catalog+JFOXH1": "#5e4fa2", "set2_jaspar": "#f4a582"}      # purple / DE orange (config color)
SETL = {"set1_catalog+JFOXH1": "catalog CWM (+ JASPAR FOXH1)", "set2_jaspar": "JASPAR"}
ORIC = {"FF": "#1f77b4", "FR": "#2ca02c", "RF": "#ff7f0e", "RR": "#d62728"}
long = pd.read_csv(os.path.join(TAB, "calls_long.tsv"), sep="\t")
thr = {l.split(":")[0]: l for l in open(os.path.join(TAB, "null_thresholds.txt"))}
d_thr = float(thr["applied"].split("delta>")[1].split(" ")[0])
order = ["EOMES×MIXL1", "EOMES×SOX17", "MIXL1×SOX17", "EOMES×FOXH1", "MIXL1×FOXH1", "SOX17×FOXH1"]
sets = list(SETC)
het = long[long.kind == "heterotypic"].copy(); het["pair"] = pd.Categorical(het.pair, order)
def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ---- fig1: delta per pair, two bars per pair ----
fig, ax = plt.subplots(figsize=(7.5, 3.6)); x = np.arange(len(order)); w = 0.36
for i, s in enumerate(sets):
    d = het[het.set == s].set_index("pair").reindex(order)
    ax.bar(x + (i - .5) * w, d.delta, w * 0.94, color=SETC[s], label=SETL[s], edgecolor="none")
    for xi, (dv, call) in enumerate(zip(d.delta, d.call)):
        if call != "none": ax.text(x[xi] + (i - .5) * w, dv + 0.01, call, ha="center", va="bottom", fontsize=7, color="#333")
ax.axhline(d_thr, color="#888", lw=1, ls="--"); ax.text(len(order) - .5, d_thr + .005, f"DE null 95th pct Δ={d_thr:.2f}", ha="right", va="bottom", fontsize=7, color="#666")
ax.axhline(0, color="#333", lw=.6); ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("synergy Δ = ΔJ(opt) − ΔS  (log counts)"); ax.set_title("Pairwise synergy in DE, catalog vs JASPAR inserts (n=100 backgrounds, gaps 0–200)")
ax.legend(frameon=False, fontsize=8); save(fig, "fig1_delta_by_pair")

# ---- fig2: distance curves grid (rows = pairs, cols = sets) ----
fig, axes = plt.subplots(len(order), 2, figsize=(9.5, 2.1 * len(order)), sharex=True)
for r, pair in enumerate(order):
    for c, s in enumerate(sets):
        ax = axes[r, c]; row = het[(het.pair == pair) & (het.set == s)].iloc[0]
        z = np.load(os.path.join(RES, f"{row.idA}__{row.idB}__DE.npz")); gaps = z["gaps"]
        for oi, on in enumerate(z["orients"]):
            ax.plot(gaps, z["dJ"][oi], lw=1.2, color=ORIC[str(on)], label=str(on))
        dS = float(z["dA"] + z["dB"]); ax.axhline(dS, color="#333", lw=.8, ls="--")
        ax.axhspan(dS, dS + d_thr, color="#bbb", alpha=.25, lw=0)
        ax.plot(row.opt_gap, row.dJ_opt, "o", ms=5, mfc="none", mec="k", mew=1)
        top = max(dS + d_thr * 1.15, np.nanmax(z["dJ"]) * 1.15, 0.1); ax.set_ylim(min(np.nanmin(z["dJ"]), 0) - 0.02, top)
        ax.text(200, dS + d_thr, "ΔS + null p95", ha="right", va="bottom", fontsize=6, color="#888")
        ax.set_title(f"{pair}   Δ={row.delta:+.2f}  {row.opt_orient} @ {int(row.opt_gap)} bp   Z={row.maxZ:.1f}   [{row.call}]", fontsize=8)
        if r == 0: ax.text(0.5, 1.22, SETL[s], transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold", color=SETC[s])
        if c == 0: ax.set_ylabel("ΔJ (log counts)")
        if r == 0 and c == 0: ax.legend(frameon=False, fontsize=7, ncol=4, loc="center right")
for ax in axes[-1]: ax.set_xlabel("inner-edge gap (bp)")
fig.suptitle("Joint effect ΔJ vs spacing per orientation; dashed = additive ΔS; grey = up to the DE-null threshold; circle = optimum", y=1.01, fontsize=10)
fig.tight_layout(); save(fig, "fig2_distance_curves")

# ---- fig3: single-motif effect per insert ----
ins = pd.read_csv(os.path.join(TAB, "inserts.tsv"), sep="\t")
ins["src"] = np.where(ins.id.str.startswith("J_"), "JASPAR", "catalog"); ins["tf"] = pd.Categorical(ins.tf, ["EOMES", "MIXL1", "SOX17", "FOXH1"]); ins = ins.sort_values(["tf", "src"])
fig, ax = plt.subplots(figsize=(6.5, 3.2)); x = np.arange(4); w = .36
for i, src in enumerate(["catalog", "JASPAR"]):
    d = ins[ins.src == src].set_index("tf").reindex(["EOMES", "MIXL1", "SOX17", "FOXH1"])
    col = SETC[sets[0]] if src == "catalog" else SETC[sets[1]]
    ax.bar(x + (i - .5) * w, d.dA_DE.fillna(0), w * .94, color=col, label=src, edgecolor="none")
    for xi, (v, cons) in enumerate(zip(d.dA_DE, d.consensus)):
        if pd.notna(v): ax.text(x[xi] + (i - .5) * w, v + .01, cons, ha="center", va="bottom", fontsize=6, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(["EOMES", "MIXL1", "SOX17", "FOXH1"]); ax.axhline(0, color="#333", lw=.6)
ax.set_ylabel("single-insert ΔA in DE (log counts)"); ax.set_title("Each insert alone in the DE model"); ax.legend(frameon=False, fontsize=8)
ax.set_ylim(top=max(ins.dA_DE.max() * 1.6, 0.1)); save(fig, "fig3_inserts")

# ---- fig4: homodimers ----
homo = long[long.kind == "homodimer"].drop_duplicates(["idA", "idB"]).copy()
homo["src"] = np.where(homo.idA.str.startswith("J_"), "JASPAR", "catalog"); homo["tf"] = pd.Categorical(homo.tfA, ["EOMES", "MIXL1", "SOX17", "FOXH1"])
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for i, src in enumerate(["catalog", "JASPAR"]):
    d = homo[homo.src == src].set_index("tf").reindex(["EOMES", "MIXL1", "SOX17", "FOXH1"])
    col = SETC[sets[0]] if src == "catalog" else SETC[sets[1]]
    ax.bar(x + (i - .5) * w, d.delta.fillna(0), w * .94, color=col, label=src, edgecolor="none")
    for xi, (v, call) in enumerate(zip(d.delta, d.call)):
        if pd.notna(v) and call != "none": ax.text(x[xi] + (i - .5) * w, v + .01, call, ha="center", va="bottom", fontsize=7)
ax.axhline(d_thr, color="#888", lw=1, ls="--"); ax.axhline(0, color="#333", lw=.6)
ax.set_xticks(x); ax.set_xticklabels(["EOMES×EOMES", "MIXL1×MIXL1", "SOX17×SOX17", "FOXH1×FOXH1"])
ax.set_ylabel("homodimer synergy Δ"); ax.set_title("Homotypic pairs (A×A) in DE"); ax.legend(frameon=False, fontsize=8); save(fig, "fig4_homodimers")
