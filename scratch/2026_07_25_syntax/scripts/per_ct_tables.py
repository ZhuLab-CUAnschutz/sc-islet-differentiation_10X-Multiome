#!/usr/bin/env python
"""Per-cell-type metadata tables for the synergistic pairs. For each of the 22 cell types, one TSV with
every synergistic pair (582), that cell type's re-optimized synergy metrics (syn_allct), whether the
pair is synergistic *in that CT*, plus full pair/motif metadata (TFs, categories, families, TomTom,
DeepLIFT, primary CT). Also a combined long table. CPU only."""
import glob, os
import numpy as np, pandas as pd

SYN = os.path.dirname(os.path.abspath(__file__)); OUT = f"{SYN}/figures/per_ct_tables"
os.makedirs(OUT, exist_ok=True)

sa = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{SYN}/RES/syn_allct/summary__*.tsv")], ignore_index=True)
meta = pd.read_csv(f"{SYN}/metadata.tsv", sep="\t").set_index("short_id")
# pair-level metadata (per motif A/B)
def mcol(idx, col): return meta[col].get(idx, "") if col in meta.columns else ""
def motif_meta(mid, sfx):
    return {f"tf_{sfx}": mcol(mid, "curator_tf"), f"category_{sfx}": mcol(mid, "curator_category"),
            f"family_{sfx}": mcol(mid, "curator_family"), f"best_match_{sfx}": mcol(mid, "best_match_tf"),
            f"tf_candidates_{sfx}": mcol(mid, "tf_candidates")}
# primary-CT calls (which CT each pair was optimized in) + DeepLIFT
prim = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{SYN}/RES/full/summary__*.tsv")+glob.glob(f"{SYN}/RES/subs/summary__*.tsv")], ignore_index=True).drop_duplicates(["idA","idB"], keep="last")
prim_map = {(r.idA, r.idB): (r.ct, round(float(r.delta),3)) for r in prim.itertuples()}
dl = {}
for f in glob.glob(f"{SYN}/RES/deeplift/deeplift__*.tsv"):
    for r in pd.read_csv(f, sep="\t").itertuples():
        if pd.notna(getattr(r, "frac_in", None)): dl[(r.idA, r.idB)] = round(float(r.frac_in), 3)
# MT null-independent threshold from single-task null (for "synergistic in this CT")
nul = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{SYN}/RES/null/summary__*.tsv")], ignore_index=True)
D_THR = max(0.15, float(np.nanpercentile(nul.delta, 95))); Z_THR = max(4.0, float(np.nanpercentile(nul.maxZ, 95)))

rows = []
for r in sa.itertuples():
    prim_ct, prim_delta = prim_map.get((r.idA, r.idB), ("", np.nan))
    d = {"idA": r.idA, "idB": r.idB, "tfA": mcol(r.idA, "curator_tf"), "tfB": mcol(r.idB, "curator_tf"),
         "cell_type": r.ct,
         "dJ_opt": r.dJ_opt, "dS": r.dS, "delta": r.delta, "opt_orient": r.opt_orient, "opt_gap": r.opt_gap,
         "opt_center_center": r.opt_center_center, "maxZ": r.maxZ, "wilcoxon_p": r.wilcoxon_p,
         "synergistic_in_ct": bool((r.delta > D_THR) and (r.wilcoxon_p < 0.001)),
         "hard_in_ct": bool((r.maxZ > Z_THR) and (r.delta > D_THR) and (r.wilcoxon_p < 0.001)),
         "primary_ct": prim_ct, "primary_delta": prim_delta,
         "deeplift_frac_in": dl.get((r.idA, r.idB), np.nan),
         "deeplift_flag": ("artifact?" if dl.get((r.idA, r.idB), 1) < 0.4 else "ok"),
         "wa": r.wa, "wb": r.wb}
    d.update(motif_meta(r.idA, "A")); d.update(motif_meta(r.idB, "B"))
    rows.append(d)
long = pd.DataFrame(rows)
cols = ["idA","idB","tfA","tfB","cell_type","dJ_opt","dS","delta","opt_orient","opt_gap","opt_center_center",
        "maxZ","wilcoxon_p","synergistic_in_ct","hard_in_ct","primary_ct","primary_delta","deeplift_frac_in",
        "deeplift_flag","wa","wb","category_A","category_B","family_A","family_B","best_match_A","best_match_B",
        "tf_candidates_A","tf_candidates_B"]
long = long[[c for c in cols if c in long.columns]]
long.to_csv(f"{SYN}/figures/synergistic_pairs_allCT_long.tsv", sep="\t", index=False)

ctm = pd.read_csv(f"{SYN}/cell_type_metadata.tsv", sep="\t").sort_values("display_order")
for ct in ctm.cell_type:
    sub = long[long.cell_type == ct].sort_values("delta", ascending=False)
    if len(sub): sub.to_csv(f"{OUT}/synergy__{ct}.tsv", sep="\t", index=False)
print(f"thresholds: Δ>{D_THR:.3f} Z>{Z_THR:.2f}")
print(f"wrote long table ({len(long)} rows) + {len(glob.glob(OUT+'/*.tsv'))} per-CT tables")
print("per-CT synergistic counts:")
print(long[long.synergistic_in_ct].groupby("cell_type").size().reindex(ctm.cell_type).fillna(0).astype(int).to_string())
