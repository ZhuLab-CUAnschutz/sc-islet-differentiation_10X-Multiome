"""Phase 7.3 — render a Fig-3e-style family-rollup HTML for the MC cluster catalog.

Conceptually matches Liu et al. 2026 Fig 3e: one row per BROAD FAMILY GROUP,
aggregating across the cluster_final motifs that share a primary TF family.

Columns (left -> right):
  1. CWM logo            — representative member (highest total_seqlets) modisco CWM
  2. Vierstra PWM logo   — primary_subname's archetype PWM
  3. Direction           — pos / neg
  4. Total instances     — sum across members across 22 CTs
  5. Region stacked bar  — promoter / 5UTR / 3UTR / exon / intron / distal
  6. TSS distance box    — per-CT median |TSS distance|, boxed across 22 CTs
  7. Compartment bars    — three stacked bars:
                            (a) `grouping` lineage (11 bins)
                            (b) Endocrine vs Non-endocrine binary
                            (c) Coarse stage (DE / Foregut / PP / Early endocrine /
                                              Late endocrine / Non-pancreatic)
  8. n_variants          — count of cluster_final members rolled into this family

Inputs (auto-located under `--catalog_dir`):
  qc/motif_classification.tsv
  08_final_averages_metadata.tsv
  clustered_motifs.modisco.h5
  combined_mapped.meme              (also used to render flank-aware CWM logo)
  annotation/per_motif_summary.tsv
  annotation/per_motif_per_ct.tsv
  hits/hit_count_matrix.csv

Cell-type metadata + palette: config/cell_type_metadata.tsv (`grouping` column).

Output: {out_dir}/fig3e_summary.html
"""
import argparse, base64, io, re, html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py


# ---------- compartment mappings ----------
# Built from config/cell_type_metadata.tsv `grouping` column, config/stage_metadata.tsv,
# and config/endocrine_metadata.tsv (single source of truth).
# Cell-type -> differentiation stage (D-day) per doc/EXPERIMENTAL_DESIGN.md table.
STAGE_BY_CT = {
    "DE":                        "D4",
    "PGT1":                      "D7",  "PGT2": "D7", "PGT3": "D7",
    "PFG1":                      "D9",  "PFG2": "D9",
    "PP1":                       "D9",  "PP2":  "D9",
    "ENP_phase1":                "D15",
    "early_ENP":                 "D12", "late_ENP": "D15",
    "proliferating_endocrine":   "D15",
    "early_SC_alpha":            "D22", "early_SC_beta": "D22", "early_SC_EC": "D22",
    "late_SC_alpha":             "D45", "late_SC_beta":  "D45", "late_SC_EC":  "D45",
    "SC_delta_GHRL":             "D45",
    "exocrine":                  "D45", "liver":         "D45", "FB_FLT1":     "D45",
}
STAGE_ORDER = ["D4", "D7", "D9", "D12", "D15", "D22", "D45"]
STAGE_COLORS = {  # config/stage_metadata.tsv
    "D4": "#fef0d9", "D7": "#fdcc8a", "D9":  "#fc8d59", "D12": "#e34a33",
    "D15": "#b30000", "D22": "#7f0000", "D45": "#4a0000",
}

# Endocrine vs non-endocrine: derived from `grouping` membership.
ENDOCRINE_GROUPINGS = {"endocrine_progenitor", "proliferating", "SC_beta",
                       "SC_alpha", "SC_EC", "SC_delta"}
ENDO_ORDER = ["endocrine", "non_endocrine"]
ENDO_COLORS = {"endocrine": "#5e4fa2", "non_endocrine": "#f46d43"}  # config/endocrine_metadata.tsv

# Region (genomic feature) palette. Liu 2026 Fig 3e uses categorical contrast:
# promoter+UTRs in warm hues, gene body (exon/intron) in cool, distal grey.
REGION_ORDER = ["promoter", "5UTR", "exon", "intron", "3UTR", "distal"]
REGION_COLORS = {
    "promoter": "#d7301f",  # bright red — most visually prominent (paper convention)
    "5UTR":     "#fdae61",  # orange
    "exon":     "#1a9850",  # green
    "intron":   "#74add1",  # light blue
    "3UTR":     "#fdcc8a",  # pale orange
    "distal":   "#bababa",  # neutral grey
}
DIRECTION_COLORS = {"pos": "#2c7bb6", "neg": "#d7191c"}

# Paper-style "most similar motif": per-motif TomTom hit at E <= 10 (Liu 2026 Methods).
TOMTOM_E_THRESH_PAPER = 10.0


_VIERSTRA_LOOSE_DF = {}  # catalog_dir -> grouped DataFrame, cached for re-render speed


def _load_vierstra_loose(catalog_dir: Path) -> dict:
    """Cache + return {Query_ID -> sub-DataFrame} from the 7.1 loose-E TomTom run."""
    key = str(catalog_dir)
    if key in _VIERSTRA_LOOSE_DF:
        return _VIERSTRA_LOOSE_DF[key]
    fp = Path(catalog_dir) / "tomtom" / "vierstra_loose" / "tomtom.tsv"
    if not fp.exists():
        _VIERSTRA_LOOSE_DF[key] = {}
        return {}
    # Comment-char manual strip (tomtom embeds '#' in cluster_final#N motif names)
    raw = [ln for ln in fp.read_text().splitlines()
           if ln.strip() and not ln.lstrip().startswith("#")]
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(raw)), sep="\t")
    if "Query_ID" not in df.columns or "E-value" not in df.columns:
        _VIERSTRA_LOOSE_DF[key] = {}
        return {}
    by_q = {q: sub for q, sub in df.groupby("Query_ID")}
    _VIERSTRA_LOOSE_DF[key] = by_q
    return by_q


def _paper_top_hit(catalog_dir: Path, motif_display: str, h5_key: str,
                   evalue: float = TOMTOM_E_THRESH_PAPER):
    """Liu 2026-style "most similar motif": top TomTom hit at E<=`evalue` against
    Vierstra archetypes. Returns (target_id, e-value, orientation) where
    orientation is "+" or "-" so caller can RC the PWM to align with the CWM.

    Source: aggregate TSV from 7.1's `vierstra_loose` run.
    """
    by_q = _load_vierstra_loose(catalog_dir)
    sub = by_q.get(motif_display)
    if sub is None or sub.empty:
        return None, None, None
    keep = sub[sub["E-value"] <= evalue].sort_values("E-value")
    if keep.empty:
        return None, None, None
    row = keep.iloc[0]
    orient = str(row.get("Orientation", "+")) if "Orientation" in keep.columns else "+"
    return str(row["Target_ID"]), float(row["E-value"]), orient


def _rc_ppm(ppm: np.ndarray) -> np.ndarray:
    """Reverse-complement a PWM: reverse positions, swap A<->T and C<->G."""
    return ppm[::-1, [3, 2, 1, 0]]


# ---------- helpers ----------

def b64_png(fig, dpi=120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_meme_to_ppms(meme_text: str) -> dict[str, np.ndarray]:
    """Return name -> PWM array (W,4)."""
    out = {}
    cur_name, cur_rows = None, None
    for ln in meme_text.splitlines():
        ln = ln.rstrip()
        if ln.startswith("MOTIF"):
            if cur_name is not None and cur_rows:
                out[cur_name] = np.array(cur_rows)
            cur_name = ln.split()[1]
            cur_rows = []
        elif ln.startswith("letter-probability") or ln.startswith("URL"):
            continue
        else:
            parts = ln.strip().split()
            if cur_name is not None and len(parts) == 4:
                try:
                    cur_rows.append([float(x) for x in parts])
                except ValueError:
                    pass
    if cur_name is not None and cur_rows:
        out[cur_name] = np.array(cur_rows)
    return out


def render_pwm_logo(ppm: np.ndarray, width_px: int = 200, height_px: int = 60) -> str:
    """logomaker-style PWM logo as base64 PNG."""
    import logomaker
    df = pd.DataFrame(ppm, columns=list("ACGT"))
    info = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    logomaker.Logo(info, ax=ax, show_spines=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0, 2)
    return b64_png(fig, dpi=100)


def render_cwm_logo(cwm: np.ndarray, width_px: int = 200, height_px: int = 60) -> str:
    """Sign-aware CWM logo (positive bars up, negative bars down) via logomaker."""
    import logomaker
    df = pd.DataFrame(cwm, columns=list("ACGT"))
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    logomaker.Logo(df, ax=ax, show_spines=False, center_values=False)
    ax.set_xticks([]); ax.set_yticks([])
    return b64_png(fig, dpi=100)


def render_stacked_bar(props: dict[str, float], order: list[str], colors: dict[str, str],
                       width_px: int = 180, height_px: int = 18, total: float | None = None,
                       text: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=120)
    left = 0.0
    for key in order:
        v = float(props.get(key, 0) or 0)
        if v > 0:
            ax.barh(0, v, left=left, color=colors.get(key, "#cccccc"), edgecolor="white", linewidth=0.5)
            left += v
    ax.set_xlim(0, total if total else max(left, 1))
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    if text:
        ax.text(0.5 * (total or left), 0, text, ha="center", va="center", fontsize=6, color="white")
    return b64_png(fig, dpi=120)


def render_box(values: np.ndarray, width_px: int = 200, height_px: int = 60,
               vmax: float | None = None) -> str:
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=120)
    values = values[~np.isnan(values)]
    if len(values):
        ax.boxplot(values, vert=False, widths=0.55, patch_artist=True,
                   boxprops=dict(facecolor="#cccccc", edgecolor="#444"),
                   medianprops=dict(color="#d62728"),
                   whiskerprops=dict(color="#666"),
                   capprops=dict(color="#666"),
                   flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_yticks([])
    ax.set_xscale("log")
    if vmax is None:
        vmax = max(1e5, float(values.max()) if len(values) else 1e5)
    ax.set_xlim(50, vmax * 1.2)
    ax.tick_params(axis="x", labelsize=6)
    return b64_png(fig, dpi=120)


# ---------- aggregation ----------

def family_of(row) -> str:
    """Pick the family label for grouping."""
    cat = row["category"]
    if cat == "unresolved":
        return "Unresolved"
    if cat == "repeat":
        return "Repeat"
    fam = row.get("primary_family", "")
    if not fam or pd.isna(fam):
        return "Unresolved"
    # composite motifs are grouped by their primary anchor family
    return fam.split(",")[0].strip().upper() or "Unresolved"


def build_family_table(classify: pd.DataFrame, meta: pd.DataFrame,
                       per_motif: pd.DataFrame, per_motif_ct: pd.DataFrame,
                       hit_count_matrix: pd.DataFrame,
                       ct_meta: pd.DataFrame,
                       catalog_dir: Path, name_map: pd.DataFrame) -> pd.DataFrame:
    classify["family"] = classify.apply(family_of, axis=1)
    md = meta.merge(classify[["name", "family", "category", "primary_family",
                              "primary_subname", "primary_evalue"]],
                    on="name", how="left")
    md = md.merge(per_motif.rename(columns={"motif_name": "name"}), on="name", how="left")
    md["family"] = md["family"].fillna("Unresolved")

    # Convert hit_count_matrix (rows=CT, cols=motif) to a per-CT-per-motif long df
    hcm = hit_count_matrix.copy()
    # CT name is in the index or 'cell_type' col — accept either
    if "cell_type" in hcm.columns:
        hcm = hcm.set_index("cell_type")
    hcm.index.name = "celltype"
    long_hcm = hcm.reset_index().melt(id_vars="celltype", var_name="name", value_name="hits")

    # Per-family aggregation
    groupings = {}
    for ct in hcm.index:
        groupings[ct] = ct_meta.loc[ct, "grouping"]

    out_rows = []
    fam_cluster_members = md.groupby("family")
    for fam, sub in fam_cluster_members:
        # representative: highest total_seqlets
        sub = sub.sort_values("total_seqlets", ascending=False)
        rep = sub.iloc[0]
        n_variants = len(sub)
        member_names = sub["name"].tolist()
        # direction: majority sign among members
        pos_n = int((sub["posneg"] == "pos").sum())
        neg_n = int((sub["posneg"] == "neg").sum())
        direction = "pos" if pos_n >= neg_n else "neg"
        # "Most similar known motif" per Liu 2026: per-motif TomTom top hit at E<=10
        # against Vierstra archetypes (overrides the looser E<=0.1 7.1 primary_subname).
        # Capture orientation so the PWM can be reverse-complemented to align with CWM.
        h5_key_rep = name_map.loc[name_map["display"] == rep["name"], "h5_key"]
        h5_key_rep = h5_key_rep.iloc[0] if len(h5_key_rep) else ""
        paper_target, paper_eval, paper_orient = _paper_top_hit(catalog_dir, rep["name"], h5_key_rep)
        primary_sub = paper_target or ""

        # total hits across this family's members
        fam_hcm = long_hcm[long_hcm["name"].isin(member_names)]
        total_hits = int(fam_hcm["hits"].sum())
        per_ct_hits = fam_hcm.groupby("celltype")["hits"].sum()
        n_ct = int((per_ct_hits > 0).sum())

        # region proportions: hit-weighted across member motifs
        mem_summary = per_motif[per_motif["motif_name"].isin(member_names)]
        if total_hits and len(mem_summary):
            w = mem_summary["total_hits"].astype(float)
            region_props = {}
            for r in REGION_ORDER:
                col = f"pct_{r}"
                if col in mem_summary.columns and w.sum():
                    region_props[r] = float((mem_summary[col] * w).sum() / w.sum())
                else:
                    region_props[r] = 0.0
        else:
            region_props = {r: 0.0 for r in REGION_ORDER}

        # per-CT median TSS dist across family members: weighted average
        mem_ct = per_motif_ct[per_motif_ct["motif_name"].isin(member_names)]
        tss_values_per_ct = []
        for ct, sub_ct in mem_ct.groupby("celltype"):
            w_ct = sub_ct["n_hits"].astype(float)
            if w_ct.sum() > 0:
                med = float((sub_ct["median_tss_distance_abs"] * w_ct).sum() / w_ct.sum())
                tss_values_per_ct.append(med)
        tss_values_per_ct = np.array(tss_values_per_ct, dtype=float)

        # compartment props
        lineage_props = {}
        endo_props = {"endocrine": 0, "non_endocrine": 0}
        stage_props = {s: 0 for s in STAGE_ORDER}
        for ct, hits in per_ct_hits.items():
            if hits <= 0: continue
            grp = groupings.get(ct, "non_pancreatic")
            lineage_props[grp] = lineage_props.get(grp, 0) + hits
            endo_props["endocrine" if grp in ENDOCRINE_GROUPINGS else "non_endocrine"] += hits
            stage_key = STAGE_BY_CT.get(ct, "D45")
            stage_props[stage_key] = stage_props.get(stage_key, 0) + hits

        out_rows.append({
            "family": fam,
            "rep_motif": rep["name"],
            "direction": direction,
            "n_variants": n_variants,
            "n_ct_with_hits": n_ct,
            "total_hits": total_hits,
            "primary_subname": primary_sub,
            "primary_orientation": paper_orient or "+",
            "member_names": ",".join(member_names),
            "region_props": region_props,
            "tss_values_per_ct": tss_values_per_ct.tolist(),
            "lineage_props": lineage_props,
            "endo_props": endo_props,
            "stage_props": stage_props,
        })

    # Sort like the Liu 2026 paper: pos first, neg second, each section ordered
    # by ascending median |TSS distance| so promoter-proximal motifs sit at the
    # top. Unresolved + Repeat tucked at the very end.
    df = pd.DataFrame(out_rows)

    def _median_tss(r):
        v = np.asarray(r["tss_values_per_ct"], dtype=float)
        v = v[~np.isnan(v)]
        return float(np.median(v)) if len(v) else float("inf")

    df["sort_tss"] = df.apply(_median_tss, axis=1)

    def _rank(r):
        f = r["family"]
        if f == "Unresolved": return (3, r["sort_tss"])
        if f == "Repeat":     return (4, r["sort_tss"])
        return (0 if r["direction"] == "pos" else 1, r["sort_tss"])
    df["rank"] = df.apply(_rank, axis=1)
    df = df.sort_values("rank").reset_index(drop=True).drop(columns=["rank", "sort_tss"])
    return df


# ---------- render ----------

CSS = """
body { font-family: -apple-system, sans-serif; padding: 22px; }
h1   { margin: 0 0 10px 0; font-size: 22px; }
.meta { color: #555; font-size: 14px; margin-bottom: 18px; }
table { border-collapse: collapse; }
th   { background: #f5f5f5; padding: 8px 10px; text-align: left; font-size: 13px;
       border-bottom: 2px solid #999; position: sticky; top: 0; }
td   { padding: 7px 10px; border-bottom: 1px solid #eee; font-size: 14px; vertical-align: middle; }
.fam_name { font-weight: 700; font-size: 16px; }
.fam_sub  { color: #555; font-size: 13px; }
.dir_pos { color: #2c7bb6; font-weight: 700; font-size: 16px; }
.dir_neg { color: #d7191c; font-weight: 700; font-size: 16px; }
.num    { text-align: right; font-variant-numeric: tabular-nums; }
img.logo { display: block; }
.legend { margin: 8px 0 10px 0; font-size: 15px; line-height: 1.8; }
.legend b { font-size: 15px; }
.swatch { display: inline-block; width: 14px; height: 14px; margin: 0 5px 0 10px;
          vertical-align: middle; border: 1px solid #999; border-radius: 2px; }
.legend .swatch:first-of-type { margin-left: 6px; }
"""


def render_html(table: pd.DataFrame, h5: h5py.File, ppms: dict[str, np.ndarray],
                ct_meta: pd.DataFrame, out_html: Path):
    # legends
    def legend(order, colors, label):
        chips = " ".join(
            f"<span class='swatch' style='background:{colors[k]}'></span>{html.escape(k)}"
            for k in order
        )
        return f"<div class='legend'><b>{label}:</b> {chips}</div>"

    lineage_order = ct_meta.sort_values("display_order")["grouping"].drop_duplicates().tolist()
    lineage_palette = (ct_meta.drop_duplicates("grouping").set_index("grouping")["color"]).to_dict()

    # Column layout: family label LEFT (Fig 3e convention), then CWM / PWM / ...
    head = [
        ("Family (n CT)", 180),
        ("De novo motif", 220),
        ("Most similar known motif", 220),
        ("Direction", 60),
        ("Total instances", 110),
        ("Genomic features", 200),
        ("Median distance to TSS (log bp)", 220),
        ("Tissue compartment", 200),
        ("Endocrine", 110),
        ("Stage", 200),
        ("Variants", 70),
    ]

    rows_html = []
    for _, r in table.iterrows():
        # Family label (left, sticky-ish)
        fam_label = (f"<span class='fam_name'>{html.escape(r['family'])}</span>"
                     f" <span class='fam_sub'>({r['n_ct_with_hits']})</span>")

        # CWM logo
        rep = r["rep_motif"]
        m = re.search(r"cluster_final#(\d+)", rep)
        cf = f"cluster_final#{m.group(1)}" if m else None
        cwm_img = ""
        for ptn in ["pos_patterns", "neg_patterns"]:
            if ptn in h5 and cf and cf in h5[ptn]:
                cwm = np.asarray(h5[ptn][cf]["contrib_scores"])
                cwm_img = "<img class='logo' src='data:image/png;base64," + render_cwm_logo(cwm) + "'/>"
                break
        if not cwm_img and rep in ppms:
            cwm_img = "<img class='logo' src='data:image/png;base64," + render_pwm_logo(ppms[rep]) + "'/>"

        # PWM (Vierstra archetype). Orient to align with CWM: RC if TomTom said "-".
        ppm_img = ""
        sub = r["primary_subname"]
        orient = r.get("primary_orientation", "+")
        if sub and sub in ppms:
            ppm = ppms[sub]
            if orient == "-":
                ppm = _rc_ppm(ppm)
            ppm_img = (
                "<img class='logo' src='data:image/png;base64,"
                + render_pwm_logo(ppm) + "'/>"
                + f"<span class='fam_sub'>{html.escape(sub)}{' (RC)' if orient == '-' else ''}</span>"
            )
        elif sub:
            ppm_img = f"<span class='fam_sub'>{html.escape(str(sub))} (no PWM)</span>"
        else:
            ppm_img = "<span class='fam_sub'>—</span>"

        # Direction
        dir_str = ("<span class='dir_pos'>+</span>" if r["direction"] == "pos"
                   else "<span class='dir_neg'>−</span>")

        # Region bar
        region_img = render_stacked_bar(r["region_props"], REGION_ORDER, REGION_COLORS,
                                        width_px=180, height_px=16, total=100)
        region_html = "<img class='logo' src='data:image/png;base64," + region_img + "'/>"

        # TSS box
        tss_vals = np.asarray(r["tss_values_per_ct"], dtype=float)
        if len(tss_vals):
            tss_html = "<img class='logo' src='data:image/png;base64," + render_box(tss_vals, 200, 56) + "'/>"
        else:
            tss_html = "<span class='fam_sub'>—</span>"

        # Lineage / Endocrine / Stage bars (raw hit counts; total = total_hits)
        lin_img = render_stacked_bar(r["lineage_props"], lineage_order, lineage_palette,
                                     width_px=180, height_px=16, total=r["total_hits"] or 1)
        lin_html = "<img class='logo' src='data:image/png;base64," + lin_img + "'/>"

        endo_img = render_stacked_bar(r["endo_props"], ENDO_ORDER, ENDO_COLORS,
                                      width_px=100, height_px=16, total=r["total_hits"] or 1)
        endo_html = "<img class='logo' src='data:image/png;base64," + endo_img + "'/>"

        stage_img = render_stacked_bar(r["stage_props"], STAGE_ORDER, STAGE_COLORS,
                                       width_px=180, height_px=16, total=r["total_hits"] or 1)
        stage_html = "<img class='logo' src='data:image/png;base64," + stage_img + "'/>"

        rows_html.append(
            "<tr>"
            f"<td>{fam_label}</td>"
            f"<td>{cwm_img}</td>"
            f"<td>{ppm_img}</td>"
            f"<td>{dir_str}</td>"
            f"<td class='num'>{r['total_hits']:,}</td>"
            f"<td>{region_html}</td>"
            f"<td>{tss_html}</td>"
            f"<td>{lin_html}</td>"
            f"<td>{endo_html}</td>"
            f"<td>{stage_html}</td>"
            f"<td class='num'>{r['n_variants']}</td>"
            "</tr>"
        )

    header_html = "".join(
        f"<th style='width:{w}px'>{html.escape(h)}</th>" for h, w in head
    )

    legends_html = "\n".join([
        legend(REGION_ORDER, REGION_COLORS, "Genomic features"),
        legend(lineage_order, lineage_palette, "Tissue compartment"),
        legend(ENDO_ORDER, ENDO_COLORS, "Endocrine"),
        legend(STAGE_ORDER, STAGE_COLORS, "Stage"),
    ])

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>MC cluster Fig 3e summary</title><style>{CSS}</style></head>"
        f"<body><h1>MC cluster catalog — Fig 3e-style family summary</h1>"
        f"<div class='meta'>{len(table)} family groups · "
        f"{int(table['n_variants'].sum())} cluster_final motifs · "
        f"{int(table['total_hits'].sum()):,} total hits across 22 cell types</div>"
        f"{legends_html}"
        f"<table><thead><tr>{header_html}</tr></thead><tbody>"
        + "\n".join(rows_html) + "</tbody></table></body></html>"
    )
    print(f"wrote {out_html}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog_dir", required=True)
    p.add_argument("--ct_metadata", default="/cellar/users/aklie/data/datasets/sc-islet-differentiation_10X-Multiome/config/cell_type_metadata.tsv")
    p.add_argument("--vierstra_meme", default="/cellar/users/aklie/data/ref/motifs/jvierstra/motif-clustering-v2.0beta/motifs.meme")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    cdir = Path(args.catalog_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cdir / "summary"
    out_html = out_dir / "fig3e_summary.html"

    classify = pd.read_csv(cdir / "qc" / "motif_classification.tsv", sep="\t")
    meta = pd.read_csv(cdir / "08_final_averages_metadata.tsv", sep="\t")
    per_motif = pd.read_csv(cdir / "annotation" / "per_motif_summary.tsv", sep="\t")
    per_motif_ct = pd.read_csv(cdir / "annotation" / "per_motif_per_ct.tsv", sep="\t")
    hit_count = pd.read_csv(cdir / "hits" / "hit_count_matrix.csv", index_col=0)
    ct_meta = pd.read_csv(args.ct_metadata, sep="\t").set_index("cell_type")

    # MEME PWMs (covers cluster_final + named display names)
    ppms = parse_meme_to_ppms((cdir / "combined_mapped.meme").read_text())
    # Vierstra PWMs (covers primary_subname targets)
    vierstra_text = Path(args.vierstra_meme).read_text() if Path(args.vierstra_meme).exists() else ""
    ppms.update(parse_meme_to_ppms(vierstra_text))

    # Load name_map so we can resolve display name -> h5 key for per-motif TomTom lookup
    name_map = pd.read_csv(cdir / "clustered_motifs_name_map_unique.tsv",
                           sep="\t", header=None, names=["h5_key", "display"])

    h5 = h5py.File(cdir / "clustered_motifs.modisco.h5", "r")
    try:
        table = build_family_table(classify, meta, per_motif, per_motif_ct, hit_count, ct_meta,
                                   catalog_dir=cdir, name_map=name_map)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.drop(columns=["region_props", "tss_values_per_ct", "lineage_props",
                            "endo_props", "stage_props"]).to_csv(
            out_dir / "family_rollup.tsv", sep="\t", index=False
        )
        print(f"wrote {out_dir/'family_rollup.tsv'} ({len(table)} families)")
        render_html(table, h5, ppms, ct_meta, out_html)
    finally:
        h5.close()


if __name__ == "__main__":
    main()
