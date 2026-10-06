"""Stage 3.9: per-variant page figures for the variant browser.

For each selected variant, render one clean panel per cell type plus a cell-type
-colored effect barplot, written to ``results/variants/pages/{vid}/``. The page
builder (Stage 5c) lays these out as a dedicated HTML page per variant.

Each cell-type panel has, on a shared scale across all cell types:
  - predicted accessibility (count-scaled), reference dashed + lighter and alternate
    solid + darker, both in that cell type's catalog color;
  - reference and alternate DeepLiftShap attribution logos, annotated with the finemo
    **called hits** for that cell type and allele (from ``hits_disruption.tsv``), so the
    boxes match the hit table exactly (not an independent seqlet matcher).

Predictions share one y-max (the largest across cell types) and attributions share
another, so panels are directly comparable. Model is loaded and freed per cell type
to bound a30 memory.

The GPU intermediates (coverage, onehots, attributions) are cached per variant to
``intermediates.npz``. Only the first run needs a GPU; style/layout changes re-render
from the cache on CPU alone (``--force`` to overwrite PNGs, without ``--recompute``),
so this can run in the CPU env with no model load.

Usage:
    # first render (GPU): compute + cache + draw
    python src/3_report/9_variant_pages.py --config config/config.yaml [--variant chr:pos:a1:a2]
    # re-render after a style change (CPU only, uses the cache)
    python src/3_report/9_variant_pages.py --config config/config.yaml --force
    # options: [--rank-file ...] [--max-examples 50] [--chunk i/N] [--halfwidth 50]
    #          [--dpi 120] [--recompute]
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyfaidx
import torch
from matplotlib import pyplot as plt
from tangermeme.plot import plot_logo

from common import io, metadata
from common import variant_tracks as vt
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib, savefig

log = setup_logging(__file__)
SEQ_CENTER = vt.SEQ_LEN // 2       # 1057; variant base sits here in the forced sequence
TARGET_CENTER = vt.TARGET_LEN // 2  # 500; variant base in the predicted profile


def lighten(color, amount: float):
    """Blend toward white (amount 0..1)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def darken(color, amount: float):
    """Blend toward black (amount 0..1)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def called_hit_boxes(hits, vid, ct, allele_key, c0, win) -> pd.DataFrame:
    """finemo called hits for one variant/cell type/allele as plot_logo annotations.

    Converts each hit's genomic span to full-sequence coordinates (variant base at
    SEQ_CENTER) and keeps those overlapping the plotted window. Columns match what
    plot_logo reads: example_idx (the ST##:TF label), start, end, attribution (score).
    """
    g = hits[(hits["variant_id"] == vid) & (hits["cell_type"] == ct)
             & (hits["allele"] == allele_key)]
    rows = []
    for _, h in g.iterrows():
        s = SEQ_CENTER + (int(h["start"]) - c0)
        e = SEQ_CENTER + (int(h["end"]) - c0)
        if e < win["plot_seq_start"] or s > win["plot_seq_end"]:
            continue
        rows.append({"example_idx": h["motif"], "start": s, "end": e,
                     "attribution": float(h.get("hit_importance", 1.0) or 1.0)})
    return pd.DataFrame(rows, columns=["example_idx", "start", "end", "attribution"])


def standing_boxes(standing, vid, ct, c0) -> pd.DataFrame:
    """Standing v1.1 genome-wide catalog hits overlapping this variant/cell type,
    converted to full-sequence coordinates (Stage 4.4 already filtered to overlaps)."""
    g = standing[(standing["variant_id"] == vid) & (standing["cell_type"] == ct)]
    rows = [{"example_idx": r["motif"], "start": SEQ_CENTER + (int(r["start"]) - c0),
             "end": SEQ_CENTER + (int(r["end"]) - c0)} for _, r in g.iterrows()]
    return pd.DataFrame(rows, columns=["example_idx", "start", "end"])


_ARRAYS = ("cov_r", "cov_a", "oh_r", "oh_a", "attr_r", "attr_a")


def compute_intermediates(cfg, order, ref, alt) -> dict:
    """GPU pass: per-cell-type count-scaled coverage, onehots, and attributions.

    This is the only expensive (GPU) step; its output is cached so later style/layout
    changes re-render on CPU alone. Model is loaded and freed per cell type.
    """
    per_ct: dict = {}
    for ct in order:
        try:
            model, cw = vt.load_model(cfg, ct)
        except Exception as e:  # noqa: BLE001 - a missing model shouldn't kill the page
            log.warning("no model for %s (%s); skipping panel", ct, e)
            continue
        oh_r, prof_r, cnt_r = vt.predict_profile_counts(model, ref)
        oh_a, prof_a, cnt_a = vt.predict_profile_counts(model, alt)
        per_ct[ct] = {
            "cov_r": (prof_r * cnt_r).astype("float32"),
            "cov_a": (prof_a * cnt_a).astype("float32"),
            "oh_r": oh_r.astype("float32"), "oh_a": oh_a.astype("float32"),
            "attr_r": vt.attributions(cw, oh_r).cpu().numpy().astype("float32"),
            "attr_a": vt.attributions(cw, oh_a).cpu().numpy().astype("float32"),
        }
        del model, cw
        torch.cuda.empty_cache()
    return per_ct


def save_cache(path, per_ct: dict) -> None:
    """Persist per-cell-type intermediates to one compressed .npz."""
    flat = {f"{ct}||{k}": v for ct, d in per_ct.items() for k, v in d.items()}
    flat["__cts__"] = np.array(list(per_ct.keys()))
    np.savez_compressed(path, **flat)


def load_cache(path) -> dict:
    """Load cached intermediates (CPU only; no model/GPU needed)."""
    z = np.load(path, allow_pickle=True)
    return {str(ct): {k: z[f"{ct}||{k}"] for k in _ARRAYS} for ct in z["__cts__"]}


def shared_scales(per_ct: dict, hw: int):
    """Global prediction and attribution y-maxima so panels share a scale.

    Predictions span the full 1 kb output window, so their y-max is over the whole
    profile; attributions are shown zoomed to +/-hw, so their y-max uses that window.
    """
    pred_ymax = attr_ymax = 0.0
    sw = slice(SEQ_CENTER - hw, SEQ_CENTER + hw)
    for d in per_ct.values():
        pred_ymax = max(pred_ymax, float(d["cov_r"].max()), float(d["cov_a"].max()))
        attr_ymax = max(attr_ymax, float(np.abs(d["attr_r"][0][:, sw]).max()),
                        float(np.abs(d["attr_a"][0][:, sw]).max()))
    return pred_ymax, attr_ymax


# One lane (row) per hit source, so nothing overlaps and the source is unambiguous.
HIT_COLOR = "#111111"      # finemo called hit (this variant's SHAP, per allele)
SEQLET_COLOR = "#8e44ad"   # seqlet-matcher match (this variant's attributions, per allele)
V11_COLOR = "#00897b"      # standing v1.1 genome-wide catalog hit (reference peaks)
LANE_NAMES = ["v1.1 catalog", "finemo", "seqlet"]
LANE_COLORS = [V11_COLOR, HIT_COLOR, SEQLET_COLOR]


def draw_lanes(ax, lane_dfs, sa, hw):
    """Draw one row per hit source under a logo: a colored bar at each hit's span plus a
    single combined ST##:TF label per lane (all shown hits overlap the variant, so one
    label per lane keeps rows clean and non-overlapping). Empty lanes stay blank."""
    n = len(lane_dfs)
    ax.set_xlim(-0.5, 2 * hw - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # lane 0 (v1.1) at top
    ax.set_yticks(range(n))
    ax.set_yticklabels(LANE_NAMES, fontsize=7)
    ax.set_xticks([0, hw, 2 * hw])
    ax.set_xticklabels([-hw, 0, hw])
    ax.tick_params(axis="x", labelsize=7)
    ax.axvline(hw, color="0.6", lw=0.8, ls=":")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for i, df in enumerate(lane_dfs):
        color = LANE_COLORS[i]
        ax.axhline(i, color="0.9", lw=0.5, zorder=0)
        if df is None or df.empty:
            continue
        motifs = []
        for _, a in df.iterrows():
            ax.plot([a["start"] - sa, a["end"] - sa], [i, i], color=color, lw=5,
                    solid_capstyle="butt")
            motifs.append(str(a["example_idx"]))
        label = ", ".join(dict.fromkeys(motifs))  # dedupe, keep order
        ax.text(hw, i - 0.34, label, ha="center", va="bottom", fontsize=6.5, color=color)


def render_panel(ct, color, cov, attrs, ref_lanes, alt_lanes, pred_ymax, attr_ymax, hw,
                 alleles, logfc, pval, out_png, dpi):
    """One cell-type panel, three columns: predictions | ref attribution | alt attribution.

    Predictions (full 1 kb) are count-scaled, reference dashed/lighter and alternate
    solid/darker in the cell type's color. Each attribution logo (zoomed to +/-hw) has a
    track below it with one row per hit source: the standing v1.1 catalog hit, this
    variant's finemo call, and the seqlet-matcher match, each labeled ST##:TF. All shown
    hits overlap the variant. Predictions and attributions each share a y-scale across
    cell types.
    """
    cov_ref, cov_alt = cov
    ref_attr, alt_attr = attrs
    a1, a2 = alleles
    light, dark = lighten(color, 0.5), darken(color, 0.3)
    sa, sb = SEQ_CENTER - hw, SEQ_CENTER + hw

    fig, axd = plt.subplot_mosaic(
        [["pred", "reflogo", "altlogo"], ["pred", "reftrk", "alttrk"]],
        figsize=(17, 4.4),
        gridspec_kw={"height_ratios": [3, 1.5], "width_ratios": [1.25, 1, 1]})

    # Predictions over the FULL 1 kb output window (variant at the center).
    xp = np.arange(vt.TARGET_LEN) - TARGET_CENTER
    axd["pred"].plot(xp, cov_ref, color=light, lw=1.4, ls="--", label=f"ref ({a1})")
    axd["pred"].plot(xp, cov_alt, color=dark, lw=1.4, ls="-", label=f"alt ({a2})")
    axd["pred"].set_xlim(xp[0], xp[-1])
    axd["pred"].margins(x=0)
    axd["pred"].set_ylim(0, pred_ymax * 1.05)
    axd["pred"].axvline(0, color="0.55", lw=0.9, ls=":")
    axd["pred"].set_ylabel("pred. accessibility")
    axd["pred"].set_xlabel("position relative to variant (bp, full 1 kb window)")
    axd["pred"].legend(loc="upper left", fontsize=8, frameon=False)
    axd["pred"].set_title("predictions", fontsize=10)

    for lk, tk, attr, lanes, tag in [("reflogo", "reftrk", ref_attr, ref_lanes, f"ref ({a1}) attribution"),
                                     ("altlogo", "alttrk", alt_attr, alt_lanes, f"alt ({a2}) attribution")]:
        plot_logo(attr, ax=axd[lk], start=sa, end=sb, annotations=None,
                  n_tracks=3, show_extra=False, show_score=False)
        axd[lk].set_xlim(-0.5, 2 * hw - 0.5)
        axd[lk].margins(x=0)
        axd[lk].set_ylim(-attr_ymax * 1.05, attr_ymax * 1.05)
        axd[lk].axvline(hw, color="0.55", lw=0.9, ls=":")
        axd[lk].set_xticks([])
        axd[lk].set_title(tag, fontsize=10)
        draw_lanes(axd[tk], lanes, sa, hw)
        axd[tk].set_xlabel(f"position relative to variant (bp, +/-{hw})", fontsize=8)

    fig.suptitle(f"{ct}    logfc = {logfc:+.2f}    p = {pval:.2g}",
                 color=dark, fontsize=13, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.18, hspace=0.35, top=0.9)
    savefig(fig, out_png, dpi=dpi)


def render_bar(logfc_row, order, cmap, out_png, dpi, vid, trait):
    """Effect (logfc) across all cell types, colored by cell type, stage-ordered."""
    vals = [float(logfc_row[c]) for c in order]
    colors = [cmap.get(c, "0.6") for c in order]
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    ax.barh(range(len(order)), vals, color=colors, edgecolor="0.25", lw=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.9)
    ax.set_xlabel("logfc (alt vs ref)")
    ax.set_title(f"{vid}  [{trait}]", fontsize=10)
    savefig(fig, out_png, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rank-file", default="results/variants/browser_variants.tsv")
    ap.add_argument("--variant", help="Only this variant_id (prototype one page)")
    ap.add_argument("--max-examples", type=int, default=50)
    ap.add_argument("--chunk", help="i/N for array parallelism")
    ap.add_argument("--halfwidth", type=int, default=50, help="bp each side of the variant")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--force", action="store_true", help="re-render even if the page PNGs exist")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute GPU intermediates instead of loading the cached .npz")
    args = ap.parse_args()

    configure_matplotlib()
    cfg = io.load_config(args.config)
    order = metadata.cell_types(cfg)
    cmap = metadata.color_map(cfg)
    genome = pyfaidx.Fasta(cfg["genome"])
    win = vt.windows()
    hw = args.halfwidth
    meme, motif_names, tf_map = vt.load_motif_tf(cfg)  # for the seqlet-matcher annotations
    hits = pd.read_csv(io.sandbox_path(cfg, "results/hits/hits_disruption.tsv"), sep="\t")
    sp = io.sandbox_path(cfg, "results/hits/standing_hit_overlaps.tsv")  # standing v1.1 catalog hits
    standing = pd.read_csv(sp, sep="\t") if sp.exists() else \
        pd.DataFrame(columns=["variant_id", "cell_type", "start", "end", "motif"])
    out_root = io.sandbox_path(cfg, "results/variants/pages")
    out_root.mkdir(parents=True, exist_ok=True)

    rank_df = pd.read_csv(io.sandbox_path(cfg, args.rank_file), sep="\t").head(args.max_examples)
    if args.variant:
        rank_df = rank_df[rank_df["variant_id"] == args.variant]
    jobs = list(zip(rank_df["trait"], rank_df["variant_id"]))
    if jobs:
        frac = vt.check_coordinates(genome, [v for _, v in jobs])
        log.info("coordinate check: %.1f%% of variants have genome[end-1]==allele1", 100 * frac)
    if args.chunk:
        i, n = (int(x) for x in args.chunk.split("/"))
        jobs = jobs[i - 1::n]
    log.info("variant pages to render: %d", len(jobs))

    by_trait: dict = {}
    for key, vid in jobs:
        by_trait.setdefault(key, []).append(vid)

    for key, vids in by_trait.items():
        td = io.sandbox_path(cfg, f"results/{key}")
        logfc = pd.read_csv(td / "celltype_vep_logfc.csv", index_col="variant_id")
        pval = pd.read_csv(td / "celltype_vep_pval.csv", index_col="variant_id")
        for vid in vids:
            if vid not in logfc.index:
                log.warning("%s not scored for %s", vid, key)
                continue
            vdir = out_root / vid.replace(":", "_")
            done = vdir / "bar.png"
            if done.exists() and not args.force:
                log.info("skip (exists) %s", vid)
                continue
            vdir.mkdir(parents=True, exist_ok=True)
            chrom, pos, a1, a2 = vid.split(":")
            pos = int(pos); c0 = pos - 1

            # GPU intermediates (coverage + attributions) are cached to .npz; a style or
            # layout change then re-renders from the cache with no model load or GPU.
            cache = vdir / "intermediates.npz"
            if cache.exists() and not args.recompute:
                per_ct = load_cache(cache)
                log.info("cache hit %s (render-only, no GPU)", vid)
            else:
                ref, alt = vt.allele_sequences(genome, chrom, pos, a1, a2)
                per_ct = compute_intermediates(cfg, order, ref, alt)
                save_cache(cache, per_ct)
                log.info("computed + cached %s (%d cell types)", vid, len(per_ct))
            pred_ymax, attr_ymax = shared_scales(per_ct, hw)

            # Render (CPU): panels on shared scales, overlaying finemo called hits and
            # seqlet-matcher matches. Both are filtered to annotations that OVERLAP the
            # variant base (the motif the variant actually sits in), not every seqlet in
            # the window; _draw_brackets then staggers labels so they never collide.
            def overlaps_variant(df):
                if df is None or len(df) == 0 or "start" not in df.columns:
                    return pd.DataFrame(columns=["example_idx", "start", "end"])
                return df[(df["start"] <= SEQ_CENTER) & (df["end"] >= SEQ_CENTER)]

            for ct, d in per_ct.items():
                rc = overlaps_variant(called_hit_boxes(hits, vid, ct, "allele1", c0, win))
                ac = overlaps_variant(called_hit_boxes(hits, vid, ct, "allele2", c0, win))
                rm = overlaps_variant(vt.annotate(d["oh_r"], torch.tensor(d["attr_r"]),
                                                  meme, win, motif_names, tf_map))
                am = overlaps_variant(vt.annotate(d["oh_a"], torch.tensor(d["attr_a"]),
                                                  meme, win, motif_names, tf_map))
                st = standing_boxes(standing, vid, ct, c0)  # genome-wide catalog hit (both alleles)
                render_panel(ct, cmap.get(ct, "#666666"), (d["cov_r"], d["cov_a"]),
                             (d["attr_r"][0], d["attr_a"][0]), [st, rc, rm], [st, ac, am],
                             pred_ymax, attr_ymax, hw, (a1, a2),
                             float(logfc.loc[vid, ct]), float(pval.loc[vid, ct]),
                             vdir / f"{ct}.png", args.dpi)
            render_bar(logfc.loc[vid], order, cmap, vdir / "bar.png", args.dpi, vid, key)
            (vdir / "meta.json").write_text(json.dumps({
                "variant_id": vid, "trait": key, "n_panels": len(per_ct),
                "pred_ymax": pred_ymax, "attr_ymax": attr_ymax, "halfwidth": hw}))
            log.info("rendered %s: %d panels", vid, len(per_ct))

    io.write_provenance(cfg, "3_report_9_variant_pages",
                        {"n_variants": len(jobs), "halfwidth": hw, "dpi": args.dpi})
    log.info("Done.")


if __name__ == "__main__":
    main()
