"""Stage 3.5: cell-type contrast panels for group/cell-type-specific variants.

For a specific variant, show predicted accessibility (reference vs alternate) and
DeepLiftShap attribution logos in the cell types where the effect is present
("on") next to cell types where it is absent ("off"), so the specificity is
visible. Rows are cell types (on then off); columns are accessibility, reference
attribution, alternate attribution.

Variant selection: named loci (config) plus, per trait, the top cell-type-specific
variants from the Stage 3.1 `celltype_specific_variants.tsv` (large, single-cell-type
effects). GPU stage; env eugene_tools.

Usage:
    python src/3_report/5_variant_celltype_contrast.py --config config/config.yaml \
        [--trait T2D] [--variants chr1:45848944:C:A,...] [--n-on 3 --n-off 3 --n-variants 3]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
from matplotlib import pyplot as plt
from tangermeme.plot import plot_logo

from common import io, metadata
from common import variant_tracks as vt
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib, savefig

log = setup_logging(__file__)


def select_on_off(logfc_row, pval_row, threshold, n_on, n_off):
    """Return (on_cts, off_cts): strongest significant vs weakest (flat) cell types."""
    order = logfc_row.abs().sort_values(ascending=False)
    sig = pval_row[pval_row <= threshold].index
    on = [c for c in order.index if c in set(sig)][:n_on]
    if not on:  # no significant cell type: fall back to the single strongest
        on = [order.index[0]]
    off = [c for c in order.index[::-1] if c not in on][:n_off]  # smallest |logfc|
    return on, off


def contrast_figure(cfg, genome, meme, motif_names, motif_tf, cache,
                    variant_id, trait, logfc_row, pval_row, on, off, out_path, annot=None,
                    ct_order=None, attr_hw=None):
    chrom, pos = variant_id.split(":")[0], int(variant_id.split(":")[1])
    a1, a2 = variant_id.split(":")[2], variant_id.split(":")[3]
    ref, alt = vt.allele_sequences(genome, chrom, pos, a1, a2)
    win = vt.windows()
    a, b = win["plot_target_start"], win["plot_target_end"]
    cts = ct_order if ct_order is not None else (on + off)
    on_set = set(on)
    fig, axes = plt.subplots(len(cts), 3, figsize=(16, 2.1 * len(cts)),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})
    axes = np.atleast_2d(axes)
    import torch
    for i, ct in enumerate(cts):
        model, cw = vt.load_model(cfg, ct)  # load per cell type, freed below (bounds GPU memory)
        ref_oh, ref_prof, ref_cnt = vt.predict_profile_counts(model, ref)
        alt_oh, alt_prof, alt_cnt = vt.predict_profile_counts(model, alt)
        ref_attr = vt.attributions(cw, ref_oh)
        alt_attr = vt.attributions(cw, alt_oh)
        ref_sq = vt.annotate(ref_oh, ref_attr, meme, win, motif_names, motif_tf)
        alt_sq = vt.annotate(alt_oh, alt_attr, meme, win, motif_names, motif_tf)

        tag = "ON" if ct in on_set else "off"
        lf, pv = logfc_row[ct], pval_row[ct]
        center = vt.PLOT_WINDOW // 2
        ax0 = axes[i, 0]
        # count-scaled coverage (profile prob x predicted counts) so the effect shows
        ax0.plot(range(vt.PLOT_WINDOW), ref_prof[a:b] * ref_cnt, color="r", lw=0.9, label="ref")
        ax0.plot(range(vt.PLOT_WINDOW), alt_prof[a:b] * alt_cnt, color="b", lw=0.9, label="alt")
        ax0.axvline(center, color="0.4", ls=":", lw=0.8)
        color = "#c0392b" if ct in on_set else "0.35"
        ax0.set_ylabel(f"{ct}\n[{tag}]", fontsize=8, rotation=0, ha="right", va="center", color=color)
        ax0.set_title(f"logfc={lf:.2f}  p={pv:.3g}", fontsize=8, color=color)
        if i == 0:
            ax0.legend(fontsize=7, loc="upper left")
        # attribution window: zoomed to +/- attr_hw around the variant with only the
        # variant-overlapping seqlet(s) labeled, or the full plot window otherwise
        vc = vt.SEQ_LEN // 2  # variant position in sequence coords
        if attr_hw:
            a_start, a_end, a_center = vc - attr_hw, vc + attr_hw, attr_hw
        else:
            a_start, a_end, a_center = win["plot_seq_start"], win["plot_seq_end"], center
        for ax, attr, sq, lab in [(axes[i, 1], ref_attr, ref_sq, "ref attr"),
                                  (axes[i, 2], alt_attr, alt_sq, "alt attr")]:
            sq_use = sq[(sq["start"] <= vc) & (sq["end"] >= vc)] if attr_hw else sq
            plot_logo(attr[0], ax=ax, start=a_start, end=a_end,
                      annotations=sq_use, score_key="attribution", n_tracks=1,
                      show_extra=False, show_score=False)
            ax.axvline(a_center, color="0.4", ls=":", lw=0.8)
            if i == 0:
                ax.set_title(lab, fontsize=8)
        log.info("  %s %s (%s) logfc=%.2f", variant_id, ct, tag, lf)
        del model, cw, ref_attr, alt_attr, ref_oh, alt_oh
        torch.cuda.empty_cache()
    if annot is not None:
        motif = annot.get("primary_motif") or annot.get("hit_motifs", "")
        hit_txt = f"disrupts {motif}" if annot.get("disrupts_hit") else "no called hit"
        sup = (f"#{int(annot['rank'])}  {variant_id} ({a1}->{a2}) [{trait}]  |  "
               f"{annot.get('scope', '')}  |  top={annot['top_cell_type']} "
               f"logfc={annot['logfc']:.2f} peak={annot['peak_pred_counts']:.0f}  |  {hit_txt}")
    else:
        sup = f"{variant_id} ({a1}->{a2})  [{trait}]  on vs off cell types"
    fig.suptitle(sup, fontsize=10)
    savefig(fig, out_path)


def pick_variants(cfg, trait_key, pval, logfc, n_variants):
    """Named loci plus a VARIETY of specific variants: the strongest example per
    distinct top cell type (so the batch spans lineages, not one cell type)."""
    named = [v for v in cfg["report"]["named_loci"] if v in logfc.index]
    spec_path = io.sandbox_path(cfg, f"results/{trait_key}/vep_report_out/"
                                     "celltype_vep_logfc.celltype_specific_variants.tsv")
    picked = list(named)
    if spec_path.exists():
        spec = pd.read_csv(spec_path, sep="\t")
        spec = spec[spec["top_logfc"].abs() >= 0.5].sort_values("top_z", ascending=False)
        spec = spec.drop_duplicates("top_cell_type")  # one per cell type -> variety
        picked += [v for v in spec["variant_id"].head(n_variants) if v not in picked]
    return picked


def build_variety_pool(cfg, max_variants, per_celltype):
    """Pool cell-type-specific variants across all traits and spread across top cell
    types (round-robin by top_cell_type, strongest first) for a diverse batch.

    Returns a list of (trait_key, variant_id) of length <= max_variants.
    """
    frames = []
    for trait in cfg["traits"]:
        p = io.sandbox_path(cfg, f"results/{trait['key']}/vep_report_out/"
                                 "celltype_vep_logfc.celltype_specific_variants.tsv")
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            df["trait"] = trait["key"]
            frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["top_logfc"].abs() >= 0.5].sort_values("top_z", ascending=False)
    picked, per_ct, seen = [], {}, set()
    # round-robin: pass through strongest-first, capping picks per top cell type
    for cap in range(1, per_celltype + 1):
        for _, r in pool.iterrows():
            if len(picked) >= max_variants:
                break
            ct, vid = r["top_cell_type"], r["variant_id"]
            if vid in seen or per_ct.get(ct, 0) >= cap:
                continue
            picked.append((r["trait"], vid)); seen.add(vid); per_ct[ct] = per_ct.get(ct, 0) + 1
        if len(picked) >= max_variants:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trait")
    ap.add_argument("--variants", help="comma-separated variant ids (needs --trait)")
    ap.add_argument("--n-on", type=int, default=3)
    ap.add_argument("--n-off", type=int, default=3)
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--variety-batch", action="store_true",
                    help="select a diverse set across all traits and cell types")
    ap.add_argument("--max-variants", type=int, default=75)
    ap.add_argument("--per-celltype", type=int, default=5)
    ap.add_argument("--rank-file", help="ranking tsv: render top --max-examples in that order")
    ap.add_argument("--max-examples", type=int, default=80)
    ap.add_argument("--all-celltypes", action="store_true",
                    help="render every cell type (display order) per variant, for the variant browser")
    ap.add_argument("--attr-halfwidth", type=int, default=None,
                    help="zoom attribution to +/- this many bp around the variant and label only "
                         "the variant-overlapping seqlet (default: 60 in all-celltypes mode, else full window)")
    ap.add_argument("--chunk", help="i/N: process only chunk i of N (1-indexed) for array jobs")
    args = ap.parse_args()

    configure_matplotlib()
    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    genome = pyfaidx.Fasta(cfg["genome"])
    meme, motif_names, motif_tf = vt.load_motif_tf(cfg)
    threshold = cfg["significance"]["pval_threshold"]
    all_ct_order = metadata.cell_types(cfg) if args.all_celltypes else None
    attr_hw = args.attr_halfwidth if args.attr_halfwidth is not None else (60 if args.all_celltypes else None)
    out_dir = io.sandbox_path(cfg, "results/variants/allct" if args.all_celltypes
                              else "results/variants/contrast")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    study = {t["key"]: t["study"] for t in cfg["traits"]}
    annot_map: dict = {}
    # Build the (trait, variant) work list.
    if args.rank_file:
        rank_df = pd.read_csv(io.sandbox_path(cfg, args.rank_file), sep="\t").head(args.max_examples)
        jobs = list(zip(rank_df["trait"], rank_df["variant_id"]))
        annot_map = {r["variant_id"]: r for _, r in rank_df.iterrows()}
    elif args.variety_batch:
        jobs = build_variety_pool(cfg, args.max_variants, args.per_celltype)
    else:
        jobs = []
        for trait in [t for t in cfg["traits"] if not args.trait or t["key"] == args.trait]:
            td = io.sandbox_path(cfg, f"results/{trait['key']}")
            logfc = pd.read_csv(td / "celltype_vep_logfc.csv", index_col="variant_id")
            pval = pd.read_csv(td / "celltype_vep_pval.csv", index_col="variant_id")
            vids = (args.variants.split(",") if args.variants
                    else pick_variants(cfg, trait["key"], pval, logfc, args.n_variants))
            jobs += [(trait["key"], v) for v in vids]

    if jobs:  # guard against off-by-one on the FULL set (before chunking; a per-chunk
        # check on 1-2 variants would false-fail on the ~4% strand-flipped variants)
        frac = vt.check_coordinates(genome, [v for _, v in jobs])
        log.info("coordinate check: %.1f%% of variants have genome[end-1]==allele1", 100 * frac)
    if args.chunk:  # i/N for array parallelism
        i, n = (int(x) for x in args.chunk.split("/"))
        jobs = jobs[i - 1::n]
    log.info("contrast jobs to render: %d", len(jobs))

    # Process grouped by trait so each per-trait matrix loads once.
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
            annot = annot_map.get(vid)
            if args.all_celltypes:  # variant browser: every cell type, keyed by variant
                out = out_dir / f"{vid.replace(':', '_')}.allct.pdf"
                on = [c for c in all_ct_order if pval.loc[vid, c] <= threshold]
                off, ct_order = [], all_ct_order
            else:
                prefix = f"rank{int(annot['rank']):02d}_" if annot is not None else ""
                out = out_dir / f"{prefix}{vid.replace(':', '_')}.{study[key]}.celltype_contrast.pdf"
                on, off = select_on_off(logfc.loc[vid], pval.loc[vid], threshold, args.n_on, args.n_off)
                ct_order = None
            if out.exists():
                continue  # idempotent: skip already-rendered
            contrast_figure(cfg, genome, meme, motif_names, motif_tf, cache,
                            vid, key, logfc.loc[vid], pval.loc[vid], on, off, out,
                            annot=annot, ct_order=ct_order, attr_hw=attr_hw)
            log.info("%s [%s] -> %s", vid, key, out.name)

    io.write_provenance(cfg, "3_report_5_variant_celltype_contrast", {
        "n_jobs": len(jobs), "variety_batch": args.variety_batch,
        "max_variants": args.max_variants, "per_celltype": args.per_celltype,
        "n_on": args.n_on, "n_off": args.n_off})
    log.info("Done.")


if __name__ == "__main__":
    main()
