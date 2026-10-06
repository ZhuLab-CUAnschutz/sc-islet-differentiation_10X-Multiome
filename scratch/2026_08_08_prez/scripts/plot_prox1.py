#!/usr/bin/env python
"""Presentation re-render of the PROX1 variant effect on chosen cell types.

Drives the Stage 3.2 machinery of the 2026_08_02 variant-effect pipeline (imported, not
copied, so the models / coordinates / seqlet annotation are identical) with three changes
needed for a slide:

  * explicit --variants and --cell-types instead of "top N by p-value"
  * shared y-limits across the reference and alternate attribution panels. Autoscaling them
    independently hides the effect: in the Aug-05 render the ref panel scales to 0.1 and the
    alt to 0.01, so a genuine ~10x collapse of attribution at the HNF1 site reads as merely
    "different letters".
  * --force, since the upstream script skips anything already rendered

The 2026_08_02 sandbox is not modified.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyfaidx
import seaborn as sns
import torch
from bpnetlite.bpnet import BPNet, CountWrapper
from matplotlib import pyplot as plt
from tangermeme.plot import plot_logo

VEP_SRC = Path("/carter/users/aklie/projects/islet_organoid_differentiation/"
               "sc-islet-differentiation_10X-Multiome/scratch/2026_08_02_variant_effects/src")


def _import_stage(vep_src: Path):
    sys.path.insert(0, str(vep_src))
    sys.path.insert(0, str(vep_src / "3_report"))
    import importlib
    mod = importlib.import_module("2_plot_variant")
    from common import io
    from common.plotting import configure_matplotlib, savefig
    return mod, io, configure_matplotlib, savefig


def _share_attr_ylim(axes):
    """One y-scale across both attribution panels, so ref and alt are directly comparable."""
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def _mark_variant(axes, M):
    """Dashed rule at the variant, which sits at the center of the plotted window."""
    for ax in axes:
        ax.axvline(M.PLOT_WINDOW // 2, color="k", ls="--", lw=1, zorder=0, alpha=0.7)


def plot_pair(M, profiles, counts, attrs, seqlets, win, title, out_path, savefig):
    ref_prof, alt_prof = profiles
    a, b = win["plot_target_start"], win["plot_target_end"]
    with sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        axes[0].plot(range(M.PLOT_WINDOW), ref_prof[a:b], color="r", lw=1,
                     label=f"Reference (log counts {np.log(counts[0]):.2f})")
        axes[0].plot(range(M.PLOT_WINDOW), alt_prof[a:b], color="b", lw=1,
                     label=f"Alternate (log counts {np.log(counts[1]):.2f})")
        axes[0].set_ylabel("Predicted Accessibility")
        axes[0].legend(fontsize=12, loc="upper left")
        axes[0].set_title(title)
        for ax, attr, sq, lab in [(axes[1], attrs[0], seqlets[0], "Reference"),
                                  (axes[2], attrs[1], seqlets[1], "Alternate")]:
            plot_logo(attr[0], ax=ax, start=win["plot_seq_start"], end=win["plot_seq_end"],
                      annotations=sq, score_key="attribution", n_tracks=5,
                      show_extra=False, show_score=False)
            ax.set_ylabel(f"{lab}\nattributions")
        _share_attr_ylim(axes[1:])
        _mark_variant(axes, M)
        savefig(fig, out_path)


def plot_delta(M, profiles, counts, attrs, seqlets, win, title, out_path, savefig):
    ref_prof, alt_prof = profiles
    a, b = win["plot_target_start"], win["plot_target_end"]
    diff = np.asarray(alt_prof[a:b]) - np.asarray(ref_prof[a:b])
    with sns.plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1, 1]})
        axes[0].plot(range(M.PLOT_WINDOW), diff, color="purple", lw=1.5,
                     label=f"delta Accessibility (log fold change {np.log(counts[1] / counts[0]):.2f})")
        axes[0].fill_between(range(M.PLOT_WINDOW), diff, 0, where=(diff >= 0), color="purple", alpha=0.3)
        axes[0].fill_between(range(M.PLOT_WINDOW), diff, 0, where=(diff < 0), color="purple", alpha=0.15)
        axes[0].axhline(0, color="gray", ls="--", lw=1)
        axes[0].set_ylabel("delta Predicted\nAccessibility")
        axes[0].legend(fontsize=12, loc="upper left")
        axes[0].set_title(title)
        for ax, attr, sq, lab in [(axes[1], attrs[0], seqlets[0], "Reference"),
                                  (axes[2], attrs[1], seqlets[1], "Alternate")]:
            plot_logo(attr[0], ax=ax, start=win["plot_seq_start"], end=win["plot_seq_end"],
                      annotations=sq, score_key="attribution", n_tracks=5,
                      show_extra=False, show_score=False)
            ax.set_ylabel(f"{lab}\nattributions")
        _share_attr_ylim(axes[1:])
        _mark_variant(axes, M)
        savefig(fig, out_path)


def main(a):
    M, io, configure_matplotlib, savefig = _import_stage(Path(a.vep_src))
    configure_matplotlib()
    cfg = io.load_config(a.config)
    genome = pyfaidx.Fasta(cfg["genome"])
    meme = str(io.repo_path(cfg, cfg["catalog_meme"]))
    motif_names, motif_tf = M.load_motif_tf(cfg)
    win = M._windows()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vid in a.variants.split(","):
        chrom, pos, a1, a2 = vid.split(":")[0], int(vid.split(":")[1]), vid.split(":")[2], vid.split(":")[3]
        for ct in a.cell_types.split(","):
            stem = f"{ct}.variant_{vid.replace(':', '_')}_predictions_attributions"
            out_main, out_delta = out_dir / f"{stem}.pdf", out_dir / f"{stem}_delta.pdf"
            if out_main.exists() and out_delta.exists() and not a.force:
                print(f"[skip] {vid} {ct}", flush=True)
                continue

            model = BPNet.from_chrombpnet(str(io.model_path(cfg, ct))).cuda().eval()
            wrapper = CountWrapper(model).cuda().eval()

            ref, alt = M.allele_sequence(genome, chrom, pos, a1, a2)
            ref_oh, ref_prof, ref_cnt = M.predict_counts_profile(model, ref)
            alt_oh, alt_prof, alt_cnt = M.predict_counts_profile(model, alt)
            ref_attr = M.attributions(wrapper, ref_oh)
            alt_attr = M.attributions(wrapper, alt_oh)
            ref_sq = M.annotate(ref_oh, ref_attr, meme, win, motif_names, motif_tf)
            alt_sq = M.annotate(alt_oh, alt_attr, meme, win, motif_names, motif_tf)

            title = f"{chrom}:{pos} ({a1} -> {a2})  [{ct}]"
            plot_pair(M, (ref_prof, alt_prof), (ref_cnt, alt_cnt), (ref_attr, alt_attr),
                      (ref_sq, alt_sq), win, title, out_main, savefig)
            plot_delta(M, (ref_prof, alt_prof), (ref_cnt, alt_cnt), (ref_attr, alt_attr),
                       (ref_sq, alt_sq), win, title, out_delta, savefig)
            rows.append(f"{vid}\t{ct}\t{np.log(ref_cnt):.4f}\t{np.log(alt_cnt):.4f}\t"
                        f"{np.log(alt_cnt / ref_cnt):.4f}")
            print(f"[plot] {vid} {ct}  log counts ref={np.log(ref_cnt):.3f} alt={np.log(alt_cnt):.3f} "
                  f"logfc={np.log(alt_cnt / ref_cnt):+.3f}", flush=True)

            del model, wrapper
            torch.cuda.empty_cache()

    if rows:
        with open(out_dir / "prox1_logcounts.tsv", "w") as fh:
            fh.write("variant_id\tcell_type\tlog_counts_ref\tlog_counts_alt\tlogfc\n")
            fh.write("\n".join(rows) + "\n")
    print("[done]", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="2026_08_02_variant_effects config/config.yaml")
    p.add_argument("--vep_src", default=str(VEP_SRC))
    p.add_argument("--variants", default="chr1:213977102:T:A")
    p.add_argument("--cell-types", dest="cell_types", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
