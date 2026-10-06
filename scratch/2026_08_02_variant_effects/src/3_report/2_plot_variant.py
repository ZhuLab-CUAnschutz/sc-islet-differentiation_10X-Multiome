"""Stage 3.2: per-variant prediction and attribution plots.

Ports the 2025_11_03 `6_variant_test` notebook: for a variant in a cell type, load
the ChromBPNet model, predict reference and alternate accessibility, compute
DeepLiftShap attributions for both alleles, and render two PDFs: allele tracks
with attribution logos, and the delta (alt - ref) version.

Selects variants per trait from the config (named loci plus the top-N by minimum
p-value) and plots each in its strongest cell type. GPU stage; run on carter-gpu.

Usage:
    python src/3_report/2_plot_variant.py --config config/config.yaml \
        [--trait T2D] [--top-celltypes 1]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import seaborn as sns
import seqpro as sp
import torch
from bpnetlite.bpnet import BPNet, CountWrapper
from matplotlib import pyplot as plt
from tangermeme.annotate import annotate_seqlets
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.io import read_meme
from tangermeme.plot import plot_logo
from tangermeme.predict import predict
from tangermeme.seqlet import recursive_seqlets

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib, savefig

log = setup_logging(__file__)

SEQ_LEN = 2114
TARGET_LEN = 1000
MUT_WINDOW = 500
PLOT_WINDOW = 300
SHAP_DTYPE = torch.float64


def _windows():
    seq_center = SEQ_LEN // 2
    target_center = TARGET_LEN // 2
    return {
        "seq_start": seq_center - MUT_WINDOW // 2,
        "seq_end": seq_center + MUT_WINDOW // 2,
        "plot_seq_start": seq_center - PLOT_WINDOW // 2,
        "plot_seq_end": seq_center + PLOT_WINDOW // 2,
        "plot_target_start": target_center - PLOT_WINDOW // 2,
        "plot_target_end": target_center + PLOT_WINDOW // 2,
    }


def log_softmax_profile(y_profile: torch.Tensor) -> torch.Tensor:
    """Log-softmax a predicted profile over its length dimension."""
    z = y_profile.shape
    y = y_profile.reshape(y_profile.shape[0], -1)
    y = torch.nn.functional.log_softmax(y, dim=-1)
    return y.reshape(*z)


def allele_sequence(genome, chrom: str, pos: int, allele1: str, allele2: str):
    """Reference and alternate 2114-bp sequences centered on the variant.

    Force both alleles into the center (matching variant-scorer); the genome base
    can differ from allele1 (strand-reported variants), which would otherwise flip
    the predicted direction relative to the scored logfc.
    """
    half = SEQ_LEN // 2
    c0 = pos - 1  # variant_id position is the bed 1-based 'end'; pyfaidx is 0-based
    seq = genome[chrom][c0 - half:c0 + half].seq.upper()
    if seq[half] != allele1:
        log.warning("genome base %s != allele1 %s at %s:%d (forcing allele1)",
                    seq[half], allele1, chrom, pos)
    ref = seq[:half] + allele1 + seq[half + 1:]
    alt = seq[:half] + allele2 + seq[half + 1:]
    return ref, alt


def predict_counts_profile(model, seq: str):
    """Return (profile softmax-prob, exp(count)) for one sequence on GPU."""
    onehot = np.expand_dims(sp.ohe(seq, alphabet=sp.DNA), 0).transpose(0, 2, 1)
    profile, counts = predict(model, torch.tensor(onehot).float().cuda())
    counts = float(np.squeeze(np.exp(counts.cpu().detach().numpy())))
    profile = np.squeeze(torch.exp(log_softmax_profile(profile)).cpu().detach().numpy())
    return onehot, profile, counts


def attributions(count_wrapper, onehot):
    """DeepLiftShap attributions for the count head of one sequence."""
    return deep_lift_shap(
        count_wrapper.type(SHAP_DTYPE),
        torch.tensor(onehot).type(SHAP_DTYPE).cuda(),
        n_shuffles=20, batch_size=8, verbose=False, warning_threshold=1e-4,
    )


def annotate(onehot, attr, meme_path, win, motif_names, motif_tf):
    """Recursive seqlets on the mutation window, annotated with catalog motif names."""
    seqlets = recursive_seqlets(
        attr[:, :, win["seq_start"]:win["seq_end"]].sum(dim=1),
        threshold=0.05, additional_flanks=5)
    idxs = annotate_seqlets(
        torch.tensor(onehot)[:, :, win["seq_start"]:win["seq_end"]], seqlets, meme_path)[0][:, 0]
    seqlets["start"] += win["seq_start"]
    seqlets["end"] += win["seq_start"]
    names = list(np.atleast_1d(motif_names[np.atleast_1d(idxs)]))
    seqlets["example_idx"] = motif_tf.reindex(names).fillna("").to_numpy()
    return seqlets


def plot_pair(profiles, counts, attrs, seqlets, win, title, out_path):
    """Two-allele accessibility with reference and alternate attribution logos."""
    ref_prof, alt_prof = profiles
    ref_attr, alt_attr = attrs
    ref_seqlets, alt_seqlets = seqlets
    a, b = win["plot_target_start"], win["plot_target_end"]
    with __import__("seaborn").plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1, 1]})
        axes[0].plot(range(PLOT_WINDOW), ref_prof[a:b], color="r", lw=1,
                     label=f"Reference ({np.log(counts[0]):.2f})")
        axes[0].plot(range(PLOT_WINDOW), alt_prof[a:b], color="b", lw=1,
                     label=f"Alternate ({np.log(counts[1]):.2f})")
        axes[0].set_ylabel("Predicted Accessibility")
        axes[0].legend(fontsize=12, loc="upper left")
        axes[0].set_title(title)
        for ax, attr, sq in [(axes[1], ref_attr, ref_seqlets), (axes[2], alt_attr, alt_seqlets)]:
            plot_logo(attr[0], ax=ax, start=win["plot_seq_start"], end=win["plot_seq_end"],
                      annotations=sq, score_key="attribution", n_tracks=3,
                      show_extra=False, show_score=False)
            ax.set_ylabel("Attributions")
        savefig(fig, out_path)


def plot_delta(profiles, counts, attrs, seqlets, win, title, out_path):
    """Delta accessibility (alt - ref) with both attribution logos."""
    ref_prof, alt_prof = profiles
    ref_attr, alt_attr = attrs
    ref_seqlets, alt_seqlets = seqlets
    a, b = win["plot_target_start"], win["plot_target_end"]
    diff = np.asarray(alt_prof[a:b]) - np.asarray(ref_prof[a:b])
    with __import__("seaborn").plotting_context("paper", font_scale=1.5):
        fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1, 1]})
        axes[0].plot(range(PLOT_WINDOW), diff, color="purple", lw=1.5,
                     label=f"delta Accessibility ({np.log(counts[1] / counts[0]):.2f})")
        axes[0].fill_between(range(PLOT_WINDOW), diff, 0, where=(diff >= 0), color="purple", alpha=0.3)
        axes[0].fill_between(range(PLOT_WINDOW), diff, 0, where=(diff < 0), color="purple", alpha=0.15)
        axes[0].axhline(0, color="gray", ls="--", lw=1)
        axes[0].set_ylabel("delta Predicted Accessibility")
        axes[0].legend(fontsize=12, loc="upper left")
        axes[0].set_title(title)
        for ax, attr, sq in [(axes[1], ref_attr, ref_seqlets), (axes[2], alt_attr, alt_seqlets)]:
            plot_logo(attr[0], ax=ax, start=win["plot_seq_start"], end=win["plot_seq_end"],
                      annotations=sq, score_key="attribution", n_tracks=3,
                      show_extra=False, show_score=False)
            ax.set_ylabel("Attributions")
        savefig(fig, out_path)


def load_motif_tf(cfg) -> tuple[np.ndarray, pd.Series]:
    """Catalog motif names and a name -> ``ST##:TF`` label map for seqlet annotation.

    Labels are rooted in the v1.1 catalog: curator_tf (best_match_tf fallback),
    prefixed with the catalog short id, matching the hit tables and contrast figures.
    """
    meme = str(io.repo_path(cfg, cfg["catalog_meme"]))
    names = np.array(list(read_meme(meme).keys()))
    tf_map = io.load_catalog_tf(cfg)  # short_id -> curator_tf
    label = pd.Series({sid: io.catalog_label(sid, tf_map) for sid in names})  # ST##:TF
    return names, label


def select_variants(cfg, credible, pval, trait_key, top_celltypes):
    """Return list of (variant_id, cell_type) to plot for a trait."""
    named = set(cfg["report"]["named_loci"])
    top_n = cfg["report"]["top_n_variants"]
    present = [v for v in credible[credible["trait"] == trait_key]["id"].unique() if v in pval.index]
    top_ids = pval.loc[present].min(axis=1).sort_values().head(top_n).index
    chosen = list(named.intersection(present)) + [v for v in top_ids if v not in named]
    pairs = []
    for vid in chosen:
        for ct in pval.loc[vid].sort_values().head(top_celltypes).index:
            pairs.append((vid, ct))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--trait", help="Only this trait key; default all")
    ap.add_argument("--top-celltypes", type=int, default=1)
    args = ap.parse_args()

    configure_matplotlib()
    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    credible = io.load_credible_sets(cfg).set_index("id", drop=False)
    genome = pyfaidx.Fasta(cfg["genome"])
    meme = str(io.repo_path(cfg, cfg["catalog_meme"]))
    motif_names, motif_tf = load_motif_tf(cfg)
    win = _windows()
    out_dir = io.sandbox_path(cfg, "results/variants")
    out_dir.mkdir(parents=True, exist_ok=True)

    traits = [t for t in cfg["traits"] if not args.trait or t["key"] == args.trait]
    model_cache: dict[str, tuple] = {}
    for trait in traits:
        key = trait["key"]
        pval = pd.read_csv(io.sandbox_path(cfg, f"results/{key}/celltype_vep_pval.csv"),
                           index_col="variant_id")
        for vid, ct in select_variants(cfg, credible, pval, key, args.top_celltypes):
            chrom, pos, a1, a2 = vid.split(":")[0], int(vid.split(":")[1]), vid.split(":")[2], vid.split(":")[3]
            stem = f"{ct}.variant_{vid.replace(':', '_')}_predictions_attributions"
            out_main, out_delta = out_dir / f"{stem}.pdf", out_dir / f"{stem}_delta.pdf"
            if out_main.exists() and out_delta.exists():
                continue  # idempotent: skip already-rendered
            if ct not in model_cache:
                # Keep only the current cell type's model on the GPU: caching all 22
                # single-task models at once overruns the a30 during DeepLiftShap.
                for k in list(model_cache):
                    m, w = model_cache.pop(k); del m, w
                torch.cuda.empty_cache()
                mp = io.model_path(cfg, ct)
                model = BPNet.from_chrombpnet(str(mp)).cuda().eval()
                model_cache[ct] = (model, CountWrapper(model).cuda().eval())
            model, count_wrapper = model_cache[ct]

            ref, alt = allele_sequence(genome, chrom, pos, a1, a2)
            ref_oh, ref_prof, ref_cnt = predict_counts_profile(model, ref)
            alt_oh, alt_prof, alt_cnt = predict_counts_profile(model, alt)
            ref_attr = attributions(count_wrapper, ref_oh)
            alt_attr = attributions(count_wrapper, alt_oh)
            ref_sq = annotate(ref_oh, ref_attr, meme, win, motif_names, motif_tf)
            alt_sq = annotate(alt_oh, alt_attr, meme, win, motif_names, motif_tf)

            title = f"{chrom}:{pos} ({a1} -> {a2})  [{ct}]"
            plot_pair((ref_prof, alt_prof), (ref_cnt, alt_cnt), (ref_attr, alt_attr),
                      (ref_sq, alt_sq), win, title, out_main)
            plot_delta((ref_prof, alt_prof), (ref_cnt, alt_cnt), (ref_attr, alt_attr),
                       (ref_sq, alt_sq), win, title, out_delta)
            log.info("plotted %s in %s", vid, ct)

    io.write_provenance(cfg, "3_report_2_plot_variant", {
        "traits": [t["key"] for t in traits], "top_celltypes": args.top_celltypes})
    log.info("Done.")


if __name__ == "__main__":
    main()
