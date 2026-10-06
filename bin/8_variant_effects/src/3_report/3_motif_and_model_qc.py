"""Stage 3.3: motif logo wall and model-performance overview.

Two QC figures independent of the variant scoring:
  - a grid of the v1.1 catalog motif logos, annotated with their TF matches
    (ports 2025_11_04/motif_wall), and
  - per-cell-type ChromBPNet performance (counts correlation and profile JSD)
    from the single-task metrics table.

CPU stage; run in env eugene_tools.

Usage:
    python src/3_report/3_motif_and_model_qc.py --config config/config.yaml
"""

import argparse
from pathlib import Path

import logomaker as lm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tangermeme.io import read_meme

from common import io, metadata
from common.logging_setup import setup_logging
from common.plotting import configure_matplotlib, savefig

log = setup_logging(__file__)
ALPHABET = "ACGT"


def motif_wall(cfg, out_path: Path, n_cols: int = 10) -> None:
    """Grid of catalog motif logos (information content), titled by TF match."""
    meme = str(io.repo_path(cfg, cfg["catalog_meme"]))
    motif_dict = read_meme(meme)
    names = list(motif_dict.keys())
    md = pd.read_csv(io.repo_path(cfg, cfg["catalog_metadata"]), sep="\t").set_index("short_id")
    tf_col = "best_match_tf" if "best_match_tf" in md.columns else md.columns[1]

    n = len(names)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.1 * n_rows), dpi=300)
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        ppm = motif_dict[name].numpy().T
        ic = lm.transform_matrix(pd.DataFrame(ppm, columns=list(ALPHABET)),
                                 from_type="probability", to_type="information")
        lm.Logo(ic, ax=ax, fade_below=0.1)
        tf = md[tf_col].get(name, "")
        ax.set_title(f"{name}\n{tf}", fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    savefig(fig, out_path)
    log.info("motif wall: %d logos -> %s", n, out_path)


def model_performance(cfg, md, stage_colors, out_path: Path) -> None:
    """Per-cell-type counts correlation and profile JSD, ordered and stage-colored."""
    metrics = pd.read_csv(io.repo_path(cfg, "results/3_single_task_models/models/metrics/model_metrics.csv"))
    metrics = metrics.rename(columns={"Cell Type": "cell_type"}).set_index("cell_type")
    order = [c for c in md.index if c in metrics.index]
    metrics = metrics.loc[order]
    colors = md.loc[order, "stage"].map(stage_colors)

    panels = [
        ("counts_metrics_peaks_pearsonr", "Counts Pearson r"),
        ("counts_metrics_peaks_spearmanr", "Counts Spearman r"),
        ("profile_metrics_peaks_median_jsd", "Profile median JSD"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3 * len(panels)), sharex=True)
    for ax, (col, label) in zip(axes, panels):
        ax.bar(metrics.index, metrics[col], color=colors, edgecolor="black")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Cell type")
    axes[-1].tick_params(axis="x", rotation=90)
    axes[0].set_title("ChromBPNet single-task performance (fold_0)")
    fig.tight_layout()
    savefig(fig, out_path)
    log.info("model performance -> %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    configure_matplotlib()
    sns.set_style("white")
    cfg = io.load_config(args.config)
    md = metadata.load_cell_type_metadata(cfg)
    _, stage_colors = metadata.stage_order(cfg)

    qc = io.sandbox_path(cfg, "results/qc")
    qc.mkdir(parents=True, exist_ok=True)
    motif_wall(cfg, qc / "sc-islet-differentiation_motif_logos.pdf")
    model_performance(cfg, md, stage_colors, qc / "model_performance_overview.pdf")

    io.write_provenance(cfg, "3_report_3_motif_and_model_qc", {"catalog_version": cfg["catalog_version"]})
    log.info("Done.")


if __name__ == "__main__":
    main()
