"""Shared metrics + plotting for the seq->expression models (Steps 3-6).

Metrics follow the POC's "both, equally" success bar:
  - across-gene   : per TRACK, correlation across genes (pred vs measured)
  - across-track  : per GENE, correlation across the 22 cell types  <- the hard one
  - marker recovery: does the model rank known markers specific to the right state?

All fully functional (numpy/pandas/scipy/sklearn/matplotlib) and independent of the
decima API, so they work for the zero-shot baseline and every trained arm alike.

Convention: `pred` and `meas` are pandas DataFrames indexed by track (cell type),
columns = genes, already aligned to the same tracks x genes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# Marker genes -> cell type(s) they should be specific to (edit as needed).
# Types must match config/cell_type_metadata.tsv `cell_type` values.
DEFAULT_MARKERS = {
    "INS": ["late_SC_beta", "early_SC_beta"],
    "IAPP": ["late_SC_beta", "early_SC_beta"],
    "GCG": ["late_SC_alpha", "early_SC_alpha"],
    "ARX": ["late_SC_alpha", "early_SC_alpha"],
    "TPH1": ["late_SC_EC", "early_SC_EC"],
    "SLC18A1": ["late_SC_EC", "early_SC_EC"],
    "LMX1A": ["late_SC_EC", "early_SC_EC"],
    "SST": ["SC_delta_GHRL"],
    "GHRL": ["SC_delta_GHRL"],
    "PDX1": ["PP1", "PP2"],
    "NKX6-1": ["PP1", "PP2"],
    "NEUROG3": ["early_ENP", "late_ENP", "ENP_phase1"],
    "SOX17": ["DE"],
    "FOXA2": ["DE"],
}


# ---------------------------------------------------------------- normalization
def cpm_log1p(counts_df: pd.DataFrame) -> pd.DataFrame:
    """CPM + log1p on a (tracks x genes) summed-counts matrix (per-track library)."""
    lib = counts_df.sum(axis=1).replace(0, np.nan)
    cpm = counts_df.div(lib, axis=0) * 1e6
    return np.log1p(cpm)


def align(pred: pd.DataFrame, meas: pd.DataFrame):
    """Align two (tracks x genes) frames to shared tracks and genes."""
    tracks = [t for t in meas.index if t in pred.index]
    genes = meas.columns.intersection(pred.columns)
    return pred.loc[tracks, genes], meas.loc[tracks, genes]


# --------------------------------------------------------------------- metrics
def _corr(x, y, method):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    f = spearmanr if method == "spearman" else pearsonr
    return f(x[ok], y[ok])[0]


def across_gene_correlation(pred, meas, method="spearman") -> pd.Series:
    """Per-track correlation across genes. Series indexed by track."""
    pred, meas = align(pred, meas)
    return pd.Series(
        {t: _corr(pred.loc[t], meas.loc[t], method) for t in pred.index}, name="across_gene"
    )


def across_track_correlation(pred, meas, method="spearman") -> pd.Series:
    """Per-gene correlation across tracks (cell-type specificity). Series by gene.

    Spearman is the default because Decima's zero-shot output is on a different
    (relative/normalized) scale than our counts; ranks are the fair comparison.
    """
    pred, meas = align(pred, meas)
    return pd.Series(
        {g: _corr(pred[g], meas[g], method) for g in pred.columns}, name="across_track"
    )


def marker_auroc(pred: pd.DataFrame, markers=None) -> tuple[float, pd.DataFrame]:
    """Marker recovery: for each marker gene, treat its annotated cell type(s) as
    positives and rank all tracks by predicted expression -> per-marker AUROC.
    Returns (mean AUROC, per-marker table)."""
    markers = markers or DEFAULT_MARKERS
    rows = []
    for gene, pos_types in markers.items():
        if gene not in pred.columns:
            continue
        scores = pred[gene]
        labels = scores.index.isin(pos_types).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        try:
            auroc = roc_auc_score(labels, scores.values)
        except ValueError:
            auroc = np.nan
        rows.append({"gene": gene, "auroc": auroc, "pos_types": ",".join(pos_types)})
    tab = pd.DataFrame(rows)
    return (float(np.nanmean(tab["auroc"])) if len(tab) else np.nan), tab


def summarize(pred, meas, markers=None, method="spearman") -> dict:
    """One row of the scorecard: median across-gene r, median across-track r, marker AUROC."""
    ag = across_gene_correlation(pred, meas, method)
    at = across_track_correlation(pred, meas, method)
    mauroc, _ = marker_auroc(pred, markers)
    return {
        "across_gene_median": float(np.nanmedian(ag)),
        "across_track_median": float(np.nanmedian(at)),
        "marker_auroc": mauroc,
    }


# ------------------------------------------------------------------ scorecard
def update_scorecard(scorecard_csv: str, arm: str, metrics: dict):
    """Append/replace a row for `arm` and re-render the scorecard PNG next to it."""
    os.makedirs(os.path.dirname(scorecard_csv), exist_ok=True)
    cols = ["arm", "across_gene_median", "across_track_median", "marker_auroc"]
    if os.path.exists(scorecard_csv):
        df = pd.read_csv(scorecard_csv)
    else:
        df = pd.DataFrame(columns=cols)
    df = df[df["arm"] != arm]
    row = {"arm": arm, **{k: metrics.get(k) for k in cols[1:]}}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[cols]
    df.to_csv(scorecard_csv, index=False)

    _render_scorecard(df, scorecard_csv.replace(".csv", ".png"))
    return df


def _render_scorecard(df: pd.DataFrame, png_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["across_gene_median", "across_track_median", "marker_auroc"]
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 3, 0.5 * len(df) + 1.5))
    data = df.set_index("arm")[metrics].astype(float)
    im = ax.imshow(data.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["across-gene\n(median)", "across-track\n(median)", "marker\nAUROC"])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.3f}",
                    ha="center", va="center",
                    color="white" if (np.isfinite(v) and v < 0.6) else "black")
    ax.set_title("Seq→expression POC scorecard")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- shared figures
def fig_across_gene_scatter(pred, meas, tracks, path, label="prediction"):
    """Per-track measured-vs-predicted scatter for a few tracks (e.g. design targets)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = [t for t in tracks if t in pred.index and t in meas.index]
    n = max(len(tracks), 1)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow), squeeze=False)
    for ax, t in zip(axes.ravel(), tracks):
        x, y = meas.loc[t], pred.loc[t]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=3, alpha=0.2)
        r = _corr(x, y, "spearman")
        ax.set_title(f"{t}  (r={r:.2f})", fontsize=8)
        ax.set_xlabel("measured (log1p CPM)")
        ax.set_ylabel(label)
    for ax in axes.ravel()[len(tracks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_across_track_violin(at_series, path, title="Cell-type specificity"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = at_series.dropna().values
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(vals):
        ax.violinplot(vals, showmedians=True)
    ax.set_ylabel("per-gene across-track Spearman r")
    ax.set_title(f"{title} (median={np.nanmedian(vals):.3f})")
    ax.axhline(0, ls="--", c="grey", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- config colors
def cell_type_colors():
    """Best-effort load of project cell-type colors; else None."""
    try:
        from config.loader import load_cell_type_colors  # type: ignore

        return load_cell_type_colors()
    except Exception:
        return None
