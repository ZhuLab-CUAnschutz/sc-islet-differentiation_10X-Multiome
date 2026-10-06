"""
plot_locus.py
=============

Plotting helpers for cross-cell-type ChromBPNet locus predictions produced by
1_predict_contribs.py. Generalizable to any locus / construct .npz.

Figures
-------
- predicted_accessibility_barplot : per-cell-type total predicted counts,
  trajectory-ordered & config-colored (tests reporter ON/OFF readout).
- cross_celltype_panel : per-cell-type ROWS x 2 COLUMNS
      left  = predicted accessibility track (shared y-scale)
      right = count-contribution logo with TOMTOM-labeled seqlets
- named_tf_scan : annotate seqlets in the contribution logo with a dict of
  named TF PWMs (EOMES / T-TBXT / FOXH1) and report match positions/p-values.

Run under eugene_tools env.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tangermeme.plot import plot_logo
from tangermeme.seqlet import recursive_seqlets
from tangermeme.annotate import annotate_seqlets, read_meme

import chrombpnet_locus as cl
import config
from config import (HOCOMOCO_MEME, CHROMBPNET_CATALOG_MEME, ANNOTATION_DBS,
                    TF_COLORS)

# back-compat aliases (config is the source of truth)
NAMED_TF_HOCOMOCO = config.NAMED_TF
VIERSTRA_MEME = config.VIERSTRA_CONSENSUS_MEME


# ----------------------------------------------------------------------------
# npz access
# ----------------------------------------------------------------------------
class LocusResults:
    """Thin accessor over the saved .npz."""

    def __init__(self, npz_path):
        self.d = np.load(npz_path, allow_pickle=True)
        self.name = str(self.d["name"])
        self.celltypes = [str(c) for c in self.d["celltypes"]]
        self.variants = [str(v) for v in self.d["variants"]]
        self.input_len = int(self.d["input_len"])
        self.output_len = int(self.d["output_len"])
        self.out_start, self.out_end = cl.output_window(self.input_len, self.output_len)

    def variant_offset(self, variant):
        """Offset of the original locus within this variant's window (0 for the
        as-is construct; 233 for the N/shuffle-centered enhancer)."""
        k = f"meta_offset__{variant}"
        return int(self.d[k]) if k in self.d else 0

    def counts(self, variant, ct):
        return float(self.d[f"counts__{variant}__{ct}"])

    def profile(self, variant, ct):
        return self.d[f"profile__{variant}__{ct}"]

    def pred(self, variant, ct):
        return self.d[f"pred__{variant}__{ct}"]

    def has_profile(self, variant=None):
        """True if base-resolution profile/pred is stored (single-task ChromBPNet).
        Multitask outputs only a per-cell-type scalar -> False -> logos-only panel."""
        v = variant or self.variants[0]
        c = self.celltypes[0]
        return f"pred__{v}__{c}" in self.d

    def attr(self, variant, ct):
        return self.d[f"attr__{variant}__{ct}"]            # (4, input_len)

    def ohe(self, variant):
        return self.d[f"ohe__{variant}"]                   # (4, input_len)

    def ordered_celltypes(self, subset=None):
        order = cl.trajectory_order()
        cts = subset or self.celltypes
        return [c for c in order if c in cts]


# ----------------------------------------------------------------------------
# coordinate helpers (everything in INPUT coordinates, 0..input_len)
# ----------------------------------------------------------------------------
def _profile_slice(res, variant, ct, rs, re):
    """Predicted signal (counts*profile) over input-coord [rs,re).
    Returns nan outside the model output window."""
    full = np.full(res.input_len, np.nan)
    full[res.out_start:res.out_end] = res.pred(variant, ct)
    return full[rs:re]


# ----------------------------------------------------------------------------
# bar plot: predicted accessibility across cell types
# ----------------------------------------------------------------------------
def predicted_accessibility_barplot(res, variant="construct", subset=None,
                                    log=True, ax=None, figsize=(10, 4)):
    cts = res.ordered_celltypes(subset)
    cmap = cl.color_map()
    vals = [res.counts(variant, c) for c in cts]
    if log:
        vals = np.log1p(vals)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(cts)), vals, color=[cmap[c] for c in cts],
           edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels(cts, rotation=90)
    ax.set_ylabel("log(1+predicted counts)" if log else "predicted counts")
    ax.set_title(f"{res.name} — predicted accessibility [{variant}]")
    return ax


# ----------------------------------------------------------------------------
# seqlets + annotation
# ----------------------------------------------------------------------------
def _proj_attr(res, variant, ct):
    """Projected (observed-base) count contributions for one cell type, (L,)."""
    return (res.attr(variant, ct) * res.ohe(variant)).sum(axis=0)


def call_seqlets_all(res, variant, subset=None, threshold=0.05,
                     min_seqlet_len=5, max_seqlet_len=25, region=None,
                     normalize=False):
    """Robust seqlet calling across cell types.

    recursive_seqlets estimates its p-value null from the distribution of
    attribution sums; with a SINGLE 2114 bp track that estimate is fragile. Here
    we stack the projected contributions for ALL cell types into (n_ct, L) and
    call recursive_seqlets ONCE, so the null is pooled across tracks. Seqlets are
    returned per example (tagged with `celltype`).

    normalize : if True, z-scale each track before stacking so the pooled null
        is not dominated by high-count models (per-model count-logit scales
        differ). Reported attributions are then in SD units. Default False
        (raw), which lets genuinely weak cell types (e.g. PGT3) call fewer hits.
    """
    cts = res.ordered_celltypes(subset)
    tracks = np.stack([_proj_attr(res, variant, c) for c in cts])      # (n_ct, L)
    if normalize:
        sd = tracks.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        tracks = tracks / sd
    seqlets = recursive_seqlets(tracks, threshold=threshold,
                                min_seqlet_len=min_seqlet_len,
                                max_seqlet_len=max_seqlet_len).reset_index(drop=True)
    ex = seqlets.iloc[:, 0].astype(int).values
    seqlets["celltype"] = [cts[i] for i in ex]
    if region is not None:
        rs, re = region
        mid = (seqlets.iloc[:, 1] + seqlets.iloc[:, 2]) / 2
        seqlets = seqlets[(mid >= rs) & (mid < re)].copy()
    return seqlets


def call_seqlets(res, variant, ct, threshold=0.05, min_seqlet_len=5,
                 max_seqlet_len=25, region=None):
    """Single-cell-type seqlet calling (fragile null; prefer call_seqlets_all).
    Kept for convenience / single-track use."""
    proj = _proj_attr(res, variant, ct)[None, :]
    seqlets = recursive_seqlets(proj, threshold=threshold,
                                min_seqlet_len=min_seqlet_len,
                                max_seqlet_len=max_seqlet_len)
    if region is not None:
        rs, re = region
        mid = (seqlets.iloc[:, 1] + seqlets.iloc[:, 2]) / 2
        seqlets = seqlets[(mid >= rs) & (mid < re)].copy()
    return seqlets


def load_db(spec):
    """Resolve a motif-DB spec into (labels_list, pwm_dict).
    spec may be: a MEME path str; a {label: hocomoco_id} dict (resolved against
    HOCOMOCO); or a {label: pwm} dict."""
    if isinstance(spec, str):
        mm = read_meme(spec)
        return list(mm.keys()), mm
    if isinstance(spec, dict) and spec and isinstance(next(iter(spec.values())), str):
        mm = read_meme(HOCOMOCO_MEME)
        d = {label: mm[mid] for label, mid in spec.items()}
        return list(d.keys()), d
    return list(spec.keys()), spec


def annotate(res, variant, seqlets, motifs=NAMED_TF_HOCOMOCO, n_nearest=1):
    """Annotate seqlets with nearest motif via TOMTOM against any DB (`motifs` =
    a {label:id} dict, a {label:pwm} dict, or a MEME path).

    Returns a DataFrame: start, end, label, pvalue. The same construct sequence
    underlies every cell type, so the example index is forced to 0 (single ohe).
    """
    labels, mdict = load_db(motifs)
    ohe = torch.tensor(res.ohe(variant)[None].astype("float32"))   # (1,4,L)
    s = seqlets.reset_index(drop=True).copy()
    s.iloc[:, 0] = 0                                               # all map to ohe[0]
    idx, pval = annotate_seqlets(ohe, s, mdict, n_nearest=n_nearest)
    idx = idx[:, 0].cpu().numpy()
    pval = pval[:, 0].cpu().numpy()
    return pd.DataFrame({
        "start": s.iloc[:, 1].values,
        "end": s.iloc[:, 2].values,
        "label": [labels[i] if 0 <= i < len(labels) else None for i in idx],
        "pvalue": pval,
    })


# ----------------------------------------------------------------------------
# main two-column panel
# ----------------------------------------------------------------------------
def _shared_attr_ylim(res, variant, cts, rs, re, pad=1.1):
    """Common (vmin, vmax) for projected count contributions across all rows in
    the region, so contribution logos are on the SAME y-scale."""
    lo, hi = 0.0, 0.0
    for c in cts:
        proj = (res.attr(variant, c) * res.ohe(variant))[:, rs:re]
        lo = min(lo, float(proj.min()))
        hi = max(hi, float(proj.max()))
    return lo * pad, hi * pad


def _short_label(name):
    """Shorten a Vierstra archetype name ('AC0490:TBX/EOMES:T-box AC0490:...')
    to a compact family tag ('TBX/EOMES'); pass other labels through."""
    s = str(name).split()[0]
    parts = s.split(":")
    return parts[1] if len(parts) >= 2 else s


def _seqlet_annotations(res, variant, seqs_ct, db, pthresh=0.05, label_fn=_short_label):
    """Build a plot_logo `annotations` DataFrame from called seqlets.
    Seqlets with a DB match at p<pthresh are named by their motif; the rest are
    shown as unlabeled 'seqlet' bars (so all boundaries appear). Coords ABSOLUTE.
    """
    if not len(seqs_ct):
        return None
    ann = annotate(res, variant, seqs_ct, motifs=db)
    names, starts, ends, scores = [], [], [], []
    for (_, sr), (_, a) in zip(seqs_ct.iterrows(), ann.iterrows()):
        sig = (a["label"] is not None) and (a["pvalue"] < pthresh)
        names.append(label_fn(a["label"]) if sig else "seqlet")
        starts.append(int(sr.iloc[1])); ends.append(int(sr.iloc[2]))
        scores.append(round(-np.log10(a["pvalue"]), 1) if sig else 0.0)
    return pd.DataFrame({"motif_name": names, "start": starts, "end": ends, "score": scores})


def cross_celltype_panel(res, variant="construct", region=None, subset=None,
                         annot_db=VIERSTRA_MEME, seqlet_thresh=0.05, annot_pthresh=0.05,
                         figsize=None, shared_acc_scale=True, shared_attr_scale=True,
                         annotate_logos=True):
    """Per-cell-type rows; left=predicted accessibility, right=contribution logo
    with seqlets annotated natively by `plot_logo` (labeled bars below the logo).

    `region`=(rs,re) in INPUT coords (defaults to the model output window).
    `annot_db` is the DB used to name seqlets (default full Vierstra; may be a
    MEME path or a {label:id}/{label:pwm} dict). Contribution logos share one
    symmetric y-scale across rows when `shared_attr_scale`. Seqlets are called
    once across all cell-type tracks (pooled null; call_seqlets_all).
    """
    cts = res.ordered_celltypes(subset)
    if region is None:
        region = (res.out_start, res.out_end)
    rs, re = region
    cmap = cl.color_map()
    accessibility = res.has_profile(variant)        # multitask has no profile -> logos only

    all_seqs = call_seqlets_all(res, variant, subset=cts,
                                threshold=seqlet_thresh, region=region) \
        if annotate_logos else None

    # shared scales
    acc_max = 0.0
    if accessibility and shared_acc_scale:
        for c in cts:
            seg = _profile_slice(res, variant, c, rs, re)
            if np.isfinite(seg).any():
                acc_max = max(acc_max, np.nanmax(seg))
    # plot_logo uses a SCALAR symmetric ylim when annotating
    lo, hi = _shared_attr_ylim(res, variant, cts, rs, re, pad=1.05)
    attr_scalar = max(abs(lo), abs(hi)) if shared_attr_scale else None

    n = len(cts)
    ncols = 2 if accessibility else 1
    figsize = figsize or (16 if accessibility else 13, 1.7 * n + 1)
    fig, axes = plt.subplots(n, ncols, figsize=figsize, squeeze=False,
                             gridspec_kw={"width_ratios": [1, 2.6]} if accessibility else None)

    for i, c in enumerate(cts):
        ax_logo = axes[i, 1] if accessibility else axes[i, 0]

        if accessibility:
            # left: predicted accessibility track (counts * profile)
            ax_acc = axes[i, 0]
            seg = _profile_slice(res, variant, c, rs, re)
            ax_acc.fill_between(np.arange(rs, re), np.nan_to_num(seg), color=cmap[c], lw=0)
            ax_acc.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=9)
            if shared_acc_scale and acc_max > 0:
                ax_acc.set_ylim(0, acc_max * 1.02)
            ax_acc.set_xlim(rs, re); ax_acc.set_yticks([])
            ax_acc.text(0.98, 0.9, f"n={res.counts(variant,c):.0f}",
                        transform=ax_acc.transAxes, ha="right", va="top", fontsize=8)
            if i == 0:
                ax_acc.set_title("predicted accessibility (total counts = n)", fontsize=10)
        else:
            # no profile (multitask): label row + per-cell-type scalar prediction on the logo axis
            ax_logo.set_ylabel(f"{c}\n(pred={res.counts(variant,c):.0f})",
                               rotation=0, ha="right", va="center", fontsize=8,
                               color=cmap[c])

        # contribution logo with native plot_logo seqlet annotations
        proj_attr = res.attr(variant, c) * res.ohe(variant)
        annots = None
        if all_seqs is not None:
            annots = _seqlet_annotations(res, variant, all_seqs[all_seqs.celltype == c],
                                         annot_db, pthresh=annot_pthresh)
        plot_logo(proj_attr, ax=ax_logo, start=rs, end=re,
                  annotations=annots, ylim=attr_scalar)
        ax_logo.set_yticks([])
        # plot_logo uses relative x (0..width); relabel ticks to absolute coords
        xt = np.linspace(0, re - rs, 5)
        ax_logo.set_xticks(xt)
        ax_logo.set_xticklabels([str(int(rs + t)) for t in xt])
        if i == 0:
            ax_logo.set_title(f"count contributions + seqlets (named vs "
                              f"{'Vierstra' if isinstance(annot_db,str) else 'TFs'})",
                              fontsize=10)

    fig.suptitle(f"{res.name} — {variant}  region {rs}-{re}", y=1.0, fontsize=12)
    fig.tight_layout()
    return fig, axes


def seqlet_table(res, variant="construct", region=None, subset=None,
                 dbs=ANNOTATION_DBS, seqlet_thresh=0.05):
    """Tidy table of called seqlet boundaries for every cell type, annotated
    against EACH database in `dbs` (default: catalog, named3, vierstra).

    Seqlets are called once across all cell-type tracks (pooled null;
    call_seqlets_all). For each DB the nearest motif + TOMTOM p-value are added
    as columns `<db>_label` / `<db>_pvalue`. Coords are INPUT coordinates.
    """
    seqs = call_seqlets_all(res, variant, subset=subset,
                            threshold=seqlet_thresh, region=region)
    if not len(seqs):
        return pd.DataFrame()
    seqs = seqs.reset_index(drop=True)
    counts = {c: round(res.counts(variant, c), 1) for c in res.ordered_celltypes(subset)}
    out = pd.DataFrame({
        "celltype": seqs["celltype"].values,
        "counts": [counts[c] for c in seqs["celltype"].values],
        "start": seqs.iloc[:, 1].astype(int).values,
        "end": seqs.iloc[:, 2].astype(int).values,
        "length": (seqs.iloc[:, 2] - seqs.iloc[:, 1]).astype(int).values,
        "feature": [config.feature_of_span(int(s), int(e))
                    for s, e in zip(seqs.iloc[:, 1], seqs.iloc[:, 2])],
        "seqlet_attribution": seqs["attribution"].round(4).values,
        "seqlet_pvalue": seqs["p-value"].values,
    })
    for db_name, spec in (dbs or {}).items():
        ann = annotate(res, variant, seqs, motifs=spec)
        out[f"{db_name}_label"] = ann["label"].values
        out[f"{db_name}_pvalue"] = ann["pvalue"].values
    return out.sort_values(["celltype", "start"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# FIMO-hit contribution across cell types (trajectory dynamics)
# ----------------------------------------------------------------------------
def fimo_contribution_matrix(res, fimo_df, variant="centered", subset=None,
                             signed=True):
    """For each FIMO hit, the model's count-contribution summed over the hit
    window, per cell type. Connects sequence motif occurrences (FIMO, locus
    coords) to model-attributed importance, across the developmental trajectory.

    fimo_df: FIMO hits (cols motif_id,start,stop,strand,q-value); start/stop are
    1-based locus coords (the FASTA scanned == the locus inside each variant).
    Returns a DataFrame: one row per hit, columns = hit metadata + one per cell
    type (trajectory-ordered). `signed`=False uses sum|contribution|.
    """
    off = res.variant_offset(variant)
    cts = res.ordered_celltypes(subset)
    proj = {c: (res.attr(variant, c) * res.ohe(variant)).sum(0) for c in cts}
    rows = []
    for _, h in fimo_df.iterrows():
        s0 = int(h["start"]) - 1 + off          # FIMO is 1-based -> 0-based + offset
        e0 = int(h["stop"]) + off
        rec = {"motif": h["motif_id"], "start": int(h["start"]), "stop": int(h["stop"]),
               "strand": h["strand"], "qvalue": float(h.get("q-value", np.nan)),
               "feature": h.get("feature", config.feature_of_span(int(h["start"]), int(h["stop"]))),
               "id": f"{h['motif_id']}@{int(h['start'])}{h['strand']}"}
        for c in cts:
            seg = proj[c][s0:e0]
            rec[c] = float(seg.sum() if signed else np.abs(seg).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# context sweeps (slide / keepcenter) — shared plotter
# ----------------------------------------------------------------------------
def plot_context_sweep(df, mode, ax_abs=None, ax_norm=None):
    """Plot a context sweep produced by 06_context_sweep.py.

    mode="slide"      -> x = enhancer offset (0=left .. center .. right)
    mode="keepcenter" -> x = kept central width (large=full construct .. small)
    Two axes: absolute counts (mean±sd over seeds) and counts normalized to a
    reference (centered offset for slide; full construct for keepcenter)."""
    cmap = cl.color_map()
    order = [c for c in cl.trajectory_order() if c in df["celltype"].unique()]
    g = df.groupby(["celltype", "x"])["counts"].agg(["mean", "std"]).reset_index()
    xs = sorted(df["x"].unique(), reverse=(mode == "keepcenter"))
    ref_x = xs[len(xs) // 2] if mode == "slide" else max(xs)   # center / full
    if ax_abs is None or ax_norm is None:
        import matplotlib.pyplot as plt
        _, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(16, 6))

    for c in order:
        s = g[g.celltype == c].set_index("x").reindex(xs).reset_index()
        ax_abs.plot(range(len(xs)), s["mean"], "-o", ms=3, color=cmap[c], lw=1.4, label=c)
        ax_abs.fill_between(range(len(xs)), s["mean"] - s["std"], s["mean"] + s["std"],
                            color=cmap[c], alpha=0.13, lw=0)
        ref = g[(g.celltype == c) & (g.x == ref_x)]["mean"].values[0]
        if ref:
            ax_norm.plot(range(len(xs)), (s["mean"] / ref).values, "-o", ms=3,
                         color=cmap[c], lw=1.2, label=c)
    for ax in (ax_abs, ax_norm):
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=0, fontsize=8)
        ax.axvline(xs.index(ref_x), color="k", ls="--", lw=1)
    xlab = ("enhancer offset (bp): 0=left, center, right" if mode == "slide"
            else "kept central width (bp): full → enhancer → core (flanks dinuc-shuffled)")
    ax_abs.set_xlabel(xlab); ax_norm.set_xlabel(xlab)
    ax_abs.set_ylabel("predicted counts (mean±sd over seeds)")
    ax_norm.set_ylabel(f"counts / counts({'centered' if mode=='slide' else 'full'})")
    ax_norm.axhline(1.0, color="0.6", lw=0.6)
    ax_abs.set_title("absolute"); ax_norm.set_title("normalized")
    ax_abs.legend(fontsize=6, ncol=2, loc="best")
    return ax_abs, ax_norm


# ----------------------------------------------------------------------------
# PWM logo helpers (for TF / archetype displays)
# ----------------------------------------------------------------------------
def ic_scale(pwm):
    """Probability matrix (4,w) -> information-content-scaled (bits) for logos."""
    p = np.clip(np.asarray(pwm, dtype=float), 1e-6, 1)
    p = p / p.sum(0, keepdims=True)
    return p * (2 + (p * np.log2(p)).sum(0))


def pwm_logo_b64(pwm, w=2.0, h=0.6):
    """Render an IC logo of a PWM to a base64 PNG data URI (for HTML embedding)."""
    import io, base64
    fig, ax = plt.subplots(figsize=(w, h))
    try:
        plot_logo(ic_scale(pwm), ax=ax)
    except Exception:
        pass
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
