"""Config loading, path resolution, score IO, and provenance for all stages.

Config values are read once from ``config/config.yaml``. Relative input paths
resolve against the work dir; catalog and config paths resolve against the repo. Variant
scores join credible-set annotations on the canonical id ``chr:end:a1:a2``.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# variant_scores.tsv columns emitted by the current variant-scorer. Stages select
# by name (variant_id, logfc, abs_logfc_x_jsd.pval), so column order does not matter.
SCORE_COLS = [
    "chr", "pos", "end", "allele1", "allele2", "variant_id",
    "allele1_pred_counts", "allele2_pred_counts",
    "logfc", "abs_logfc", "jsd", "original_jsd", "logfc_x_jsd", "abs_logfc_x_jsd",
    "logfc.pval", "abs_logfc.pval", "jsd.pval", "logfc_x_jsd.pval", "abs_logfc_x_jsd.pval",
]


def load_config(config_path: str | Path) -> dict:
    """Load config.yaml and record its own resolved directory as ``_config_dir``."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def sandbox_path(cfg: dict, rel: str) -> Path:
    """Resolve a sandbox-relative path (e.g. an input) to an absolute Path."""
    return Path(cfg["sandbox"]) / rel


def repo_path(cfg: dict, rel: str) -> Path:
    """Resolve a repo-relative path (e.g. a catalog file) to an absolute Path."""
    return Path(cfg["repo"]) / rel


def config_dir(cfg: dict) -> Path:
    """Directory holding the cell-type metadata and facet TSVs.

    Defaults to the repo ``config/`` so metadata has a single source; override with
    ``config_dir`` in config.yaml to point at a private copy.
    """
    return Path(cfg.get("config_dir") or repo_path(cfg, "config"))


def model_path(cfg: dict, cell_type: str) -> Path:
    """Absolute path to a cell type's bias-corrected ChromBPNet model."""
    return Path(cfg["model_root"]) / cell_type / cfg["model_subpath"]


def load_variant_scores(scores_dir: Path, cell_type: str) -> pd.DataFrame:
    """Load one cell type's variant_scores.tsv, indexed by canonical variant_id.

    Raises FileNotFoundError with an actionable message if the file is missing.
    """
    path = Path(scores_dir) / cell_type / f"{cell_type}.variant_scores.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing scores for {cell_type}: {path}")
    df = pd.read_csv(path, sep="\t")
    return df.set_index("variant_id")


def build_score_matrices(scores_dir: Path, cell_types: list[str],
                         cols=("logfc", "abs_logfc_x_jsd.pval")):
    """Return one variant x cell-type DataFrame per requested score column.

    All cell types must cover the same variant set; mismatches fail loudly.
    """
    frames = {c: {} for c in cols}
    ref_index = None
    for ct in cell_types:
        df = load_variant_scores(scores_dir, ct)
        if ref_index is None:
            ref_index = df.index
        elif not df.index.equals(ref_index):
            if set(df.index) != set(ref_index):
                raise ValueError(f"{ct} variant set differs from the first cell type")
            df = df.reindex(ref_index)
        for c in cols:
            frames[c][ct] = df[c]
    return tuple(pd.DataFrame(frames[c]) for c in cols)


def load_catalog_tf(cfg: dict) -> pd.Series:
    """v1.1 catalog short_id (ST##) -> TF label, curator_tf preferred over best_match_tf.

    This is the single source for TF names across all deliverables, so every motif
    is rooted in the catalog and its ST## id.
    """
    md = pd.read_csv(repo_path(cfg, cfg["catalog_metadata"]), sep="\t")
    cur = md["curator_tf"].astype("string").str.strip()
    tf = cur.where(cur.notna() & (cur != ""), md["best_match_tf"].astype("string"))
    return pd.Series(tf.fillna("").values, index=md["short_id"].values)


def catalog_label(short_id, tf_map: pd.Series) -> str:
    """Format a motif as 'ST##:TF' (falls back to ST## alone if no TF)."""
    tf = tf_map.get(short_id, "") if short_id is not None else ""
    return f"{short_id}:{tf}" if tf else str(short_id)


def load_hit_map(cfg: dict) -> dict:
    """variant_id -> called-hit info from results/hits/hits_disruption.tsv.

    Each value is a dict: ``cts`` (cell types with a hit), ``motifs`` (union of
    ST##:TF), ``by_ct`` ({cell_type: motifs ordered by count in that cell type}),
    ``tf_of`` ({ST##:TF -> curator TF}), ``counts`` (motif -> total rows). Empty
    dict if the hits file is absent. The by_ct map lets callers pick the motif that
    matches a variant's strongest cell types rather than a global vote.
    """
    p = sandbox_path(cfg, "results/hits/hits_disruption.tsv")
    if not p.exists():
        return {}
    h = pd.read_csv(p, sep="\t", usecols=["variant_id", "cell_type", "tf", "motif"])
    out: dict = {}
    for vid, g in h.groupby("variant_id"):
        by_ct = {ct: list(sub["motif"].value_counts().index) for ct, sub in g.groupby("cell_type")}
        out[vid] = {
            "cts": sorted(set(g["cell_type"])),
            "motifs": sorted(set(g["motif"].astype(str))),
            "by_ct": by_ct,
            "tf_of": dict(zip(g["motif"], g["tf"])),
            "counts": g["motif"].value_counts(),
        }
    return out


def primary_hit_for(info: dict | None, ct_order) -> tuple:
    """Dominant hit motif keyed to the strongest cell types, not a global vote.

    Walk ``ct_order`` (cell types by descending |effect|) and return the top motif
    called in the first cell type that has a hit, as (``ST##:TF``, curator TF). Falls
    back to the globally most frequent motif, then to ('none', 'none').
    """
    if not info:
        return ("none", "none")
    for ct in ct_order:
        ms = info["by_ct"].get(ct)
        if ms:
            return (ms[0], info["tf_of"].get(ms[0], ""))
    m = info["counts"].index[0]
    return (m, info["tf_of"].get(m, ""))


def order_hits_by_ct(info: dict | None, ct_order) -> list:
    """Union of a variant's hit motifs, ordered so the strongest cell types' motifs
    come first (then any remaining), for a relevance-ordered disrupts-hit list."""
    if not info:
        return []
    seen: set = set()
    ordered: list = []
    for ct in ct_order:
        for m in info["by_ct"].get(ct, []):
            if m not in seen:
                seen.add(m); ordered.append(m)
    for m in info["motifs"]:
        if m not in seen:
            seen.add(m); ordered.append(m)
    return ordered


def load_credible_sets(cfg: dict) -> pd.DataFrame:
    """Load the frozen credible-set annotation (one row per variant x trait)."""
    return pd.read_csv(sandbox_path(cfg, cfg["credible_sets"]), sep="\t")


def trait_variants(credible_sets: pd.DataFrame, trait_key: str) -> pd.DataFrame:
    """Rows of the credible-set table for one trait, keyed by canonical id."""
    sub = credible_sets[credible_sets["trait"] == trait_key]
    if sub.empty:
        raise ValueError(f"No variants for trait {trait_key!r} in credible_sets")
    return sub


def _git_sha(repo_dir: str | Path) -> str | None:
    """Short git sha of a checkout, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_provenance(cfg: dict, stage: str, params: dict) -> Path:
    """Write a per-stage provenance JSON: params, versions, seed, ref/catalog id.

    Records the variant-scorer git sha so a run pins the exact scoring code.
    """
    prov = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.get("seed"),
        "ref_build": cfg.get("ref_build"),
        "catalog_version": cfg.get("catalog_version"),
        "variant_scorer_sha": _git_sha(Path(cfg["scorer_dir"]).parent),
        "params": params,
    }
    out = Path(cfg["sandbox"]) / "provenance" / f"{stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(prov, f, indent=2)
    return out
