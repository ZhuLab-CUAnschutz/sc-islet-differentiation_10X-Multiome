"""Cell-type metadata and categorical facets for ordering and coloring plots.

The metadata TSVs live in the repo ``config/``. Cell types order by
``display_order``. Facets used for coloring PCA points (developmental stage,
lineage, endocrine status) each have a ``<facet>_metadata.tsv`` giving the
category order and colors, so figures stay consistent and are edited in one place.
"""

from pathlib import Path

import pandas as pd

from . import io

# Report facets: key -> (metadata file basename, column in cell_type_metadata, legend title).
FACETS = {
    "stage": ("dev_stage", "dev_stage", "Developmental stage"),
    "lineage": ("lineage", "lineage", "Lineage"),
    "endocrine_category": ("endocrine", "endocrine", "Endocrine status"),
}


def load_cell_type_metadata(cfg: dict) -> pd.DataFrame:
    """Cell-type metadata indexed by cell_type, ordered by display_order."""
    path = io.config_dir(cfg) / "cell_type_metadata.tsv"
    df = pd.read_csv(path, sep="\t").sort_values("display_order")
    return df.set_index("cell_type")


def cell_types(cfg: dict) -> list[str]:
    """The 22 cell types in display order."""
    return load_cell_type_metadata(cfg).index.tolist()


def color_map(cfg: dict) -> dict[str, str]:
    """cell_type -> hex color."""
    return load_cell_type_metadata(cfg)["color"].to_dict()


def category(cfg: dict, name: str) -> tuple[list[str], dict[str, str]]:
    """Load config/<name>_metadata.tsv -> (ordered values, value->color).

    The file must have columns ``<name>``, ``color``, ``display_order``.
    """
    path = io.config_dir(cfg) / f"{name}_metadata.tsv"
    df = pd.read_csv(path, sep="\t").sort_values("display_order")
    return df[name].tolist(), df.set_index(name)["color"].to_dict()


def facet(cfg: dict, key: str) -> dict:
    """Return {column, title, order, colors} for a report facet key in FACETS."""
    meta_name, column, title = FACETS[key]
    order, colors = category(cfg, meta_name)
    return {"column": column, "title": title, "order": order, "colors": colors}
