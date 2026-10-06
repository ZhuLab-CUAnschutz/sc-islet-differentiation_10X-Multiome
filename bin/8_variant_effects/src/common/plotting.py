"""Shared plotting setup: vector-safe PDF fonts and a save helper.

Stage-specific figures (PCA, ridgeplots, logos) build their own axes; this module
holds only what every plotting stage shares, so styling never drifts.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    """Embed real fonts in PDFs/PS (type 42) so text stays editable in Illustrator."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def savefig(fig: plt.Figure, path: str | Path, **kwargs) -> None:
    """Save a figure to path (creating parents) with a tight bounding box."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)
