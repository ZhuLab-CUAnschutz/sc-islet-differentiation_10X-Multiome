"""Resolve the CREsted enhancer-design API across the 1.7 rename boundary."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DesignAPI:
    in_silico_evolution: Any
    EnhancerOptimizer: Any
    api_style: str
    version: str
    signature: str


def load_design_api() -> DesignAPI:
    import crested

    try:
        from crested.tl.design import EnhancerOptimizer, in_silico_evolution

        api_style = "crested.tl.design (v1.7+)"
    except ImportError:
        EnhancerOptimizer = crested.utils.EnhancerOptimizer
        in_silico_evolution = crested.tl.enhancer_design_in_silico_evolution
        api_style = "legacy aliases (<v1.7)"

    signature = inspect.signature(in_silico_evolution)
    required = {
        "n_mutations",
        "target",
        "model",
        "return_intermediate",
        "no_mutation_flanks",
        "enhancer_optimizer",
        "starting_sequences",
    }
    missing = required.difference(signature.parameters)
    if missing:
        raise RuntimeError(
            "Installed CREsted lacks required sequence-evolution arguments: "
            + ", ".join(sorted(missing))
        )
    return DesignAPI(
        in_silico_evolution=in_silico_evolution,
        EnhancerOptimizer=EnhancerOptimizer,
        api_style=api_style,
        version=str(getattr(crested, "__version__", "unknown")),
        signature=str(signature),
    )
