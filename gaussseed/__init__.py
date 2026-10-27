"""Public API for GaussSeed."""

from typing import TYPE_CHECKING

from .pipeline import (
    BackendParameters,
    CanonicalGaussians,
    FilterConfig,
    GaussSeedPipeline,
    PipelineConfig,
    PipelineResult,
    adapt_backend_parameters,
    apply_similarity,
    filter_candidates,
    novelty_from_map_depth,
    project_to_view,
    robust_depth_scale,
)

__version__ = "0.1.0"

if TYPE_CHECKING:
    from .predictor import GaussSeed, GaussSeedPrediction


def __getattr__(name: str):
    """Load the neural proposal wrapper only when its Hydra stack is requested."""

    if name in {"GaussSeed", "GaussSeedPrediction"}:
        from .predictor import GaussSeed, GaussSeedPrediction

        return {"GaussSeed": GaussSeed, "GaussSeedPrediction": GaussSeedPrediction}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BackendParameters",
    "CanonicalGaussians",
    "FilterConfig",
    "GaussSeed",
    "GaussSeedPipeline",
    "GaussSeedPrediction",
    "PipelineConfig",
    "PipelineResult",
    "adapt_backend_parameters",
    "apply_similarity",
    "filter_candidates",
    "novelty_from_map_depth",
    "project_to_view",
    "robust_depth_scale",
    "__version__",
]
