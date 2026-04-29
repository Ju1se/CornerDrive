"""
FLPG L1: Linear Defense
Byzantine-robust gradient aggregation with outlier detection.

Core functionality:
- Geometric median aggregation using Weiszfeld algorithm
- Cosine deviation filtering for suspect detection
- Batch processing and routing to L2 audit
"""

from .aggregation import AggregationResult, filter_suspects, geometric_median, cosine_similarity

__version__ = "1.0.0"
__author__ = "FLPG Team"


def __getattr__(name):
    if name in {
        "app",
        "GradientSubmission",
        "SubmissionResponse",
        "BatchResult",
        "HealthResponse",
    }:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AggregationResult",
    "filter_suspects",
    "geometric_median",
    "cosine_similarity",
    "app",
    "GradientSubmission",
    "SubmissionResponse",
    "BatchResult",
    "HealthResponse",
]
