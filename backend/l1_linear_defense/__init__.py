"""
FLPG L1: Linear Defense
Byzantine-robust gradient aggregation with outlier detection.

Core functionality:
- Geometric median aggregation using Weiszfeld algorithm
- Cosine deviation filtering for suspect detection
- Batch processing and routing to L2 audit
"""

from .aggregation import AggregationResult, filter_suspects, geometric_median, cosine_similarity
from .server import app, GradientSubmission, SubmissionResponse, BatchResult, HealthResponse

__version__ = "1.0.0"
__author__ = "FLPG Team"

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