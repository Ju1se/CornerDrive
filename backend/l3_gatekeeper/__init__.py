"""
FLPG L3: Global Validation
The Gatekeeper - Validates aggregated model updates against golden dataset.

Core functionality:
- Golden dataset drift validation
- Model checkpoint and rollback
- Commit hash generation for blockchain
"""

from .validator import (
    ValidationDecision,
    ValidationResult,
    GoldenDatasetManager,
    Gatekeeper,
    SimpleMLP,
)

__version__ = "1.0.0"
__author__ = "FLPG Team"

__all__ = [
    "ValidationDecision",
    "ValidationResult",
    "GoldenDatasetManager",
    "Gatekeeper",
    "SimpleMLP",
]