"""
Shared utilities for FLPG.
"""

from .bounds import (
    clamp_value,
    clamp_delta,
    validate_policy_bounds,
    validate_policy_step,
    apply_safety_clamps,
)
from .hashing import (
    compute_policy_hash,
    compute_commitment,
    verify_commitment,
    sign_policy_hash,
)

__all__ = [
    # Bounds
    "clamp_value",
    "clamp_delta",
    "validate_policy_bounds",
    "validate_policy_step",
    "apply_safety_clamps",

    # Hashing
    "compute_policy_hash",
    "compute_commitment",
    "verify_commitment",
    "sign_policy_hash",
]
