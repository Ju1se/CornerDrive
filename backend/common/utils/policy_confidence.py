"""
Helpers for turning audit sample sizes into policy confidence signals.
"""

from __future__ import annotations


def compute_policy_adjustment_scale(audit_sample_size: int) -> float:
    """
    Convert audit sample size into a multiplier for policy-step aggressiveness.

    `0` means "unknown / unavailable" for backward compatibility and does not
    dampen changes. Positive small samples reduce how far policy may move.
    """
    if audit_sample_size <= 0:
        return 1.0
    if audit_sample_size < 10:
        return 0.2
    if audit_sample_size < 25:
        return 0.4
    if audit_sample_size < 50:
        return 0.6
    if audit_sample_size < 100:
        return 0.8
    return 1.0


def describe_policy_sample_band(audit_sample_size: int) -> str:
    """Return a coarse label for audit-sample confidence."""
    if audit_sample_size <= 0:
        return "unknown"
    if audit_sample_size < 10:
        return "tiny"
    if audit_sample_size < 25:
        return "small"
    if audit_sample_size < 50:
        return "medium"
    if audit_sample_size < 100:
        return "large"
    return "high"
