"""
Utility functions for policy bounds validation.
"""

from typing import Any, Tuple
from ..schemas.policy import Policy, PolicyBounds, PolicyMaxStep


def clamp_value(value: float, bounds: Tuple[float, float]) -> float:
    """
    Clamp a value to specified bounds.

    Args:
        value: Value to clamp
        bounds: (min, max) tuple

    Returns:
        Clamped value
    """
    min_val, max_val = bounds
    return max(min_val, min(max_val, value))


def clamp_delta(
    current: float,
    proposed: float,
    max_delta: float
) -> float:
    """
    Clamp the change between two values to a maximum delta.

    Args:
        current: Current value
        proposed: Proposed value
        max_delta: Maximum allowed change

    Returns:
        Adjusted proposed value respecting max delta
    """
    delta = proposed - current
    clamped_delta = max(-max_delta, min(max_delta, delta))
    return current + clamped_delta


def validate_policy_bounds(policy: Policy) -> Tuple[bool, list[str]]:
    """
    Validate that all policy values are within acceptable bounds.

    Args:
        policy: Policy to validate

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []
    bounds = PolicyBounds().model_dump()

    for field, (min_val, max_val) in bounds.items():
        value = getattr(policy, field, None)
        if value is None:
            continue

        if value < min_val or value > max_val:
            violations.append(
                f"{field}={value} outside bounds [{min_val}, {max_val}]"
            )

    return len(violations) == 0, violations


def validate_policy_step(
    current: Policy,
    proposed: Policy
) -> Tuple[bool, list[str]]:
    """
    Validate that policy changes don't exceed maximum step size.

    Args:
        current: Current policy
        proposed: Proposed policy

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []
    max_steps = PolicyMaxStep().model_dump()

    for field, max_delta in max_steps.items():
        current_val = getattr(current, field, None)
        proposed_val = getattr(proposed, field, None)

        if current_val is None or proposed_val is None:
            continue

        delta = abs(proposed_val - current_val)
        if delta > max_delta:
            violations.append(
                f"{field} change {delta} exceeds max step {max_delta}"
            )

    return len(violations) == 0, violations


def apply_safety_clamps(
    current: Policy,
    proposed: Policy,
    telemetry: Any = None
) -> tuple[Policy, list[str]]:
    """
    Apply additional safety clamps based on telemetry context.

    This implements the safety guard rules from the implementation guide.

    Args:
        current: Current policy
        proposed: Proposed policy
        telemetry: Optional telemetry data for context-aware clamping

    Returns:
        Tuple of (safely clamped policy, reasons)
    """
    import copy

    result = copy.deepcopy(proposed)
    reasons = []

    # Rule: Block increased slashing if false slash estimate is high
    if telemetry and hasattr(telemetry, 'false_slash_estimate'):
        if telemetry.false_slash_estimate > 0.05:
            if result.slash_multiplier > current.slash_multiplier:
                result.slash_multiplier = current.slash_multiplier
                reasons.append(
                    "Blocked: Cannot increase slash_multiplier with "
                    f"high false slash estimate ({telemetry.false_slash_estimate:.2%})"
                )

    # Rule: Be cautious with theta_rare if corner accuracy is low
    if telemetry and hasattr(telemetry, 'corner_accuracy'):
        if telemetry.corner_accuracy < 0.70:
            # Don't make theta_rare stricter (lower) when corner performance is weak
            if result.theta_rare < current.theta_rare:
                result.theta_rare = current.theta_rare
                reasons.append(
                    "Blocked: Cannot tighten theta_rare with "
                    f"low corner accuracy ({telemetry.corner_accuracy:.2%})"
                )

    # Rule: Don't reduce audit strictness when drift is high
    if telemetry and hasattr(telemetry, 'golden_drift_score'):
        if telemetry.golden_drift_score > 0.15:
            # Don't reduce theta_tol (make it more permissive for fraud)
            if result.theta_tol > current.theta_tol:
                result.theta_tol = current.theta_tol
                reasons.append(
                    "Blocked: Cannot relax theta_tol with "
                    f"high golden drift ({telemetry.golden_drift_score:.2%})"
                )

    # Rule: Don't reduce recheck probability when hash mismatches are high
    if telemetry and hasattr(telemetry, 'hash_mismatch_rate'):
        if telemetry.hash_mismatch_rate > 0.01:
            if result.recheck_probability < current.recheck_probability:
                result.recheck_probability = current.recheck_probability
                reasons.append(
                    "Blocked: Cannot reduce recheck_probability with "
                    f"high hash mismatch rate ({telemetry.hash_mismatch_rate:.2%})"
                )

    return result, reasons
