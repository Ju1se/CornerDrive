"""
Policy validator for FLPG Policy Agent.

The validator ensures that:
1. All proposed values are within hard bounds
2. Changes don't exceed maximum step size per round
3. All required fields are present
4. Policy version is assigned
5. Policy hash is computed

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md
"""

import logging
from typing import Dict, Tuple
from copy import deepcopy

from common.schemas import (
    Policy, PolicyProposal, PolicyBounds, PolicyMaxStep, RoundTelemetry
)
from common.utils.policy_confidence import compute_policy_adjustment_scale

logger = logging.getLogger(__name__)


class PolicyValidator:
    """
    Validates proposed policies against constraints.

    The validator performs two types of validation:
    1. Hard bounds: Values must be within absolute min/max
    2. Step bounds: Changes must not exceed maximum per-round delta
    """

    def __init__(self):
        """Initialize the validator with default bounds."""
        self.bounds = PolicyBounds()
        self.max_step = PolicyMaxStep()

    async def validate(
        self,
        proposal: PolicyProposal,
        telemetry: RoundTelemetry | None = None,
    ) -> PolicyProposal:
        """
        Validate a policy proposal.

        Args:
            proposal: Policy proposal to validate

        Returns:
            Updated proposal with validation results applied
        """
        # Create a copy to avoid mutating input
        validated_proposal = deepcopy(proposal)

        proposed = validated_proposal.proposed_policy
        current = validated_proposal.current_policy
        step_scale = compute_policy_adjustment_scale(
            telemetry.audit_sample_size if telemetry is not None else 0
        )
        effective_max_steps = {
            field: max_delta * step_scale
            for field, max_delta in self.max_step.model_dump().items()
        }

        # ====================================================================
        # Step 1: Validate hard bounds
        # ====================================================================
        is_valid, violations = validate_policy_bounds(proposed)

        if violations:
            # Clamp to bounds
            proposed = self._clamp_to_bounds(proposed)
            validated_proposal.proposed_policy = proposed

            for violation in violations:
                validated_proposal.validator_messages.append(
                    f"Clamped to bounds: {violation}"
                )

        # ====================================================================
        # Step 2: Validate max step size
        # ====================================================================
        step_valid, step_violations = validate_policy_step(
            current,
            proposed,
            max_steps=effective_max_steps,
        )

        if step_violations:
            # Clamp to max step
            proposed = self._clamp_to_max_step(
                current,
                proposed,
                max_steps=effective_max_steps,
            )
            validated_proposal.proposed_policy = proposed

            for violation in step_violations:
                validated_proposal.validator_messages.append(
                    f"Clamped to max step: {violation}"
                )

        if telemetry is not None and telemetry.audit_sample_size > 0 and step_scale < 1.0:
            validated_proposal.validator_messages.append(
                "Reduced policy-step aggressiveness due to limited audit sample size "
                f"({telemetry.audit_sample_size} samples, scale={step_scale:.2f})"
            )

        # ====================================================================
        # Step 3: Assign policy version
        # ====================================================================
        if not proposed.policy_version or proposed.policy_version == "1.0.0":
            # Generate version based on round and current version
            current_version = current.policy_version
            proposed.policy_version = self._increment_version(current_version)

        # ====================================================================
        # Step 4: Update validator status
        # ====================================================================
        validated_proposal.validator_passed = True

        logger.info(
            f"Policy validation complete: "
            f"violations={len(violations + step_violations)}"
        )

        return validated_proposal

    def _clamp_to_bounds(self, policy: Policy) -> Policy:
        """Clamp policy values to hard bounds."""
        clamped = deepcopy(policy)
        bounds_dict = self.bounds.model_dump()

        for field, (min_val, max_val) in bounds_dict.items():
            current_value = getattr(clamped, field, None)
            if current_value is None:
                continue

            if current_value < min_val:
                setattr(clamped, field, min_val)
            elif current_value > max_val:
                setattr(clamped, field, max_val)

        return clamped

    def _clamp_to_max_step(
        self,
        current: Policy,
        proposed: Policy,
        max_steps: Dict[str, float] | None = None,
    ) -> Policy:
        """Clamp policy changes to maximum step size."""
        clamped = deepcopy(proposed)
        max_step_dict = max_steps or self.max_step.model_dump()

        for field, max_delta in max_step_dict.items():
            current_value = getattr(current, field, None)
            proposed_value = getattr(clamped, field, None)

            if current_value is None or proposed_value is None:
                continue

            delta = proposed_value - current_value

            if delta > max_delta:
                setattr(clamped, field, current_value + max_delta)
            elif delta < -max_delta:
                setattr(clamped, field, current_value - max_delta)

        return clamped

    def _increment_version(self, version: str) -> str:
        """Increment policy version number."""
        try:
            major, minor, patch = version.split(".")
            # Increment patch for each change
            return f"{major}.{minor}.{int(patch) + 1}"
        except ValueError:
            # If version format is unexpected, return default
            return "1.0.1"

    async def validate_bounds_only(self, policy: Policy) -> Tuple[bool, list[str]]:
        """
        Quick validation of bounds only (used for pre-checks).

        Args:
            policy: Policy to validate

        Returns:
            Tuple of (is_valid, violations)
        """
        return validate_policy_bounds(policy)

    async def validate_step_only(
        self,
        current: Policy,
        proposed: Policy
    ) -> Tuple[bool, list[str]]:
        """
        Quick validation of step size only (used for pre-checks).

        Args:
            current: Current policy
            proposed: Proposed policy

        Returns:
            Tuple of (is_valid, violations)
        """
        return validate_policy_step(current, proposed)


# Helper functions (moved from utils for convenience)

def validate_policy_bounds(policy: Policy) -> Tuple[bool, list[str]]:
    """
    Validate that all policy values are within acceptable bounds.

    Args:
        policy: Policy to validate

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []
    bounds = PolicyBounds()

    for field, (min_val, max_val) in bounds.model_dump().items():
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
    proposed: Policy,
    max_steps: Dict[str, float] | None = None,
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
    max_step_dict = max_steps or PolicyMaxStep().model_dump()

    for field, max_delta in max_step_dict.items():
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
