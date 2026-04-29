"""
Rule-based policy proposal engine for FLPG Policy Agent.

This is the V1 deterministic engine that:
1. Analyzes telemetry to identify system state
2. Applies predefined rules to adjust policy parameters
3. Generates clear, explainable reasons for changes
4. Respects bounded parameter constraints

This engine is deterministic and reproducible, making the system
auditable and explainable.

IMPORTANT: theta_rare directionality
- L2判定: ΔL_corner ≤ theta_rare → RARITY
- theta_rare 是负值
- theta_rare += 0.002 (如 -0.05 → -0.048): 更宽松，更多梯度被判为RARITY
- theta_rare -= 0.002 (如 -0.03 → -0.032): 更严格，更少梯度被判为RARITY

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md
"""

import logging

from common.schemas import (
    Policy, RoundTelemetry, PolicyProposal,
    PolicyBounds, PolicyMaxStep
)
from common.utils.bounds import clamp_value, clamp_delta
from policy_agent.engine.policy_modes import build_policy_decision_context

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Deterministic rule-based policy proposal engine.

    Implements four mode-oriented behaviors:

    1. Fraud suppression: tighten theta_tol, add recheck, only modestly raise slash.
    2. Rarity preservation: relax theta_rare, raise rarity reward, avoid punitive slash growth.
    3. Uncertainty protection: reduce slashing and prefer recheck when false-slash risk rises.
    4. Drift warning handling: treat drift as warning-first, then separate
       novelty-like drift from harmful shift.
    """

    def __init__(self):
        """Initialize the rule engine."""
        # Policy bounds for clamping
        self.bounds = PolicyBounds()
        self.max_step = PolicyMaxStep()

    async def propose(
        self,
        current_policy: Policy,
        telemetry: RoundTelemetry
    ) -> PolicyProposal:
        """
        Propose next-round policy based on telemetry.

        Args:
            current_policy: Current frozen policy
            telemetry: Round telemetry data

        Returns:
            PolicyProposal with proposed policy and reasons
        """
        # Start with a copy of current policy
        excluded_fields = {"round_id", "effective_from_round", "created_at"}
        proposed_dict = {
            key: value
            for key, value in current_policy.model_dump().items()
            if key not in excluded_fields
        }
        current_dict = proposed_dict.copy()

        reasons = []
        context = build_policy_decision_context(telemetry)

        if context.fraud_pressure_high:
            proposed_dict["theta_tol"] -= 0.002
            proposed_dict["recheck_probability"] += 0.05
            if not context.rarity_under_recall and not context.false_slash_risk_high:
                proposed_dict["slash_multiplier"] += 0.05
                reasons.append(
                    "Fraud pressure is high; tightened theta_tol, raised recheck, and moderately increased slashing"
                )
            else:
                reasons.append(
                    "Fraud pressure is high; tightened theta_tol and raised recheck without extra slashing to avoid harming beneficial-rare or uncertain contributors"
                )

        if context.rarity_under_recall:
            proposed_dict["theta_rare"] += 0.002
            proposed_dict["rarity_reward_multiplier"] += 0.10
            proposed_dict["recheck_probability"] += 0.05
            if proposed_dict["slash_multiplier"] > current_dict["slash_multiplier"]:
                proposed_dict["slash_multiplier"] = current_dict["slash_multiplier"]
            reasons.append(
                "Beneficial-rarity under-recall detected; relaxed theta_rare, raised the beneficial-rarity reward, and kept punitive slashing from increasing"
            )

        if context.false_slash_risk_high:
            proposed_dict["slash_multiplier"] -= 0.10
            proposed_dict["recheck_probability"] += 0.05
            reasons.append(
                "False-slash risk is high; reduced slashing and routed more cases through recheck before settlement"
            )

        if context.drift_warning:
            proposed_dict["recheck_probability"] += 0.05
            if context.novel_rarity_like_drift:
                proposed_dict["theta_rare"] += 0.001
                if proposed_dict["slash_multiplier"] > current_dict["slash_multiplier"]:
                    proposed_dict["slash_multiplier"] = current_dict["slash_multiplier"]
                reasons.append(
                    "Drift looks novelty-like rather than harmful; treated drift as warning-first, preserved beneficial rarity, and avoided stronger slashing"
                )
            elif context.harmful_shift_like_drift:
                proposed_dict["theta_drift"] -= 0.002
                reasons.append(
                    "Drift appears harmful to the main task; tightened theta_drift and raised recheck for confirmation"
                )
                if context.fraud_pressure_high and not context.false_slash_risk_high and not context.rarity_under_recall:
                    proposed_dict["slash_multiplier"] += 0.05
                    reasons.append(
                        "Coupled harmful drift with fraud pressure; allowed a modest additional slashing increase"
                    )
            else:
                if proposed_dict["slash_multiplier"] > current_dict["slash_multiplier"]:
                    proposed_dict["slash_multiplier"] = current_dict["slash_multiplier"]
                reasons.append(
                    "Drift signal is ambiguous; increased recheck and deferred punitive tightening pending shadow re-evaluation"
                )

        if telemetry.suspect_queue_length > 100:
            proposed_dict["cosine_filter_threshold"] += 0.01
            reasons.append(
                "Suspect queue is large; tightened the L1 suspicious filter to keep up with review volume"
            )

        if (
            telemetry.audit_sample_size > 0
            and telemetry.audit_sample_size < 40
            and telemetry.honest_rate >= 0.65
            and not context.fraud_pressure_high
            and not context.false_slash_risk_high
        ):
            proposed_dict["honest_reward_multiplier"] += 0.02
            reasons.append(
                "Audit volume is thin while honest behavior remains healthy; slightly increased honest reward to support participation"
            )

        if (
            not context.active_modes
            and telemetry.honest_rate > 0.70
            and telemetry.false_slash_estimate < 0.02
        ):
            proposed_dict["recheck_probability"] -= 0.02
            reasons.append(
                "Environment looks stable and mostly honest; slightly reduced recheck load"
            )

            if (
                telemetry.audit_sample_size >= 120
                and current_dict["honest_reward_multiplier"] > 1.0
            ):
                proposed_dict["honest_reward_multiplier"] -= 0.02
                reasons.append(
                    "Healthy honest volume returned; nudged honest reward back toward its baseline"
                )

        if telemetry.rarity_retention_rate > 0.90 and telemetry.corner_accuracy > 0.78:
            proposed_dict["corner_weight"] += 0.05
            reasons.append(
                "Strong beneficial-rarity retention and corner performance; increased beneficial rare update weight"
            )

        if not reasons:
            reasons.append("Telemetry is stable; kept the policy effectively unchanged")

        # ====================================================================
        # Apply bounds and step limits
        # ====================================================================
        validated_dict, validator_messages = self._validate_proposal(
            current_dict=current_dict,
            proposed_dict=proposed_dict
        )

        # Create the proposed policy object
        proposed_policy = Policy(
            round_id=telemetry.round_id + 1,
            effective_from_round=telemetry.round_id + 1,
            **validated_dict
        )

        # Create the proposal
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed_policy,
            reasons=reasons,
            validator_passed=True,  # We just validated it
            safety_guard_passed=False,  # Will be checked by safety guard
            blocked_reasons=[],
            validator_messages=validator_messages,
            round_id=telemetry.round_id + 1,
            source_engine="rule_engine"
        )

        logger.info(
            f"Rule engine proposal: {len(reasons)} changes proposed, "
            f"{len(validator_messages)} validator messages"
        )

        return proposal

    def _validate_proposal(
        self,
        current_dict: dict,
        proposed_dict: dict
    ) -> tuple[dict, list[str]]:
        """
        Validate and clamp proposal to bounds and step limits.

        Args:
            current_dict: Current policy values
            proposed_dict: Proposed policy values (may be out of bounds)

        Returns:
            Tuple of (validated_dict, messages)
        """
        validated = {}
        messages = []

        bounds_dict = self.bounds.model_dump()
        max_step_dict = self.max_step.model_dump()

        for key, proposed_value in proposed_dict.items():
            # Skip non-tunable fields
            if key not in bounds_dict and key not in max_step_dict:
                validated[key] = proposed_value
                continue

            current_value = current_dict.get(key, proposed_value)

            # Apply step limit first
            if key in max_step_dict:
                stepped = clamp_delta(
                    current=current_value,
                    proposed=proposed_value,
                    max_delta=max_step_dict[key]
                )
                if stepped != proposed_value:
                    messages.append(
                        f"Adjusted {key} from {proposed_value} to {stepped} "
                        f"(max step: {max_step_dict[key]})"
                    )
                proposed_value = stepped

            # Apply bounds
            if key in bounds_dict:
                bounded = clamp_value(
                    value=proposed_value,
                    bounds=bounds_dict[key]
                )
                if bounded != proposed_value:
                    messages.append(
                        f"Clamped {key} from {proposed_value} to {bounded} "
                        f"(bounds: {bounds_dict[key]})"
                    )
                validated[key] = bounded
            else:
                validated[key] = proposed_value

        return validated, messages

    def explain_decision(self, proposal: PolicyProposal) -> str:
        """
        Generate a human-readable explanation of the proposal.

        Args:
            proposal: Policy proposal to explain

        Returns:
            Human-readable explanation
        """
        if not proposal.get_diff():
            return "No policy changes were proposed. The current policy is optimal."

        explanation_parts = [
            f"Policy proposal for Round {proposal.round_id}:"
        ]

        if proposal.reasons:
            explanation_parts.append("\nReasons for changes:")
            for reason in proposal.reasons:
                explanation_parts.append(f"  - {reason}")

        if proposal.validator_messages:
            explanation_parts.append("\nValidator adjustments:")
            for msg in proposal.validator_messages:
                explanation_parts.append(f"  - {msg}")

        if proposal.blocked:
            explanation_parts.append(
                "\nPROPOSAL BLOCKED by safety guard. "
                "Cannot apply to next round without override."
            )

        return "\n".join(explanation_parts)
