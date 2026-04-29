"""
Safety guard for FLPG Policy Agent.

The safety guard implements context-aware rules that block dangerous
proposals based on telemetry patterns.

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md

Safety rules from the guide:
1. If false_slash_estimate > 0.05, do not allow slash_multiplier increase.
2. If corner_accuracy is falling and rarity_retention_rate is low,
   do not tighten rarity filtering.
3. If drift looks novelty-like or ambiguous, do not punish before extra review.
4. If hash_mismatch_rate rises, do not lower recheck_probability.
5. If fraud_rate > 0.30, do not simultaneously lower both theta_tol
   strictness and recheck_probability.
"""

import logging
from typing import Optional
from copy import deepcopy

from common.schemas import Policy, RoundTelemetry, PolicyProposal
from policy_agent.engine.policy_modes import build_policy_decision_context

logger = logging.getLogger(__name__)


class SafetyGuard:
    """
    Context-aware safety guard for policy proposals.

    Implements a conservative set of proposal blocks so uncertain novelty does
    not get treated like clear fraud.
    Returns (passed, blocked_reasons) tuple.
    """

    def __init__(self):
        """Initialize the safety guard."""
        pass

    async def check(
        self,
        proposal: PolicyProposal,
        telemetry: Optional[RoundTelemetry] = None
    ) -> PolicyProposal:
        """
        Check a policy proposal for safety issues.

        Args:
            proposal: Policy proposal to check
            telemetry: Optional round telemetry for context

        Returns:
            Updated proposal with safety guard results applied
        """
        if telemetry is None:
            logger.warning("Safety guard called without telemetry, skipping checks")
            proposal.safety_guard_passed = True
            return proposal

        # Create a copy to avoid mutating input
        checked_proposal = deepcopy(proposal)

        current = checked_proposal.current_policy
        proposed = checked_proposal.proposed_policy

        # Run the 5 safety checks
        passed, blocked_reasons = self._check_safety(
            current_policy=current,
            proposed_policy=proposed,
            telemetry=telemetry
        )

        checked_proposal.safety_guard_passed = passed
        checked_proposal.blocked_reasons = blocked_reasons

        logger.info(
            f"Safety guard check: passed={passed}, "
            f"{len(blocked_reasons)} blocks"
        )

        return checked_proposal

    def _check_safety(
        self,
        current_policy: Policy,
        proposed_policy: Policy,
        telemetry: RoundTelemetry
    ) -> tuple[bool, list[str]]:
        """
        Apply all 5 safety rules from the guide.

        Returns:
            Tuple of (passed, blocked_reasons)
        """
        blocked = []
        context = build_policy_decision_context(telemetry)

        # ====================================================================
        # Safety Rule 1: If false-slash risk is elevated, do not allow
        #                 slash_multiplier increase.
        # ====================================================================
        if context.false_slash_risk_high:
            if proposed_policy.slash_multiplier > current_policy.slash_multiplier:
                blocked.append(
                    "Blocked slash increase due to high false-slash estimate"
                )

        # ====================================================================
        # Safety Rule 2: If rarity preservation mode is active, do not tighten
        #                 rarity filtering or add stronger slashing.
        # ====================================================================
        if context.rarity_under_recall:
            if proposed_policy.theta_rare < current_policy.theta_rare:
                blocked.append(
                    "Blocked tighter beneficial-rarity threshold due to weak corner retention"
                )
            if proposed_policy.slash_multiplier > current_policy.slash_multiplier:
                blocked.append(
                    "Blocked slash increase during beneficial-rarity preservation mode"
                )

        # ====================================================================
        # Safety Rule 3: Elevated drift is a warning, not a verdict. Under
        #                 novelty-like or ambiguous drift, do not tighten drift
        #                 checks or add stronger slashing before extra review.
        # ====================================================================
        if context.novel_rarity_like_drift:
            if proposed_policy.theta_drift < current_policy.theta_drift:
                blocked.append(
                    "Blocked punitive drift tightening under novelty-like drift"
                )
            if proposed_policy.slash_multiplier > current_policy.slash_multiplier:
                blocked.append(
                    "Blocked slash increase while drift looks like beneficial rarity"
                )
        elif context.drift_ambiguous:
            if proposed_policy.slash_multiplier > current_policy.slash_multiplier:
                blocked.append(
                    "Blocked slash increase while drift signal remains ambiguous"
                )
            if (
                proposed_policy.theta_drift < current_policy.theta_drift
                and proposed_policy.recheck_probability <= current_policy.recheck_probability
            ):
                blocked.append(
                    "Blocked drift tightening without extra recheck under ambiguous drift"
                )

        # ====================================================================
        # Safety Rule 4: If hash_mismatch_rate > 0.02, do not lower
        #                 recheck_probability.
        # ====================================================================
        if telemetry.hash_mismatch_rate > 0.02:
            if proposed_policy.recheck_probability < current_policy.recheck_probability:
                blocked.append(
                    "Blocked lower recheck probability due to hash mismatch increase"
                )

        # ====================================================================
        # Safety Rule 5: If fraud_rate > 0.30, do not simultaneously lower
        #                 both theta_tol strictness and recheck_probability.
        # ====================================================================
        if telemetry.fraud_rate > 0.30:
            theta_tol_relaxed = proposed_policy.theta_tol > current_policy.theta_tol
            recheck_lowered = proposed_policy.recheck_probability < current_policy.recheck_probability

            if theta_tol_relaxed and recheck_lowered:
                blocked.append(
                    "Blocked simultaneous relaxation of fraud threshold and recheck "
                    "under high fraud rate"
                )

        return len(blocked) == 0, blocked

    def clear_history(self):
        """Clear any stored history (for compatibility with tests)."""
        pass
