"""
Policy proposal schema for FLPG adaptive policy agent.
Defines the output of the policy proposal process.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from .policy import Policy

IGNORED_POLICY_DIFF_FIELDS = {
    "round_id",
    "effective_from_round",
    "policy_version",
    "created_at",
}


class PolicyProposal(BaseModel):
    """
    Output of the policy proposal process.

    Contains the current policy, proposed policy, reasons for changes,
    and validation results.
    """

    current_policy: Policy = Field(description="Current active policy")
    proposed_policy: Policy = Field(description="Proposed next-round policy")

    # Reasons for the proposal
    reasons: List[str] = Field(
        default_factory=list,
        description="Explanations for proposed changes from rule engine"
    )

    # Validation results
    validator_passed: bool = Field(
        default=False,
        description="Whether proposal passed validator checks"
    )

    safety_guard_passed: bool = Field(
        default=False,
        description="Whether proposal passed safety guard checks"
    )

    # Blocked reasons
    blocked_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons why the proposal was blocked"
    )

    # Validator messages
    validator_messages: List[str] = Field(
        default_factory=list,
        description="Messages from validator (clamping, etc.)"
    )

    # Metadata
    round_id: int = Field(description="Round this proposal applies to")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_engine: str = Field(
        default="rule_engine",
        description="Engine that generated this proposal"
    )

    # Approval status
    approved: bool = Field(
        default=False,
        description="Whether this proposal has been approved"
    )

    # Explanation (optional LLM)
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation (from LLM or template)"
    )

    # Complexity analysis (for hybrid LLM mode)
    complexity_score: Optional[float] = Field(
        default=None,
        description="Complexity confidence score (0.0 to 1.0)"
    )

    complexity_reason: Optional[str] = Field(
        default=None,
        description="Reason for complexity flag"
    )

    llm_used: Optional[bool] = Field(
        default=None,
        description="Whether LLM was used for this proposal"
    )

    @property
    def blocked(self) -> bool:
        """Check if the proposal is blocked."""
        return not self.safety_guard_passed

    def get_diff(self) -> dict:
        """
        Get the differences between current and proposed policy.
        Returns {field: (current_value, proposed_value)}
        """
        diff = {}
        current_data = self.current_policy.model_dump()
        proposed_data = self.proposed_policy.model_dump()

        for key in current_data:
            if key in IGNORED_POLICY_DIFF_FIELDS:
                continue
            if key in proposed_data and current_data[key] != proposed_data[key]:
                diff[key] = (current_data[key], proposed_data[key])

        return diff

    def summary(self) -> str:
        """Get a human-readable summary of the proposal."""
        diff = self.get_diff()

        if not diff:
            return "No policy changes were proposed. The current policy is optimal."

        changes = []
        for field, (old, new) in diff.items():
            direction = "↑" if new > old else "↓"
            changes.append(f"{field}: {old} → {new} {direction}")

        return "Policy changes: " + ", ".join(changes)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_policy": {
                    "round_id": 100,
                    "theta_tol": 0.05,
                    "theta_rare": -0.03,
                    "theta_drift": 0.05,
                    "cosine_filter_threshold": 0.70,
                    "recheck_probability": 0.0,
                    "honest_reward_multiplier": 1.0,
                    "slash_multiplier": 1.0,
                    "rarity_reward_multiplier": 1.0,
                    "corner_weight": 1.0,
                    "policy_version": "1.0.0",
                    "effective_from_round": 100,
                },
                "proposed_policy": {
                    "round_id": 101,
                    "theta_tol": 0.048,
                    "theta_rare": -0.028,
                    "theta_drift": 0.048,
                    "cosine_filter_threshold": 0.70,
                    "recheck_probability": 0.0,
                    "honest_reward_multiplier": 1.0,
                    "slash_multiplier": 1.0,
                    "rarity_reward_multiplier": 1.0,
                    "corner_weight": 1.0,
                    "policy_version": "1.0.1",
                    "effective_from_round": 101,
                },
                "reasons": ["High fraud rate detected; tightened fraud threshold"],
                "validator_passed": True,
                "safety_guard_passed": True,
                "blocked_reasons": [],
                "validator_messages": [],
                "round_id": 101,
                "approved": False,
            }
        },
    )
