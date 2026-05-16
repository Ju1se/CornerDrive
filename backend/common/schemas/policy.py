"""
Policy schema for FLPG adaptive policy agent.
Defines the tunable parameters and their bounds.

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md
"""

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
import hashlib
import json


class Policy(BaseModel):
    """
    Represents a frozen policy for a specific round.

    The policy agent only tunes bounded parameters.
    L2/L3/L4 remain the deterministic execution layers.
    """

    round_id: int = Field(ge=0, description="Round number this policy applies to")

    # L2 fraud detection threshold
    theta_tol: float = Field(
        default=0.05,
        ge=0.01,
        le=0.10,
        description="L2 fraud threshold (ΔL_main > theta_tol → FRAUD)"
    )

    # L2 beneficial-rarity detection threshold
    theta_rare: float = Field(
        default=-0.03,
        ge=-0.10,
        le=-0.005,
        description="L2 beneficial-rarity corner threshold (ΔL_corner ≤ theta_rare)"
    )

    theta_rarity_main_tol: float = Field(
        default=0.00925,
        ge=0.0,
        le=0.02,
        description=(
            "Strict L2 main-task safety threshold for clean RARITY "
            "(ΔL_corner ≤ theta_rare and ΔL_main ≤ theta_rarity_main_tol → RARITY)"
        )
    )

    # L3 drift threshold
    theta_drift: float = Field(
        default=0.05,
        ge=0.01,
        le=0.10,
        description="L3 drift threshold for golden dataset validation"
    )

    # L1 filter threshold
    cosine_filter_threshold: float = Field(
        default=0.70,
        ge=0.60,
        le=0.95,
        description="L1 suspect filter threshold"
    )

    # L3 recheck probability
    recheck_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=0.50,
        description="Probability of second-pass audit"
    )

    # L4 settlement multipliers
    honest_reward_multiplier: float = Field(
        default=1.0,
        ge=0.80,
        le=1.20,
        description="Honest contribution reward multiplier"
    )

    slash_multiplier: float = Field(
        default=1.0,
        ge=0.50,
        le=2.00,
        description="Fraud penalty multiplier"
    )

    rarity_reward_multiplier: float = Field(
        default=1.0,
        ge=0.50,
        le=2.00,
        description="Beneficial-rarity reward multiplier"
    )

    # Aggregation weight for rare-but-valuable gradients
    corner_weight: float = Field(
        default=1.0,
        ge=0.50,
        le=2.00,
        description="Weight for beneficial rare updates in aggregation"
    )

    # Metadata
    policy_version: str = Field(
        default="1.0.0",
        description="Policy version identifier"
    )

    effective_from_round: int = Field(
        default=0,
        description="Round from which this policy becomes effective"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when policy was created"
    )

    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of policy parameters.
        Used for on-chain commitment.
        """
        policy_dict = {
            "round_id": self.round_id,
            "theta_tol": self.theta_tol,
            "theta_rare": self.theta_rare,
            "theta_rarity_main_tol": self.theta_rarity_main_tol,
            "theta_drift": self.theta_drift,
            "cosine_filter_threshold": self.cosine_filter_threshold,
            "recheck_probability": self.recheck_probability,
            "honest_reward_multiplier": self.honest_reward_multiplier,
            "slash_multiplier": self.slash_multiplier,
            "rarity_reward_multiplier": self.rarity_reward_multiplier,
            "corner_weight": self.corner_weight,
            "policy_version": self.policy_version,
            "effective_from_round": self.effective_from_round,
        }

        canonical = json.dumps(policy_dict, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "round_id": 100,
                "theta_tol": 0.05,
                "theta_rare": -0.03,
                "theta_rarity_main_tol": 0.00925,
                "theta_drift": 0.05,
                "cosine_filter_threshold": 0.70,
                "recheck_probability": 0.0,
                "honest_reward_multiplier": 1.0,
                "slash_multiplier": 1.0,
                "rarity_reward_multiplier": 1.0,
                "corner_weight": 1.0,
                "policy_version": "1.0.0",
                "effective_from_round": 100,
            }
        },
    )


class PolicyBounds(BaseModel):
    """
    Hard bounds for policy parameters as specified in the guide.
    """

    theta_tol: tuple = (0.01, 0.10)
    theta_rare: tuple = (-0.10, -0.005)  # Negative values
    theta_rarity_main_tol: tuple = (0.0, 0.02)
    theta_drift: tuple = (0.01, 0.10)
    cosine_filter_threshold: tuple = (0.60, 0.95)
    recheck_probability: tuple = (0.0, 0.50)
    honest_reward_multiplier: tuple = (0.80, 1.20)
    slash_multiplier: tuple = (0.50, 2.00)
    rarity_reward_multiplier: tuple = (0.50, 2.00)
    corner_weight: tuple = (0.50, 2.00)


class PolicyMaxStep(BaseModel):
    """
    Maximum allowed change per round for each parameter.
    """

    theta_tol: float = 0.005
    theta_rare: float = 0.005
    theta_rarity_main_tol: float = 0.005
    theta_drift: float = 0.005
    cosine_filter_threshold: float = 0.03
    recheck_probability: float = 0.10
    honest_reward_multiplier: float = 0.05
    slash_multiplier: float = 0.20
    rarity_reward_multiplier: float = 0.20
    corner_weight: float = 0.20


# Default initial policy for round 0
DEFAULT_POLICY = Policy(
    round_id=0,
    theta_tol=0.05,
    theta_rare=-0.03,
    theta_rarity_main_tol=0.00925,
    theta_drift=0.05,
    cosine_filter_threshold=0.70,
    recheck_probability=0.0,
    honest_reward_multiplier=1.0,
    slash_multiplier=1.0,
    rarity_reward_multiplier=1.0,
    corner_weight=1.0,
    policy_version="1.0.0",
    effective_from_round=0
)
