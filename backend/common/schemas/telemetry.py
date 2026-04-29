"""
Telemetry schema for FLPG adaptive policy agent.
Defines the aggregated metrics collected at the end of each round.

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class RoundTelemetry(BaseModel):
    """
    Aggregated telemetry data collected at the end of each training round.

    This data drives the policy agent's decisions for the next round.
    Matches the exact specification from the implementation guide.
    """

    # Round identification
    round_id: int = Field(ge=0, description="Round number")

    # Classification rates from L2
    fraud_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of clients classified as FRAUD"
    )

    rarity_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of clients classified as beneficial RARITY"
    )

    honest_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of clients classified as HONEST"
    )

    noise_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of clients classified as NOISE"
    )

    # Model performance metrics
    main_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Main task accuracy after the round"
    )

    corner_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Corner case accuracy after the round"
    )

    # Average loss deltas (L2 metrics)
    main_loss_delta_avg: float = Field(
        description="Average change in main task loss (negative = improvement)"
    )

    corner_loss_delta_avg: float = Field(
        description="Average change in corner case loss (negative = improvement)"
    )

    # Safety and quality metrics
    false_slash_estimate: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Estimated false positive rate for fraud detection"
    )

    # L2 beneficial-rarity retention
    rarity_retention_rate: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
        description="Rate at which beneficial rare gradients are retained"
    )

    # L3 validation metrics
    golden_drift_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Drift score on golden dataset"
    )

    reject_rate_l3: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Rate of L3 rejections"
    )

    # L1 specific metrics
    cosine_outlier_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of gradients flagged by cosine filter"
    )

    l1_recheck_ratio: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Fraction of gradients routed by probabilistic L1 recheck"
    )

    # L1 suspect queue
    suspect_queue_length: int = Field(
        ge=0,
        default=0,
        description="Number of gradients in suspect queue"
    )

    audit_sample_size: int = Field(
        ge=0,
        default=0,
        description="Number of classified audit outcomes observed for this round"
    )

    # L4 settlement metrics
    avg_sbt_score: float = Field(
        ge=0.0,
        description="Average SBT score across active participants"
    )

    # Participant metrics
    new_vehicle_ratio: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Fraction of new vehicles this round"
    )

    # Security metrics
    hash_mismatch_rate: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Rate of commit hash mismatches"
    )

    recent_attack_pressure: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Estimated attack pressure (moving average)"
    )

    l1_routed_by_reason: dict[str, int] = Field(
        default_factory=dict,
        description="L1 routed audit counts keyed by routing reason"
    )

    fraud_caught_by_routing_reason: dict[str, int] = Field(
        default_factory=dict,
        description="Fraud catches keyed by the L1 routing reason"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this telemetry snapshot was recorded"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "round_id": 100,
                "fraud_rate": 0.08,
                "rarity_rate": 0.03,
                "honest_rate": 0.75,
                "noise_rate": 0.14,
                "main_accuracy": 0.92,
                "corner_accuracy": 0.68,
                "main_loss_delta_avg": -0.05,
                "corner_loss_delta_avg": -0.02,
                "false_slash_estimate": 0.01,
                "rarity_retention_rate": 0.90,
                "golden_drift_score": 0.03,
                "reject_rate_l3": 0.02,
                "cosine_outlier_ratio": 0.15,
                "l1_recheck_ratio": 0.02,
                "suspect_queue_length": 25,
                "audit_sample_size": 100,
                "avg_sbt_score": 250.0,
                "new_vehicle_ratio": 0.05,
                "hash_mismatch_rate": 0.0,
                "recent_attack_pressure": 0.10,
                "l1_routed_by_reason": {"cosine_screening": 23, "probabilistic_recheck": 2},
                "fraud_caught_by_routing_reason": {"cosine_screening": 4},
            }
        },
    )


class TelemetrySummary(BaseModel):
    """
    Aggregated summary over multiple rounds for trend analysis.
    """

    rounds: list[int] = Field(description="List of round IDs in this summary")
    window_size: int = Field(ge=1, description="Number of rounds in the window")

    # Averages over the window
    avg_fraud_rate: float
    avg_rarity_rate: float
    avg_honest_rate: float
    avg_noise_rate: float

    # Trends
    fraud_trend: str = Field(description="increasing, decreasing, or stable")
    accuracy_trend: str = Field(description="improving, degrading, or stable")

    # Stability metrics
    fraud_volatility: float = Field(description="Standard deviation of fraud rate")
    accuracy_volatility: float = Field(description="Standard deviation of accuracy")

    @classmethod
    def from_telemetry_list(cls, telemetry_list: list[RoundTelemetry]) -> "TelemetrySummary":
        """Create a summary from a list of telemetry objects."""
        if not telemetry_list:
            raise ValueError("Cannot create summary from empty list")

        window_size = len(telemetry_list)
        rounds = [t.round_id for t in telemetry_list]

        fraud_rates = [t.fraud_rate for t in telemetry_list]
        rarity_rates = [t.rarity_rate for t in telemetry_list]
        honest_rates = [t.honest_rate for t in telemetry_list]
        noise_rates = [t.noise_rate for t in telemetry_list]
        accuracies = [t.main_accuracy for t in telemetry_list]

        import statistics

        # Calculate trends
        fraud_trend = "stable"
        if len(fraud_rates) >= 3:
            recent_avg = sum(fraud_rates[-3:]) / 3
            earlier_avg = sum(fraud_rates[:3]) / min(3, len(fraud_rates))
            if recent_avg > earlier_avg * 1.2:
                fraud_trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                fraud_trend = "decreasing"

        accuracy_trend = "stable"
        if len(accuracies) >= 3:
            recent_avg = sum(accuracies[-3:]) / 3
            earlier_avg = sum(accuracies[:3]) / min(3, len(accuracies))
            if recent_avg > earlier_avg * 1.05:
                accuracy_trend = "improving"
            elif recent_avg < earlier_avg * 0.95:
                accuracy_trend = "degrading"

        return cls(
            rounds=rounds,
            window_size=window_size,
            avg_fraud_rate=sum(fraud_rates) / window_size,
            avg_rarity_rate=sum(rarity_rates) / window_size,
            avg_honest_rate=sum(honest_rates) / window_size,
            avg_noise_rate=sum(noise_rates) / window_size,
            fraud_trend=fraud_trend,
            accuracy_trend=accuracy_trend,
            fraud_volatility=statistics.stdev(fraud_rates) if len(fraud_rates) > 1 else 0.0,
            accuracy_volatility=statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        )
