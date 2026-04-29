"""
Shared policy decision context for the FLPG adaptive policy agent.

This module converts raw telemetry into interpretable operating modes so the
GLM engine, deterministic fallback, and safety guard can reason consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.schemas import RoundTelemetry
from common.utils.policy_confidence import (
    compute_policy_adjustment_scale,
    describe_policy_sample_band,
)


@dataclass(frozen=True)
class PolicyDecisionContext:
    """Derived operating modes for one telemetry snapshot."""

    fraud_pressure_high: bool
    rarity_under_recall: bool
    false_slash_risk_high: bool
    drift_warning: bool
    novel_rarity_like_drift: bool
    harmful_shift_like_drift: bool
    drift_ambiguous: bool
    low_sample_confidence: bool
    sample_size_adjustment_scale: float
    sample_size_band: str
    active_modes: tuple[str, ...]
    needs_glm_reasoning: bool

    def to_prompt_dict(self) -> dict:
        """Serialize the context for logs and LLM prompts."""
        if self.novel_rarity_like_drift:
            drift_mode = "novel_rarity_like"
        elif self.harmful_shift_like_drift:
            drift_mode = "harmful_shift_like"
        elif self.drift_ambiguous:
            drift_mode = "ambiguous"
        else:
            drift_mode = "stable"

        return {
            "active_modes": list(self.active_modes),
            "fraud_pressure_high": self.fraud_pressure_high,
            "rarity_under_recall": self.rarity_under_recall,
            "false_slash_risk_high": self.false_slash_risk_high,
            "drift_warning": self.drift_warning,
            "drift_interpretation": drift_mode,
            "low_sample_confidence": self.low_sample_confidence,
            "sample_size_adjustment_scale": self.sample_size_adjustment_scale,
            "sample_size_band": self.sample_size_band,
            "needs_glm_reasoning": self.needs_glm_reasoning,
        }


def build_policy_decision_context(telemetry: RoundTelemetry) -> PolicyDecisionContext:
    """
    Derive high-level operating modes from telemetry.

    The thresholds are intentionally conservative. Fraud pressure and rarity
    under-recall can often be handled deterministically, while drift-heavy or
    overlapping scenarios are treated as complex and better suited for GLM.
    """

    fraud_pressure_high = (
        telemetry.fraud_rate >= 0.18
        and (
            telemetry.main_loss_delta_avg > 0.0
            or telemetry.recent_attack_pressure >= 0.25
        )
    )

    rarity_under_recall = (
        telemetry.rarity_rate <= 0.03
        and telemetry.corner_accuracy <= 0.70
        and telemetry.corner_loss_delta_avg < 0.0
        and telemetry.main_loss_delta_avg <= 0.01
    )

    false_slash_risk_high = (
        telemetry.false_slash_estimate >= 0.08
        or (
            telemetry.false_slash_estimate >= 0.05
            and (
                telemetry.rarity_retention_rate < 0.80
                or telemetry.corner_accuracy <= 0.70
            )
        )
    )

    drift_warning = telemetry.golden_drift_score >= 0.06

    novel_rarity_like_drift = drift_warning and (
        telemetry.corner_loss_delta_avg < -1e-6
        and telemetry.main_loss_delta_avg <= 0.005
        and (
            telemetry.rarity_rate <= 0.05
            or telemetry.corner_accuracy <= 0.75
            or telemetry.rarity_retention_rate < 0.85
        )
    )

    harmful_shift_like_drift = drift_warning and (
        telemetry.main_loss_delta_avg > 0.01
        and telemetry.corner_loss_delta_avg >= -0.001
        and (
            telemetry.fraud_rate >= 0.10
            or telemetry.recent_attack_pressure >= 0.20
        )
    )

    drift_ambiguous = drift_warning and not novel_rarity_like_drift and not harmful_shift_like_drift
    sample_size_adjustment_scale = compute_policy_adjustment_scale(telemetry.audit_sample_size)
    sample_size_band = describe_policy_sample_band(telemetry.audit_sample_size)
    low_sample_confidence = telemetry.audit_sample_size > 0 and sample_size_adjustment_scale < 1.0

    active_modes = tuple(
        name
        for name, active in (
            ("fraud_suppression", fraud_pressure_high),
            ("rarity_preservation", rarity_under_recall),
            ("uncertainty_protection", false_slash_risk_high),
            ("drift_warning", drift_warning),
            ("low_sample_caution", low_sample_confidence),
        )
        if active
    )

    top_level_count = sum(
        (
            fraud_pressure_high,
            rarity_under_recall,
            false_slash_risk_high,
            drift_warning,
        )
    )
    needs_glm_reasoning = drift_warning or top_level_count > 1

    return PolicyDecisionContext(
        fraud_pressure_high=fraud_pressure_high,
        rarity_under_recall=rarity_under_recall,
        false_slash_risk_high=false_slash_risk_high,
        drift_warning=drift_warning,
        novel_rarity_like_drift=novel_rarity_like_drift,
        harmful_shift_like_drift=harmful_shift_like_drift,
        drift_ambiguous=drift_ambiguous,
        low_sample_confidence=low_sample_confidence,
        sample_size_adjustment_scale=sample_size_adjustment_scale,
        sample_size_band=sample_size_band,
        active_modes=active_modes,
        needs_glm_reasoning=needs_glm_reasoning,
    )
