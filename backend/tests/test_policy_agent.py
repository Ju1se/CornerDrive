"""
Unit tests for FLPG Policy Agent.

Tests cover:
- Rule engine with exact 8 rules from guide
- Safety guard with exact 5 rules from guide
- Policy validation (bounds and step limits)
- PolicyProposal schema

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md
"""

import pytest
import asyncio
from datetime import datetime, timezone

from common.schemas import (
    Policy, RoundTelemetry, PolicyProposal,
    PolicyBounds, PolicyMaxStep, DEFAULT_POLICY
)
from policy_agent.engine.glm_policy_engine import GLMPolicyEngine
from policy_agent.engine.policy_modes import build_policy_decision_context
from policy_agent.engine.rule_engine import RuleEngine
from policy_agent.constraints.safety_guard import SafetyGuard
from policy_agent.constraints.validator import PolicyValidator
from policy_agent.storage.redis_store import proposal_recency_key
from common.utils.policy_confidence import compute_policy_adjustment_scale
from common.utils.round_stats import compute_recent_attack_pressure, compute_round_rates


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def current_policy():
    """Create a test current policy."""
    return Policy(
        round_id=100,
        theta_tol=0.05,
        theta_rare=-0.03,
        theta_drift=0.05,
        cosine_filter_threshold=0.70,
        recheck_probability=0.0,
        slash_multiplier=1.0,
        rarity_reward_multiplier=1.0,
        corner_weight=1.0,
        policy_version="1.0.0",
        effective_from_round=100
    )


@pytest.fixture
def sample_telemetry():
    """Create sample round telemetry."""
    return RoundTelemetry(
        round_id=100,
        fraud_rate=0.08,
        rarity_rate=0.03,
        honest_rate=0.75,
        noise_rate=0.14,
        main_accuracy=0.92,
        corner_accuracy=0.68,
        main_loss_delta_avg=-0.05,
        corner_loss_delta_avg=-0.02,
        false_slash_estimate=0.01,
        rarity_retention_rate=0.90,
        golden_drift_score=0.03,
        reject_rate_l3=0.02,
        cosine_outlier_ratio=0.15,
        suspect_queue_length=25,
        avg_sbt_score=250.0,
        new_vehicle_ratio=0.05,
        hash_mismatch_rate=0.0,
        recent_attack_pressure=0.10
    )


# ============================================================================
# Rule Engine Tests (8 exact rules from guide)
# ============================================================================

class TestRuleEngine:
    """Test the rule engine with exact 8 rules from guide."""

    @pytest.mark.asyncio
    async def test_rule_1_high_fraud_rate(self, current_policy):
        """
        Rule 1: If fraud_rate > 0.20, tighten fraud threshold and increase slashing.
        Expected: theta_tol -= 0.002, slash_multiplier += 0.05
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.25,  # High fraud
            rarity_rate=0.05,
            honest_rate=0.50,
            noise_rate=0.22,
            main_accuracy=0.85,
            corner_accuracy=0.75,
            main_loss_delta_avg=0.02,
            corner_loss_delta_avg=-0.01,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.85,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=200.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.30
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check theta_tol tightened (decreased by 0.002)
        assert proposal.proposed_policy.theta_tol == pytest.approx(0.048)
        # Check slash_multiplier increased modestly
        assert proposal.proposed_policy.slash_multiplier == pytest.approx(1.05)
        # Check reason was added
        assert any("fraud" in r.lower() for r in proposal.reasons)

    @pytest.mark.asyncio
    async def test_rule_2_low_corner_accuracy_low_rarity(self, current_policy):
        """
        Rule 2: If corner_accuracy < 0.75 and rarity_rate < 0.05,
                 make rarity detection more permissive and raise rarity reward.
        Expected: theta_rare += 0.002, rarity_reward_multiplier += 0.10
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,  # Low rarity (< 0.05)
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.70,  # Low corner (< 0.75)
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check theta_rare relaxed (increased toward 0 by 0.002)
        assert proposal.proposed_policy.theta_rare == pytest.approx(-0.028)
        # Check rarity_reward_multiplier increased
        assert proposal.proposed_policy.rarity_reward_multiplier == pytest.approx(1.10)
        # Check reason was added
        assert any("rarity" in r.lower() for r in proposal.reasons)

    @pytest.mark.asyncio
    async def test_rule_3_high_false_slash(self, current_policy):
        """
        Rule 3: If false_slash_estimate > 0.05, reduce slashing and increase recheck.
        Expected: slash_multiplier -= 0.10, recheck_probability += 0.05
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.05,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.75,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.08,  # High false slash (> 0.05)
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check slash_multiplier reduced
        assert proposal.proposed_policy.slash_multiplier == pytest.approx(0.90)
        # Check recheck_probability increased
        assert proposal.proposed_policy.recheck_probability == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_rule_supports_honest_participation_when_volume_is_thin(self, current_policy):
        """Low audit volume with healthy honesty should modestly raise honest reward support."""
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.05,
            rarity_rate=0.04,
            honest_rate=0.82,
            noise_rate=0.09,
            main_accuracy=0.93,
            corner_accuracy=0.76,
            main_loss_delta_avg=-0.01,
            corner_loss_delta_avg=-0.01,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.92,
            golden_drift_score=0.02,
            reject_rate_l3=0.01,
            cosine_outlier_ratio=0.08,
            suspect_queue_length=12,
            audit_sample_size=24,
            avg_sbt_score=240.0,
            new_vehicle_ratio=0.04,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.04,
        )

        proposal = await engine.propose(current_policy, telemetry)

        assert proposal.proposed_policy.honest_reward_multiplier == pytest.approx(1.02)
        assert any("honest reward" in reason.lower() for reason in proposal.reasons)

    @pytest.mark.asyncio
    async def test_rule_4_high_golden_drift(self, current_policy):
        """
        Rule 4: Harmful drift should tighten theta_drift and raise recheck.
        Expected: theta_drift -= 0.002, recheck_probability += 0.05
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.12,
            rarity_rate=0.04,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.71,
            main_loss_delta_avg=0.02,
            corner_loss_delta_avg=0.0,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.90,
            golden_drift_score=0.08,  # High drift (> 0.05)
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.24
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check theta_drift tightened (decreased)
        assert proposal.proposed_policy.theta_drift == pytest.approx(0.048)
        # Check recheck_probability increased
        assert proposal.proposed_policy.recheck_probability == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_rule_5_high_suspect_queue(self, current_policy):
        """
        Rule 5: If suspect_queue_length > 100, tighten L1 filter threshold.
        Expected: cosine_filter_threshold += 0.01
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.68,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=150,  # High queue (> 100)
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check cosine_filter_threshold increased (tightened)
        assert proposal.proposed_policy.cosine_filter_threshold == pytest.approx(0.71)

    @pytest.mark.asyncio
    async def test_rule_6_honest_environment(self, current_policy):
        """
        Rule 6: If honest_rate > 0.70 and false_slash_estimate < 0.02,
                 gently reduce recheck probability.
        Expected: recheck_probability -= 0.02
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.05,
            rarity_rate=0.06,
            honest_rate=0.80,  # High honest (> 0.70)
            noise_rate=0.12,
            main_accuracy=0.92,
            corner_accuracy=0.74,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,  # Low false slash (< 0.02)
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check recheck_probability reduced
        # Should be clamped to 0.0 (minimum) since it started at 0.0
        assert proposal.proposed_policy.recheck_probability == pytest.approx(0.0)
        assert any("honest" in r.lower() or "stable" in r.lower() for r in proposal.reasons)

    @pytest.mark.asyncio
    async def test_rule_7_high_attack_pressure(self, current_policy):
        """
        Rule 7: High attack pressure alone does not tighten controls without fraud evidence.
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.68,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.30  # High attack pressure (> 0.25)
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Attack pressure by itself should not tighten theta_tol in the current mode logic
        assert proposal.proposed_policy.theta_tol == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_rule_8_strong_rarity_retention(self, current_policy):
        """
        Rule 8: If rarity_retention_rate > 0.90 and corner_accuracy > 0.78,
                 slightly increase corner_weight.
        Expected: corner_weight += 0.05
        """
        engine = RuleEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.80,  # Good corner (> 0.78)
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.95,  # High retention (> 0.90)
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        proposal = await engine.propose(current_policy, telemetry)

        # Check corner_weight increased
        assert proposal.proposed_policy.corner_weight == pytest.approx(1.05)


# ============================================================================
# Safety Guard Tests (5 exact rules from guide)
# ============================================================================

class TestSafetyGuard:
    """Test the safety guard with exact 5 rules from guide."""

    @pytest.fixture
    def proposal(self, current_policy):
        """Create a test proposal."""
        proposed = Policy(
            round_id=101,
            theta_tol=0.06,  # Increased (relaxed)
            theta_rare=-0.02,  # Increased (relaxed)
            theta_drift=0.06,  # Increased (relaxed)
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.2,  # Increased
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        return PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

    @pytest.mark.asyncio
    async def test_safety_rule_1_high_false_slash_blocks_slash_increase(self, proposal):
        """
        Safety Rule 1: If false_slash_estimate > 0.05, do not allow slash_multiplier increase.
        """
        guard = SafetyGuard()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.68,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.08,  # High (> 0.05)
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        result = await guard.check(proposal, telemetry)

        # Should be blocked
        assert result.blocked is True
        assert any("slash" in r.lower() for r in result.blocked_reasons)

    @pytest.mark.asyncio
    async def test_safety_rule_2_weak_corner_blocks_rarity_tightening(self, current_policy):
        """
        Safety Rule 2: If corner_accuracy < 0.75 and rarity_retention_rate < 0.80,
                       do not tighten rarity filtering.
        """
        guard = SafetyGuard()
        # Create a proposal that tightens theta_rare (makes it more negative)
        proposed = Policy(
            round_id=101,
            theta_tol=0.05,
            theta_rare=-0.04,  # Tightened (more negative than current -0.03)
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.70,  # Low (< 0.75)
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.75,  # Low (< 0.80)
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        result = await guard.check(proposal, telemetry)

        # Should be blocked
        assert result.blocked is True
        assert any("rarity" in r.lower() for r in result.blocked_reasons)

    @pytest.mark.asyncio
    async def test_safety_rule_3_novelty_like_drift_blocks_theta_drift_tightening(self, current_policy):
        """
        Safety Rule 3: Under novelty-like drift, do not tighten theta_drift before extra review.
        """
        guard = SafetyGuard()
        # Create a proposal that tightens theta_drift
        proposed = Policy(
            round_id=101,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.04,  # Tightened (lower than current 0.05)
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.68,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.90,
            golden_drift_score=0.10,  # High (> 0.08)
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.10
        )

        result = await guard.check(proposal, telemetry)

        # Should be blocked
        assert result.blocked is True
        assert any("drift" in r.lower() for r in result.blocked_reasons)

    @pytest.mark.asyncio
    async def test_safety_rule_4_hash_mismatch_blocks_recheck_reduction(self, current_policy):
        """
        Safety Rule 4: If hash_mismatch_rate > 0.02, do not lower recheck_probability.
        """
        guard = SafetyGuard()
        # Create a proposal that lowers recheck_probability (it's already 0)
        proposed = Policy(
            round_id=101,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,  # Same as current (0.0)
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.08,
            rarity_rate=0.03,
            honest_rate=0.75,
            noise_rate=0.14,
            main_accuracy=0.92,
            corner_accuracy=0.68,
            main_loss_delta_avg=-0.05,
            corner_loss_delta_avg=-0.02,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.90,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.05,  # High (> 0.02)
            recent_attack_pressure=0.10
        )

        result = await guard.check(proposal, telemetry)

        # In this case, recheck_probability is the same (0.0), so it shouldn't block
        # If it were actually lowering (e.g., from 0.1 to 0.05), it would block
        # Let's test with a current policy that has higher recheck
        current_with_recheck = Policy(
            round_id=100,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.10,  # Higher
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.0",
            effective_from_round=100
        )
        proposal.current_policy = current_with_recheck

        result = await guard.check(proposal, telemetry)

        # Should be blocked
        assert result.blocked is True
        assert any("recheck" in r.lower() for r in result.blocked_reasons)

    @pytest.mark.asyncio
    async def test_safety_rule_5_high_fraud_blocks_simultaneous_relaxation(self, current_policy):
        """
        Safety Rule 5: If fraud_rate > 0.30, do not simultaneously lower both
                       theta_tol strictness and recheck_probability.
        """
        guard = SafetyGuard()
        # Create a proposal that relaxes both theta_tol AND lowers recheck
        proposed = Policy(
            round_id=101,
            theta_tol=0.06,  # Relaxed (higher than current 0.05)
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,  # Lowered from 0.10
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        # Current with higher recheck
        current_with_recheck = Policy(
            round_id=100,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.10,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.0",
            effective_from_round=100
        )
        proposal = PolicyProposal(
            current_policy=current_with_recheck,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.35,  # Very high (> 0.30)
            rarity_rate=0.03,
            honest_rate=0.50,
            noise_rate=0.12,
            main_accuracy=0.85,
            corner_accuracy=0.65,
            main_loss_delta_avg=-0.02,
            corner_loss_delta_avg=-0.01,
            false_slash_estimate=0.01,
            rarity_retention_rate=0.85,
            golden_drift_score=0.03,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=200.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.20
        )

        result = await guard.check(proposal, telemetry)

        # Should be blocked
        assert result.blocked is True
        assert any("fraud" in r.lower() or "relax" in r.lower() for r in result.blocked_reasons)


# ============================================================================
# Round-Scoped Telemetry Helpers
# ============================================================================

class TestRoundScopedTelemetry:
    """Test helper functions used by upgraded policy telemetry."""

    def test_compute_round_rates_uses_round_totals(self):
        rates = compute_round_rates(
            {
                "fraud_count": "2",
                "rare_count": "1",
                "honest_count": "5",
                "noise_count": "2",
            }
        )

        assert rates["audit_sample_size"] == 10
        assert rates["fraud_rate"] == pytest.approx(0.2)
        assert rates["rarity_rate"] == pytest.approx(0.1)
        assert rates["honest_rate"] == pytest.approx(0.5)
        assert rates["noise_rate"] == pytest.approx(0.2)

    def test_recent_attack_pressure_weights_recent_rounds(self):
        pressure = compute_recent_attack_pressure([0.05, 0.10, 0.30], alpha=0.6)

        assert pressure == pytest.approx(0.212)


# ============================================================================
# Policy Confidence Helpers
# ============================================================================

class TestPolicyConfidence:
    """Test low-sample confidence helpers used by policy decisions."""

    def test_compute_policy_adjustment_scale_by_sample_band(self):
        assert compute_policy_adjustment_scale(0) == pytest.approx(1.0)
        assert compute_policy_adjustment_scale(5) == pytest.approx(0.2)
        assert compute_policy_adjustment_scale(20) == pytest.approx(0.4)
        assert compute_policy_adjustment_scale(40) == pytest.approx(0.6)
        assert compute_policy_adjustment_scale(75) == pytest.approx(0.8)
        assert compute_policy_adjustment_scale(120) == pytest.approx(1.0)

    def test_policy_decision_context_marks_low_sample_rounds(self, sample_telemetry):
        telemetry = sample_telemetry.model_copy(update={"audit_sample_size": 20})

        context = build_policy_decision_context(telemetry)

        assert context.low_sample_confidence is True
        assert context.sample_size_adjustment_scale == pytest.approx(0.4)
        assert context.sample_size_band == "small"
        assert "low_sample_caution" in context.active_modes


# ============================================================================
# Validator Tests
# ============================================================================

class TestPolicyValidator:
    """Test the policy validator."""

    @pytest.mark.asyncio
    async def test_validator_clamps_to_bounds(self, current_policy):
        """Test that validator clamps values to hard bounds."""
        validator = PolicyValidator()

        # Create a proposal with out-of-bounds values
        proposed = Policy.model_construct(
            round_id=101,
            theta_tol=0.15,  # Above max (0.10)
            theta_rare=-0.15,  # Below min (-0.10)
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=3.0,  # Above max (2.00)
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=False,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        result = await validator.validate(proposal)

        # Bounds are applied first, then step limits relative to current policy
        assert result.proposed_policy.theta_tol == pytest.approx(0.055)
        assert result.proposed_policy.theta_rare == pytest.approx(-0.035)
        assert result.proposed_policy.slash_multiplier == pytest.approx(1.2)
        # Should have validator messages
        assert len(result.validator_messages) > 0

    @pytest.mark.asyncio
    async def test_validator_enforces_max_step(self, current_policy):
        """Test that validator enforces per-round max step limits."""
        validator = PolicyValidator()

        # Create a proposal with changes exceeding max step
        proposed = Policy(
            round_id=101,
            theta_tol=0.06,  # Change of 0.01 exceeds max step 0.005
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=False,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        result = await validator.validate(proposal)

        # Should clamp to max step
        assert result.proposed_policy.theta_tol == 0.055  # 0.05 + 0.005

    @pytest.mark.asyncio
    async def test_validator_dampens_max_step_for_small_samples(self, current_policy, sample_telemetry):
        """Test that low sample sizes reduce how far policy may move in one round."""
        validator = PolicyValidator()
        telemetry = sample_telemetry.model_copy(update={"audit_sample_size": 20})

        proposed = Policy(
            round_id=101,
            theta_tol=0.055,  # Base max step, but should be reduced to 0.002 at scale 0.4
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing low-sample damping"],
            validator_passed=False,
            safety_guard_passed=False,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        result = await validator.validate(proposal, telemetry=telemetry)

        assert result.proposed_policy.theta_tol == pytest.approx(0.052)
        assert any(
            "Reduced policy-step aggressiveness due to limited audit sample size" in message
            for message in result.validator_messages
        )


# ============================================================================
# Policy Schema Tests
# ============================================================================

class TestPolicySchema:
    """Test the Policy schema."""

    def test_policy_bounds(self):
        """Test that Policy enforces bounds."""
        with pytest.raises(ValueError):
            Policy(
                round_id=100,
                theta_tol=0.15,  # Above max
                theta_rare=-0.03,
                theta_drift=0.05,
                cosine_filter_threshold=0.70,
                recheck_probability=0.0,
                slash_multiplier=1.0,
                rarity_reward_multiplier=1.0,
                corner_weight=1.0,
                policy_version="1.0.0",
                effective_from_round=100
            )

    def test_policy_hash(self):
        """Test policy hash computation."""
        policy = Policy(
            round_id=100,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.0",
            effective_from_round=100
        )

        hash_value = policy.compute_hash()

        # Should be a 64-character hex string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_default_policy(self):
        """Test DEFAULT_POLICY is valid."""
        assert DEFAULT_POLICY.round_id == 0
        assert DEFAULT_POLICY.theta_tol == 0.05
        assert DEFAULT_POLICY.theta_rare == -0.03


# ============================================================================
# PolicyProposal Schema Tests
# ============================================================================

class TestPolicyProposalSchema:
    """Test the PolicyProposal schema."""

    def test_proposal_blocked_property(self, current_policy):
        """Test the blocked property."""
        proposed = Policy(
            round_id=101,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )

        # Test with safety_guard_passed = True
        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=True,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )
        assert proposal.blocked is False

        # Test with safety_guard_passed = False
        proposal.safety_guard_passed = False
        assert proposal.blocked is True

    def test_proposal_get_diff(self, current_policy):
        """Test get_diff method."""
        proposed = Policy(
            round_id=101,
            theta_tol=0.06,  # Changed
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.1,  # Changed
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )

        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=True,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        diff = proposal.get_diff()

        # Should have entries for changed fields
        assert "theta_tol" in diff
        assert "slash_multiplier" in diff
        assert diff["theta_tol"] == (0.05, 0.06)
        assert diff["slash_multiplier"] == (1.0, 1.1)

    def test_proposal_summary(self, current_policy):
        """Test summary method."""
        proposed = Policy(
            round_id=101,
            theta_tol=0.05,
            theta_rare=-0.03,
            theta_drift=0.05,
            cosine_filter_threshold=0.70,
            recheck_probability=0.0,
            slash_multiplier=1.0,
            rarity_reward_multiplier=1.0,
            corner_weight=1.0,
            policy_version="1.0.1",
            effective_from_round=101
        )

        proposal = PolicyProposal(
            current_policy=current_policy,
            proposed_policy=proposed,
            reasons=["Testing"],
            validator_passed=True,
            safety_guard_passed=True,
            blocked_reasons=[],
            validator_messages=[],
            round_id=101
        )

        # No changes = default message
        summary = proposal.summary()
        assert "No policy changes" in summary


class TestGLMContextAlignment:
    """Test directional corrections applied to raw GLM outputs."""

    def test_corrects_relaxed_theta_tol_under_fraud_pressure(self, current_policy):
        engine = GLMPolicyEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.19,
            rarity_rate=0.03,
            honest_rate=0.60,
            noise_rate=0.18,
            main_accuracy=0.84,
            corner_accuracy=0.67,
            main_loss_delta_avg=0.006,
            corner_loss_delta_avg=-0.003,
            false_slash_estimate=0.06,
            rarity_retention_rate=0.79,
            golden_drift_score=0.08,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.27
        )
        context = build_policy_decision_context(telemetry)

        validated, messages = engine._validate_params(
            current_policy=current_policy,
            glm_params={
                "theta_tol": 0.055,
                "theta_rare": -0.025,
                "theta_drift": 0.05,
                "recheck_probability": 0.10,
                "slash_multiplier": 0.80,
                "rarity_reward_multiplier": 1.20,
                "corner_weight": 1.0,
                "reasons": ["test"],
            },
            decision_context=context,
        )

        assert validated["theta_tol"] == pytest.approx(0.048)
        assert any("theta_tol" in message for message in messages)

    def test_corrects_relaxed_theta_drift_under_harmful_shift(self, current_policy):
        engine = GLMPolicyEngine()
        telemetry = RoundTelemetry(
            round_id=100,
            fraud_rate=0.16,
            rarity_rate=0.04,
            honest_rate=0.62,
            noise_rate=0.18,
            main_accuracy=0.79,
            corner_accuracy=0.71,
            main_loss_delta_avg=0.021,
            corner_loss_delta_avg=-0.0002,
            false_slash_estimate=0.03,
            rarity_retention_rate=0.82,
            golden_drift_score=0.10,
            reject_rate_l3=0.02,
            cosine_outlier_ratio=0.15,
            suspect_queue_length=25,
            avg_sbt_score=250.0,
            new_vehicle_ratio=0.05,
            hash_mismatch_rate=0.0,
            recent_attack_pressure=0.24
        )
        context = build_policy_decision_context(telemetry)

        validated, messages = engine._validate_params(
            current_policy=current_policy,
            glm_params={
                "theta_tol": 0.05,
                "theta_rare": -0.03,
                "theta_drift": 0.055,
                "recheck_probability": 0.05,
                "slash_multiplier": 1.0,
                "rarity_reward_multiplier": 1.0,
                "corner_weight": 1.0,
                "reasons": ["test"],
            },
            decision_context=context,
        )

        assert validated["theta_drift"] == pytest.approx(0.048)
        assert any("theta_drift" in message for message in messages)


def test_proposal_recency_key_prefers_creation_time_over_round_id():
    current_policy = Policy(
        round_id=55,
        theta_tol=0.032,
        theta_rare=-0.005,
        theta_drift=0.042,
        cosine_filter_threshold=0.70,
        recheck_probability=0.5,
        slash_multiplier=0.55,
        rarity_reward_multiplier=2.0,
        corner_weight=1.0,
        policy_version="1.0.1",
        effective_from_round=55,
        created_at=datetime(2026, 3, 18, 9, 56, 36, tzinfo=timezone.utc),
    )
    stale_high_round = PolicyProposal(
        current_policy=current_policy,
        proposed_policy=current_policy.model_copy(update={"round_id": 2001, "effective_from_round": 2001}),
        round_id=2001,
        created_at=datetime(2026, 3, 18, 10, 9, 13, tzinfo=timezone.utc),
        reasons=["stale high-round test proposal"],
        validator_passed=True,
        safety_guard_passed=True,
    )
    live_recent = PolicyProposal(
        current_policy=current_policy.model_copy(update={"round_id": 99, "effective_from_round": 99}),
        proposed_policy=current_policy.model_copy(update={"round_id": 101, "effective_from_round": 101}),
        round_id=101,
        created_at=datetime(2026, 3, 23, 6, 56, 49, tzinfo=timezone.utc),
        reasons=["live proposal should win"],
        validator_passed=True,
        safety_guard_passed=True,
    )

    proposals = [stale_high_round, live_recent]
    proposals.sort(key=proposal_recency_key, reverse=True)

    assert proposals[0].round_id == 101


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
