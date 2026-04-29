"""
Policy-specific API routes for FLPG Policy Agent.

This module defines routes for policy management, proposal,
and approval operations.

Updated to match FLPG_Adaptive_Policy_Agent_Claude_Code_Guide.md

Canonical endpoints under this router:
- GET /api/v1/policy/health
- GET /api/v1/policy/current
- GET /api/v1/policy/next
- POST /api/v1/policy/propose
- POST /api/v1/policy/activate
- GET /api/v1/policy/history/{round_id}
- GET /api/v1/policy/explanation/{round_id}
"""

import asyncio
from time import monotonic
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional
import logging

from common.schemas import Policy, PolicyProposal, RoundTelemetry
from common.config import REDIS_URL
from policy_agent.api.exceptions import (
    PolicyNotFoundException,
    TelemetryNotFoundException,
    http_exception_from_error
)

logger = logging.getLogger(__name__)

BASELINE_ANALYSIS_TTL_SECONDS = 180
_baseline_analysis_cache: dict[tuple[int, str], tuple[float, dict]] = {}
_baseline_analysis_inflight: dict[tuple[int, str], asyncio.Task] = {}
_baseline_analysis_lock: asyncio.Lock | None = None

router = APIRouter(
    prefix="/api/v1/policy",
    tags=["policy"]
)


# ============================================================================
# Dependencies
# ============================================================================

async def get_store():
    """Get Redis store instance."""
    from policy_agent.storage.redis_store import RedisPolicyStore
    return RedisPolicyStore(redis_url=REDIS_URL)


def _get_baseline_analysis_lock() -> asyncio.Lock:
    global _baseline_analysis_lock
    if _baseline_analysis_lock is None:
        _baseline_analysis_lock = asyncio.Lock()
    return _baseline_analysis_lock


def _baseline_cache_key(current_policy: Optional[Policy], rounds: int) -> tuple[int, str]:
    if current_policy is None:
        return rounds, "default-policy"
    return rounds, current_policy.model_dump_json()


def _run_baseline_analysis_sync(current_policy: Optional[Policy], rounds: int) -> dict:
    from policy_agent.analysis.baselines import build_baseline_analysis

    return asyncio.run(build_baseline_analysis(current_policy, rounds=rounds))


async def _get_cached_baseline_analysis(current_policy: Optional[Policy], rounds: int) -> dict:
    cache_key = _baseline_cache_key(current_policy, rounds)
    cached = _baseline_analysis_cache.get(cache_key)
    now = monotonic()

    if cached and now - cached[0] < BASELINE_ANALYSIS_TTL_SECONDS:
        logger.info("Serving cached baseline analysis for rounds=%s", rounds)
        return cached[1]

    lock = _get_baseline_analysis_lock()

    async with lock:
        cached = _baseline_analysis_cache.get(cache_key)
        now = monotonic()
        if cached and now - cached[0] < BASELINE_ANALYSIS_TTL_SECONDS:
            logger.info("Serving cached baseline analysis for rounds=%s", rounds)
            return cached[1]

        task = _baseline_analysis_inflight.get(cache_key)
        if task is None:
            logger.info("Starting baseline analysis computation for rounds=%s", rounds)
            policy_copy = current_policy.model_copy(deep=True) if current_policy else None
            task = asyncio.create_task(
                asyncio.to_thread(_run_baseline_analysis_sync, policy_copy, rounds)
            )
            _baseline_analysis_inflight[cache_key] = task

    try:
        result = await task
    finally:
        async with lock:
            if _baseline_analysis_inflight.get(cache_key) is task:
                _baseline_analysis_inflight.pop(cache_key, None)

    async with lock:
        _baseline_analysis_cache[cache_key] = (monotonic(), result)
        expired_keys = [
            key
            for key, (timestamp, _payload) in _baseline_analysis_cache.items()
            if monotonic() - timestamp >= BASELINE_ANALYSIS_TTL_SECONDS
        ]
        for key in expired_keys:
            if key != cache_key:
                _baseline_analysis_cache.pop(key, None)

    return result


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check(store=Depends(get_store)):
    """
    Health check endpoint.

    Returns health status of the policy agent and Redis connection.
    """
    is_healthy = await store.health_check()

    return {
        "status": "ok" if is_healthy else "degraded",
        "redis": "connected" if is_healthy else "disconnected"
    }


# ============================================================================
# Policy Query Routes
# ============================================================================

@router.get("/current", response_model=Policy)
async def get_current_policy(store=Depends(get_store)):
    """
    Get the current frozen policy.

    Returns the policy that should be used by L1/L2/L3/L4 services
    for the current round (from policy:current).
    """
    policy = await store.get_current_policy()

    if not policy:
        raise PolicyNotFoundException()

    return policy


@router.get("/next", response_model=Policy)
async def get_next_policy(store=Depends(get_store)):
    """
    Get the proposed or approved next policy.

    Returns the policy from policy:next that will be activated
    at the start of the next round.
    """
    policy = await store.get_next_policy()

    if not policy:
        raise PolicyNotFoundException()

    return policy


@router.get("/proposal/latest", response_model=PolicyProposal)
async def get_latest_proposal(store=Depends(get_store)):
    """Get the most recent stored policy proposal."""
    proposal = await store.get_latest_proposal()

    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No policy proposal found"
        )

    return proposal


@router.get("/proposal/{round_id}", response_model=PolicyProposal)
async def get_policy_proposal(round_id: int, store=Depends(get_store)):
    """Get a stored policy proposal for a specific round."""
    proposal = await store.get_proposal(round_id)

    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No policy proposal found for round {round_id}"
        )

    return proposal


@router.get("/history/{round_id}", response_model=Policy)
async def get_policy_by_round(round_id: int, store=Depends(get_store)):
    """
    Get historical policy snapshot.

    Returns the policy that was active during the specified round
    (from policy:history:r{round_id}).
    """
    policy = await store.get_policy(round_id)

    if not policy:
        raise PolicyNotFoundException(round_id)

    return policy


@router.get("/history", response_model=List[Policy])
async def get_policy_history(
    limit: int = 10,
    store=Depends(get_store)
):
    """
    Get historical policies.

    Args:
        limit: Maximum number of policies to return
        offset: Number of policies to skip
    """
    return await store.get_recent_policies(count=limit)


@router.get("/explanation/{round_id}")
async def get_policy_explanation(round_id: int, store=Depends(get_store)):
    """
    Get human-readable explanation for a policy.

    Returns the explanation text and metadata for the specified round
    (from policy:explanation:r{round_id}).
    """
    explanation = await store.get_explanation(round_id)

    if not explanation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No explanation found for round {round_id}"
        )

    return explanation


# ============================================================================
# Policy Proposal Routes
# ============================================================================

@router.post("/propose", response_model=PolicyProposal)
async def propose_next_policy(
    telemetry: RoundTelemetry,
    store=Depends(get_store)
):
    """
    Propose policy for the next round based on telemetry.

    This runs the full pipeline:
    1. Load current policy
    2. GLM engine proposes next values (direct parameter control)
    3. Validator clamps to bounds and step limits
    4. Safety guard checks context rules
    5. Returns PolicyProposal with results

    The proposal must be explicitly approved before being activated.

    UPDATED: GLM is now the primary decision maker (not advisory).
    """
    from policy_agent.engine.glm_policy_engine import GLMPolicyEngine
    from policy_agent.constraints.validator import PolicyValidator
    from policy_agent.constraints.safety_guard import SafetyGuard
    from common.schemas import DEFAULT_POLICY

    # Load current policy
    current_policy = await store.get_current_policy()
    if not current_policy:
        current_policy = DEFAULT_POLICY

    await store.save_telemetry(telemetry)

    # Initialize components
    glm_engine = GLMPolicyEngine()
    validator = PolicyValidator()
    safety_guard = SafetyGuard()

    # Run GLM engine for direct parameter control
    proposal = await glm_engine.propose(current_policy, telemetry)

    # Run validation and safety check
    proposal = await validator.validate(proposal, telemetry=telemetry)
    proposal = await safety_guard.check(proposal, telemetry)

    # Store the proposal
    await store.save_proposal(proposal)

    await store.save_explanation(
        round_id=proposal.round_id,
        explanation=proposal.summary(),
        metadata={
            "source_engine": proposal.source_engine,
            "blocked": proposal.blocked,
            "reasons": proposal.reasons,
            "validator_messages": proposal.validator_messages,
            "llm_used": proposal.llm_used,
            "params_before": {
                key: value
                for key, value in proposal.current_policy.model_dump().items()
                if isinstance(value, (int, float))
            },
            "params_after": {
                key: value
                for key, value in proposal.proposed_policy.model_dump().items()
                if isinstance(value, (int, float))
            },
        }
    )

    # If not blocked, store to policy:next
    if not proposal.blocked:
        await store.set_next_policy(proposal.proposed_policy)

    return proposal


# ============================================================================
# Policy Activation Routes
# ============================================================================

@router.post("/activate")
async def activate_next_policy(
    round_id: int,
    store=Depends(get_store)
):
    """
    Activate policy:next into policy:current at round start.

    This implements the Round Start lifecycle:
    1. Load policy:next if available
    2. Archive current policy to policy:history:r{round_id}
    3. Move policy:next to policy:current
    4. Clear policy:next

    Args:
        round_id: Round number that is starting

    Returns:
        Activation result with policy hash
    """
    activated_policy = await store.activate_next_policy(round_id)

    if not activated_policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No next policy found to activate"
        )

    return {
        "status": "activated",
        "round_id": round_id,
        "policy_hash": activated_policy.compute_hash(),
        "policy_version": activated_policy.policy_version,
    }


# ============================================================================
# Telemetry Routes
# ============================================================================

@router.get("/telemetry/latest")
async def get_latest_telemetry(store=Depends(get_store)):
    """Get the most recent telemetry data."""
    from common.schemas import RoundTelemetry

    telemetry = await store.get_latest_telemetry()

    if not telemetry:
        raise TelemetryNotFoundException()

    return telemetry


@router.post("/telemetry")
async def save_latest_telemetry(
    telemetry: RoundTelemetry,
    store=Depends(get_store)
):
    """Persist telemetry without generating a policy proposal."""
    await store.save_telemetry(telemetry)
    return {
        "status": "saved",
        "round_id": telemetry.round_id,
        "created_at": telemetry.created_at,
    }


@router.get("/telemetry/{round_id}")
async def get_telemetry_by_round(round_id: int, store=Depends(get_store)):
    """Get telemetry for a specific round."""
    telemetry = await store.get_telemetry(round_id)

    if not telemetry:
        raise TelemetryNotFoundException(round_id)

    return telemetry


@router.get("/telemetry", response_model=List[RoundTelemetry])
async def get_telemetry_history(
    limit: int = 10,
    store=Depends(get_store)
):
    """Get recent telemetry entries."""
    return await store.get_recent_telemetry(count=limit)


# ============================================================================
# Comparison Routes
# ============================================================================

@router.get("/diff/{round_a}/{round_b}")
async def compare_policies(
    round_a: int,
    round_b: int,
    store=Depends(get_store)
):
    """
    Compare policies between two rounds.

    Returns the differences in parameter values.
    """
    policy_a = await store.get_policy(round_a)
    policy_b = await store.get_policy(round_b)

    if not policy_a:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No policy found for round {round_a}"
        )

    if not policy_b:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No policy found for round {round_b}"
        )

    # Calculate diff
    diff = {}
    for field in policy_a.model_dump():
        val_a = getattr(policy_a, field, None)
        val_b = getattr(policy_b, field, None)

        if val_a != val_b:
            diff[field] = {
                "round_a": val_a,
                "round_b": val_b
            }

    return {
        "round_a": round_a,
        "round_b": round_b,
        "diff": diff,
        "has_changes": len(diff) > 0
    }


# ============================================================================
# GLM Decision Log Routes
# ============================================================================

@router.get("/glm-decisions")
async def get_glm_decision_history(
    limit: int = 20,
    offset: int = 0,
    store=Depends(get_store)
):
    """
    Get GLM decision history for parameter modifications.

    Returns a log of all GLM-based policy changes with:
    - Round ID
    - Parameters before/after
    - GLM's reasons for changes
    - Whether GLM was actually used (vs fallback)
    - Safety guard results
    """
    proposals = await store.get_recent_proposals(count=limit + offset)

    decisions = []
    for proposal in proposals[offset:]:
        telemetry = await store.get_telemetry(max(0, proposal.round_id - 1))
        diff = proposal.get_diff()
        parameter_changes = []

        for param, (before, after) in diff.items():
            if param in {"round_id", "effective_from_round", "policy_version", "created_at"}:
                continue
            if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
                continue
            parameter_changes.append({
                "param": param,
                "before": before,
                "after": after,
            })

        # Build decision entry
        decision = {
            "round_id": proposal.round_id,
            "timestamp": proposal.created_at.isoformat(),
            "llm_used": bool(proposal.llm_used),
            "source_engine": proposal.source_engine,
            "blocked": proposal.blocked,
            "reasons": proposal.reasons,
            "validator_messages": proposal.validator_messages,
            "parameters_changed": parameter_changes,
            "telemetry": None,
        }

        if telemetry:
            decision["telemetry"] = {
                "fraud_rate": telemetry.fraud_rate,
                "rarity_rate": telemetry.rarity_rate,
                "honest_rate": telemetry.honest_rate,
                "main_accuracy": telemetry.main_accuracy,
                "corner_accuracy": telemetry.corner_accuracy,
            }

        decisions.append(decision)

    return {
        "total": len(decisions),
        "offset": offset,
        "limit": limit,
        "data": decisions
    }


@router.get("/analysis/baselines")
async def get_baseline_analysis(
    rounds: int = 12,
    store=Depends(get_store),
):
    """
    Run backend-side baseline evaluation for the Data Analysis page.

    The returned payload is computed on demand by the backend evaluation
    engine so the frontend does not embed any presentation-only datasets.
    """
    rounds = max(4, min(rounds, 24))
    current_policy = await store.get_current_policy()
    return await _get_cached_baseline_analysis(current_policy, rounds)
