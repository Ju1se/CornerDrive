"""
Celery task for round-close policy proposal.

This task is triggered at the end of each training round to:
1. Collect telemetry from L1/L2/L3/L4
2. Build RoundTelemetry
3. Call GLM policy engine for direct parameter control
4. Validate and run safety guard
5. Store the proposal to policy:next

UPDATED: GLM now directly controls policy parameters (not just advisory).
Removed the rule engine - GLM is the primary decision maker.
"""

import logging
import os
from typing import Optional

from celery import Celery
from dotenv import load_dotenv
from redis.asyncio import Redis

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.schemas import RoundTelemetry, PolicyProposal, DEFAULT_POLICY
from common.config import CELERY_BROKER_URL, REDIS_URL, L2_AUDIT_QUEUE
from common.utils.round_stats import compute_recent_attack_pressure, compute_round_rates, round_stats_key

load_dotenv()

logger = logging.getLogger(__name__)

ATTACK_PRESSURE_WINDOW = 5

# Initialize Celery app
celery_app = Celery(
    'policy_agent',
    broker=CELERY_BROKER_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


@celery_app.task(name='policy.round_close')
def round_close_task(
    round_id: int,
    telemetry_data: dict,
    auto_apply: bool = False
) -> dict:
    """
    Execute round-close policy proposal pipeline.

    Pipeline (from guide):
    1. Collect telemetry from L1/L2/L3/L4
    2. Persist telemetry to Redis (telemetry:round:r{round_id}, telemetry:latest)
    3. Trigger policy proposal generation
    4. Generate proposal via rule engine
    5. Validate proposal (bounds + step limits)
    6. Run safety guard (5 context rules)
    7. Persist accepted result to policy:next and policy:history:r{round_id+1}
    8. Optionally generate explanation text

    Args:
        round_id: Round number that just closed
        telemetry_data: Telemetry data from L1/L2/L3/L4
        auto_apply: If True, automatically mark as approved

    Returns:
        Dict with pipeline results
    """
    import asyncio

    async def _run_round_close():
        from policy_agent.storage.redis_store import RedisPolicyStore
        from policy_agent.engine.glm_policy_engine import GLMPolicyEngine
        from policy_agent.constraints.validator import PolicyValidator
        from policy_agent.constraints.safety_guard import SafetyGuard

        # Initialize components
        store = RedisPolicyStore(redis_url=REDIS_URL)
        glm_engine = GLMPolicyEngine()
        validator = PolicyValidator()
        safety_guard = SafetyGuard()

        try:
            # Step 1: Load and validate telemetry
            telemetry = RoundTelemetry(**telemetry_data)

            logger.info(f"Processing round close for round {round_id}")

            # Step 2: Persist telemetry to Redis
            await store.save_telemetry(telemetry)

            # Step 3: Load current policy
            current_policy = await store.get_current_policy()
            if not current_policy:
                current_policy = DEFAULT_POLICY
                logger.info("No current policy found, using DEFAULT_POLICY")

            # Step 4: Generate proposal via GLM engine (direct parameter control)
            proposal = await glm_engine.propose(current_policy, telemetry)

            # Step 5: Validate proposal (bounds + step limits)
            proposal = await validator.validate(proposal, telemetry=telemetry)

            # Step 6: Run safety guard
            proposal = await safety_guard.check(proposal, telemetry)

            # Step 7: Store proposal
            await store.save_proposal(proposal)

            # Step 8: Store to policy:next if not blocked
            if not proposal.blocked:
                await store.set_next_policy(proposal.proposed_policy)

                # Auto-approve if enabled
                if auto_apply:
                    proposal.approved = True

                logger.info(
                    f"Policy proposal for round {round_id + 1} stored to policy:next"
                )
            else:
                logger.warning(
                    f"Policy proposal blocked: {proposal.blocked_reasons}"
                )

            # Step 9: Optionally generate explanation (from GLM reasons)
            await store.save_explanation(
                round_id=round_id + 1,
                explanation=proposal.summary(),
                metadata={
                    "source_engine": proposal.source_engine,
                    "changes": len(proposal.get_diff()),
                    "reasons": proposal.reasons,
                    "blocked": proposal.blocked,
                    "validator_messages": proposal.validator_messages,
                    "llm_used": proposal.llm_used
                }
            )

            return {
                "status": "success",
                "round_id": round_id,
                "next_round": round_id + 1,
                "blocked": proposal.blocked,
                "approved": proposal.approved,
                "changes": len(proposal.get_diff()),
                "blocked_reasons": proposal.blocked_reasons,
                "validator_messages": proposal.validator_messages,
                "policy_hash": proposal.proposed_policy.compute_hash(),
            }

        finally:
            await store.close()

    # Run async code in sync context
    return asyncio.run(_run_round_close())


@celery_app.task(name='policy.collect_telemetry')
def collect_telemetry_task(round_id: int) -> dict:
    """
    Collect telemetry from all layers (L1/L2/L3/L4).

    This task queries each layer's metrics endpoint and aggregates
    them into a RoundTelemetry object.

    Args:
        round_id: Round number to collect telemetry for

    Returns:
        Aggregated telemetry data
    """
    import asyncio

    async def _collect():
        import httpx
        from policy_agent.storage.redis_store import RedisPolicyStore

        # Layer endpoints
        endpoints = {
            "l1": os.getenv("L1_URL", "http://localhost:8081"),
            "l4": os.getenv("L4_URL", "http://localhost:8082"),
        }

        telemetry = {}
        redis_client: Optional[Redis] = None
        store: Optional[RedisPolicyStore] = None

        try:
            redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
            raw_round_stats = await redis_client.hgetall(round_stats_key(round_id))
            if raw_round_stats:
                telemetry.update(compute_round_rates(raw_round_stats))
            telemetry["suspect_queue_length"] = int(
                await redis_client.llen(L2_AUDIT_QUEUE) or 0
            )
        except Exception as e:
            logger.warning(f"Failed to collect round-scoped telemetry from Redis: {e}")
        finally:
            if redis_client is not None:
                await redis_client.close()

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Collect from L1
            try:
                resp = await client.get(f"{endpoints['l1']}/health")
                if resp.status_code == 200:
                    pass
            except Exception as e:
                logger.warning(f"Failed to collect L1 telemetry: {e}")

            # Collect from L4
            try:
                resp = await client.get(f"{endpoints['l4']}/api/v1/stats")
                if resp.status_code == 200:
                    l4_data = resp.json()
                    if not {
                        "fraud_rate",
                        "rarity_rate",
                        "honest_rate",
                        "noise_rate",
                    }.issubset(telemetry):
                        total = max(l4_data.get("total_audits", 0), 1)
                        telemetry.update({
                            "fraud_rate": l4_data.get("fraud_count", 0) / total,
                            "rarity_rate": l4_data.get("rare_count", 0) / total,
                            "honest_rate": l4_data.get("honest_count", 0) / total,
                            "noise_rate": l4_data.get("noise_count", 0) / total,
                        })
                    telemetry["avg_sbt_score"] = float(l4_data.get("total_rewards_distributed", 0.0))
            except Exception as e:
                logger.warning(f"Failed to collect L4 telemetry: {e}")

        try:
            store = RedisPolicyStore(redis_url=REDIS_URL)
            recent_telemetry = await store.get_recent_telemetry(count=ATTACK_PRESSURE_WINDOW - 1)
            historical_rates = [
                item.fraud_rate
                for item in sorted(recent_telemetry, key=lambda entry: entry.round_id)
                if item.round_id < round_id
            ]
            telemetry["recent_attack_pressure"] = compute_recent_attack_pressure(
                historical_rates + [telemetry.get("fraud_rate", 0.0)]
            )
        except Exception as e:
            logger.warning(f"Failed to compute recent attack pressure: {e}")
        finally:
            if store is not None:
                await store.close()

        # Ensure required fields with defaults
        defaults = {
            "round_id": round_id,
            "fraud_rate": 0.0,
            "rarity_rate": 0.0,
            "honest_rate": 0.0,
            "noise_rate": 0.0,
            "main_accuracy": 0.0,
            "corner_accuracy": 0.0,
            "main_loss_delta_avg": 0.0,
            "corner_loss_delta_avg": 0.0,
            "false_slash_estimate": 0.0,
            "rarity_retention_rate": 1.0,
            "golden_drift_score": 0.0,
            "reject_rate_l3": 0.0,
            "cosine_outlier_ratio": 0.0,
            "suspect_queue_length": 0,
            "audit_sample_size": 0,
            "avg_sbt_score": 0.0,
            "new_vehicle_ratio": 0.0,
            "hash_mismatch_rate": 0.0,
            "recent_attack_pressure": 0.0,
        }

        for key, value in defaults.items():
            telemetry.setdefault(key, value)

        return telemetry

    return asyncio.run(_collect())


@celery_app.task(name='policy.activate_next_policy')
def activate_next_policy_task(round_id: int) -> dict:
    """
    Activate the next policy at round start.

    This implements the Round Start lifecycle from the guide:
    1. Load policy:next if available
    2. Move it to policy:current
    3. Archive old policy to history

    Args:
        round_id: Round number that is starting

    Returns:
        Dict with activation results
    """
    import asyncio

    async def _activate():
        from policy_agent.storage.redis_store import RedisPolicyStore

        store = RedisPolicyStore(redis_url=REDIS_URL)

        try:
            # Activate next policy
            activated_policy = await store.activate_next_policy(round_id)

            if activated_policy:
                return {
                    "status": "success",
                    "round_id": round_id,
                    "policy_activated": True,
                    "policy_hash": activated_policy.compute_hash(),
                    "policy_version": activated_policy.policy_version,
                }
            else:
                return {
                    "status": "no_policy",
                    "round_id": round_id,
                    "policy_activated": False,
                    "message": "No policy found in policy:next"
                }

        finally:
            await store.close()

    return asyncio.run(_activate())
