"""
L2: Dual-Purpose Audit - Celery Worker
Processes suspect gradients from L1 queue asynchronously.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from celery import Celery
import numpy as np
import redis

from common.config import (
    CELERY_BROKER_URL,
    REDIS_URL,
    L2_AUDIT_QUEUE,
    SBT_TIER_SILVER,
    SBT_TIER_GOLD,
    SBT_TIER_PLATINUM,
)
from common.demo_audit_assets import DEMO_GRADIENT_DIM, build_demo_audit_bundle
from common.policy_loader import load_current_policy
from common.utils.round_stats import CLASSIFICATION_FIELD_MAP, round_stats_key
from .classifier import DualChannelAuditor, Classification

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    'l2_audit',
    broker=CELERY_BROKER_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_default_queue=L2_AUDIT_QUEUE,
    task_routes={
        'l2_audit.*': {'queue': L2_AUDIT_QUEUE},
    },
)

# Redis client
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
except redis.ConnectionError:
    logger.error(f"Cannot connect to Redis at {REDIS_URL}")
    redis_client = None


# Global auditor instance (lazy initialization)
_auditor = None

STAT_KEY_BY_CLASSIFICATION = {
    Classification.FRAUD: "stats:fraud_count",
    Classification.RARITY: "stats:rare_count",
    Classification.HONEST: "stats:honest_count",
    Classification.NOISE: "stats:noise_count",
}


def _update_round_statistics(policy_round: int, classification: str, sbt_points: int) -> None:
    """Store classification counters under the active policy round."""
    if not redis_client or policy_round < 0:
        return

    round_key = round_stats_key(policy_round)
    redis_client.hincrby(round_key, "audit_count", 1)

    field_name = CLASSIFICATION_FIELD_MAP.get(classification)
    if field_name:
        redis_client.hincrby(round_key, field_name, 1)

    if sbt_points > 0:
        redis_client.hincrbyfloat(round_key, "total_rewards", float(sbt_points))
    elif sbt_points < 0:
        redis_client.hincrbyfloat(round_key, "total_slashed", abs(float(sbt_points)))

    redis_client.expire(round_key, 30 * 24 * 3600)


def get_auditor() -> DualChannelAuditor:
    """Get or create auditor instance."""
    global _auditor
    if _auditor is None:
        model, main_dataset, corner_dataset = build_demo_audit_bundle()

        _auditor = DualChannelAuditor(
            model=model,
            main_dataset=main_dataset,
            corner_dataset=corner_dataset,
        )
        logger.info("L2 Auditor initialized")

    return _auditor


def _tier_for_reputation(reputation: int) -> str:
    """Map reputation into the tier buckets used by the dashboard."""
    if reputation >= SBT_TIER_PLATINUM:
        return "platinum"
    if reputation >= SBT_TIER_GOLD:
        return "gold"
    if reputation >= SBT_TIER_SILVER:
        return "silver"
    return "bronze"


def _update_vehicle_state(result_dict: dict) -> None:
    """Persist per-vehicle counters so the dashboard can show live participants."""
    if not redis_client:
        return

    vehicle_id = result_dict["vehicle_id"].lower()
    vehicle_key = f"vehicle:{vehicle_id}"
    existing = redis_client.hgetall(vehicle_key)
    is_new_vehicle = not existing

    def _get_int(field: str, default: int = 0) -> int:
        raw = existing.get(field.encode()) if existing else None
        return int(raw or default)

    def _get_float(field: str, default: float = 0.0) -> float:
        raw = existing.get(field.encode()) if existing else None
        return float(raw or default)

    prior_reputation = _get_int("reputation")
    prior_tier = _tier_for_reputation(prior_reputation)

    contribution_count = _get_int("contributions") + 1
    fraud_count = _get_int("fraud_count")
    rare_count = _get_int("rare_count")
    honest_count = _get_int("honest_count")
    noise_count = _get_int("noise_count")

    classification = result_dict["classification"]
    if classification == Classification.FRAUD.value:
        fraud_count += 1
    elif classification == Classification.RARITY.value:
        rare_count += 1
    elif classification == Classification.HONEST.value:
        honest_count += 1
    else:
        noise_count += 1

    updated_reputation = max(0, prior_reputation + int(result_dict["sbt_points"]))
    updated_tier = _tier_for_reputation(updated_reputation)
    updated_rewards = _get_float("rewards") + max(0.0, float(result_dict["sbt_points"]))
    updated_slashed = _get_float("slashed") + max(0.0, float(-result_dict["sbt_points"]))

    redis_client.hset(
        vehicle_key,
        mapping={
            "reputation": updated_reputation,
            "contributions": contribution_count,
            "fraud_count": fraud_count,
            "rare_count": rare_count,
            "honest_count": honest_count,
            "noise_count": noise_count,
            "stake": _get_float("stake", 0.01),
            "rewards": updated_rewards,
            "slashed": updated_slashed,
            "registered": "true",
            "last_classification": classification,
            "updated_at": result_dict["timestamp"],
        },
    )

    if is_new_vehicle:
        redis_client.incr("stats:total_vehicles")
        redis_client.incr(f"tier:{updated_tier}")
    elif prior_tier != updated_tier:
        redis_client.decr(f"tier:{prior_tier}")
        redis_client.incr(f"tier:{updated_tier}")

    if result_dict["sbt_points"] > 0:
        redis_client.incrbyfloat("stats:total_rewards", float(result_dict["sbt_points"]))
    elif result_dict["sbt_points"] < 0:
        redis_client.incrbyfloat("stats:total_slashed", abs(float(result_dict["sbt_points"])))


@celery_app.task(name='l2_audit.audit_gradient')
def audit_gradient(suspect_data: dict) -> dict:
    """
    Celery task to audit a suspect gradient.

    Args:
        suspect_data: {
            "vehicle_id": str,
            "gradient": List[float],
            "cosine_score": float,
            "timestamp": str,
        }

    Returns:
        Audit result dictionary
    """
    vehicle_id = suspect_data["vehicle_id"]
    gradient_list = suspect_data["gradient"]
    policy_round = suspect_data.get("policy_round")
    routing_reason = suspect_data.get("routing_reason", "cosine_screening")

    if len(gradient_list) != DEMO_GRADIENT_DIM:
        raise ValueError(
            f"Gradient dimension mismatch for {vehicle_id}: "
            f"expected {DEMO_GRADIENT_DIM}, got {len(gradient_list)}"
        )

    gradient = np.array(gradient_list, dtype=float)

    logger.info(f"Starting audit for vehicle {vehicle_id}")

    try:
        auditor = get_auditor()
        try:
            current_policy = asyncio.run(
                load_current_policy(round_id=int(policy_round)) if policy_round is not None else load_current_policy()
            )
            auditor.apply_policy(current_policy)
            if policy_round is None:
                policy_round = current_policy.round_id
        except Exception as exc:
            logger.warning(f"Failed to load current policy for L2 audit: {exc}")
            if policy_round is None:
                policy_round = 0

        result = auditor.audit(vehicle_id, gradient)

        # Convert to dict for JSON serialization
        result_dict = {
            "policy_round": int(policy_round or 0),
            "vehicle_id": result.vehicle_id,
            "classification": result.classification.value,
            "delta_loss_main": result.delta_loss_main,
            "delta_loss_corner": result.delta_loss_corner,
            "final_score": result.final_score,
            "include_in_aggregation": result.include_in_aggregation,
            "sbt_points": result.sbt_points,
            "fraud_proof": result.fraud_proof,
            "rarity_certificate": result.rarity_certificate,
            "routing_reason": routing_reason,
            "timestamp": result.timestamp.isoformat(),
        }

        # Store result in Redis if available
        if redis_client:
            try:
                redis_client.hset(
                    f"audit:{vehicle_id}:{int(result.timestamp.timestamp())}",
                    mapping={
                        "result": json.dumps(result_dict),
                        "classification": result.classification.value,
                    }
                )

                # Update statistics
                redis_client.incr(STAT_KEY_BY_CLASSIFICATION[result.classification])
                _update_round_statistics(
                    policy_round=result_dict["policy_round"],
                    classification=result.classification.value,
                    sbt_points=result.sbt_points,
                )

                # Queue for L4 settlement
                redis_client.lpush("l4_settlement_queue", json.dumps(result_dict))

                # Store recent audit for dashboard
                recent_audit = {
                    "policy_round": result_dict["policy_round"],
                    "vehicle_id": result.vehicle_id,
                    "classification": result.classification.value,
                    "delta_loss_main": result.delta_loss_main,
                    "delta_loss_corner": result.delta_loss_corner,
                    "sbt_points": result.sbt_points,
                    "routing_reason": routing_reason,
                    "timestamp": result.timestamp.isoformat(),
                }
                redis_client.lpush("recent_audits", json.dumps(recent_audit))
                redis_client.ltrim("recent_audits", 0, 999)  # Keep last 1000
                _update_vehicle_state(result_dict)

            except Exception as e:
                logger.error(f"Redis error during audit result storage: {e}")

        return result_dict

    except Exception as e:
        logger.error(f"Audit failed for {vehicle_id}: {e}")
        raise


@celery_app.task(name='l2_audit.get_statistics')
def get_statistics() -> dict:
    """Get audit statistics from Redis."""
    if not redis_client:
        return {"error": "Redis not available"}

    try:
        stats = {}
        keys = ["fraud_count", "rare_count", "honest_count", "noise_count"]

        for key in keys:
            count = int(redis_client.get(f"stats:{key}") or 0)
            stats[key.replace("_count", "")] = count

        stats["rarity"] = stats.pop("rare")

        total = sum(stats.values())
        stats["total"] = total
        stats["fraud_rate"] = stats.get("fraud", 0) / max(total, 1)

        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # Start Celery worker
    celery_app.start()
