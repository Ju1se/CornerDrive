"""
Helpers for per-round audit statistics used by policy telemetry.
"""

from __future__ import annotations

from typing import Mapping


ROUND_STATS_KEY = "stats:round:r{round_id}"

CLASSIFICATION_FIELD_MAP = {
    "FRAUD": "fraud_count",
    "RARITY": "rare_count",
    "HONEST": "honest_count",
    "NOISE": "noise_count",
}


def round_stats_key(round_id: int) -> str:
    """Build the Redis key for one round's audit counters."""
    return ROUND_STATS_KEY.format(round_id=round_id)


def _coerce_number(raw: object, default: float = 0.0) -> float:
    """Decode Redis bytes/strings into floats."""
    if raw is None:
        return default
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def read_round_stats(raw_stats: Mapping[object, object] | None) -> dict[str, float]:
    """Normalize Redis hash data into numeric per-round stats."""
    stats = raw_stats or {}
    normalized = {
        "fraud_count": _coerce_number(stats.get("fraud_count") or stats.get(b"fraud_count")),
        "rare_count": _coerce_number(stats.get("rare_count") or stats.get(b"rare_count")),
        "honest_count": _coerce_number(stats.get("honest_count") or stats.get(b"honest_count")),
        "noise_count": _coerce_number(stats.get("noise_count") or stats.get(b"noise_count")),
        "audit_count": _coerce_number(stats.get("audit_count") or stats.get(b"audit_count")),
        "total_rewards": _coerce_number(stats.get("total_rewards") or stats.get(b"total_rewards")),
        "total_slashed": _coerce_number(stats.get("total_slashed") or stats.get(b"total_slashed")),
    }

    counted_total = (
        normalized["fraud_count"]
        + normalized["rare_count"]
        + normalized["honest_count"]
        + normalized["noise_count"]
    )
    if normalized["audit_count"] < counted_total:
        normalized["audit_count"] = counted_total

    return normalized


def compute_round_rates(round_stats: Mapping[object, object] | None) -> dict[str, float]:
    """Compute per-round classification rates from normalized stats."""
    stats = read_round_stats(round_stats)
    total = max(int(stats["audit_count"]), 1)
    return {
        "fraud_rate": stats["fraud_count"] / total,
        "rarity_rate": stats["rare_count"] / total,
        "honest_rate": stats["honest_count"] / total,
        "noise_rate": stats["noise_count"] / total,
        "audit_sample_size": int(stats["audit_count"]),
    }


def compute_recent_attack_pressure(
    fraud_rates: list[float],
    alpha: float = 0.6,
) -> float:
    """
    Compute an EWMA over recent fraud rates.

    More recent rounds receive higher weight.
    """
    if not fraud_rates:
        return 0.0

    smoothed = max(0.0, min(1.0, fraud_rates[0]))
    for rate in fraud_rates[1:]:
        bounded_rate = max(0.0, min(1.0, rate))
        smoothed = alpha * bounded_rate + (1.0 - alpha) * smoothed

    return smoothed
