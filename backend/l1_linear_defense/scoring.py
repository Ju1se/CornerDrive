"""Cheap L1 visibility scores for CornerDrive-L1V3.

These scores are routing evidence only. They do not assign audit verdicts and
must not be used for rejection or settlement without L2/L4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log
from typing import Any, Mapping, Optional

import numpy as np

from .config import L1RouterConfig


@dataclass(frozen=True)
class L1Scores:
    index: int
    client_id: str
    cosine_similarity: float
    cosine_deviation: float
    log_norm: float
    norm_mad_score: Optional[float]
    sign_disagreement: Optional[float]
    reputation_risk: Optional[float]
    audit_age_score: Optional[float]
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def rank_normalize(values: list[float]) -> list[float]:
    """Return within-round percentile ranks in [0, 1].

    Ties receive the average rank. If every value is equal, all ranks are zero
    because the signal carries no within-round ordering information.
    """

    if not values:
        return []
    n = len(values)
    if n == 1:
        return [0.0]

    clean_values = [_safe_float(value) for value in values]
    if max(clean_values) - min(clean_values) <= 1e-12:
        return [0.0 for _ in clean_values]

    order = sorted(range(n), key=lambda idx: (clean_values[idx], idx))
    ranks = [0.0] * n
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and clean_values[order[end]] == clean_values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for pos in range(cursor, end):
            ranks[order[pos]] = average_rank / (n - 1)
        cursor = end
    return ranks


def norm_mad_scores(gradients: list[np.ndarray], eps: float) -> tuple[list[float], list[float]]:
    log_norms = [
        log(float(np.linalg.norm(np.ravel(gradient))) + eps)
        for gradient in gradients
    ]
    median_log_norm = float(np.median(log_norms))
    deviations = [abs(value - median_log_norm) for value in log_norms]
    mad = float(np.median(deviations))
    if mad <= eps:
        return log_norms, [0.0 for _ in gradients]
    return log_norms, [deviation / mad for deviation in deviations]


def sign_disagreement_scores(
    gradients: list[np.ndarray],
    reference: np.ndarray,
    topk_ratio: float,
    eps: float,
) -> list[float]:
    ref = np.ravel(reference)
    if ref.size == 0:
        return [0.0 for _ in gradients]

    nonzero = np.flatnonzero(np.abs(ref) > eps)
    if nonzero.size == 0:
        return [0.0 for _ in gradients]

    ratio = max(0.0, min(1.0, float(topk_ratio)))
    topk = max(1, min(nonzero.size, int(ceil(nonzero.size * ratio))))
    top_positions = nonzero[np.argpartition(np.abs(ref[nonzero]), -topk)[-topk:]]
    ref_sign = np.sign(ref[top_positions])

    disagreements: list[float] = []
    for gradient in gradients:
        grad_flat = np.ravel(gradient)
        grad_sign = np.sign(grad_flat[top_positions])
        disagreements.append(float(np.mean(grad_sign != ref_sign)))
    return disagreements


def reputation_risk_from_state(state: Mapping[str, Any] | None) -> float:
    if state is None:
        return 0.0
    reputation = _clamp(float(state.get("reputation", 1.0)))
    fraud_count = float(
        state.get("recent_fraud_count", state.get("fraud_count", 0.0))
    )
    low_reputation_risk = 1.0 - reputation
    recent_fraud_risk = _clamp(fraud_count / 3.0)
    return _clamp(low_reputation_risk + recent_fraud_risk)


def audit_age_from_state(
    state: Mapping[str, Any] | None,
    *,
    current_round: int,
    cap: int,
) -> float:
    if cap <= 0:
        return 0.0
    if state is None or state.get("last_audit_round") is None:
        return 1.0
    age = max(0, int(current_round) - int(state["last_audit_round"]))
    return _clamp(age / cap)


def compute_l1_scores(
    gradients: list[np.ndarray],
    vehicle_ids: list[str],
    reference: np.ndarray,
    config: L1RouterConfig,
    *,
    client_states: Mapping[str, Mapping[str, Any]] | None = None,
    current_round: int = 0,
) -> list[L1Scores]:
    if len(gradients) != len(vehicle_ids):
        raise ValueError("Gradients and vehicle_ids must have same length")

    similarities = [
        _safe_float(_cosine_similarity(gradient, reference))
        for gradient in gradients
    ]
    cosine_deviations = [1.0 - similarity for similarity in similarities]
    log_norms, norm_scores = norm_mad_scores(gradients, config.eps)
    sign_scores = sign_disagreement_scores(
        gradients,
        reference,
        config.sign_topk_ratio,
        config.eps,
    )
    rep_scores = [
        reputation_risk_from_state(
            client_states.get(vehicle_id) if client_states is not None else None
        )
        for vehicle_id in vehicle_ids
    ]
    age_scores = [
        audit_age_from_state(
            client_states.get(vehicle_id) if client_states is not None else None,
            current_round=current_round,
            cap=config.audit_age_cap,
        )
        for vehicle_id in vehicle_ids
    ]

    cos_rank = rank_normalize(cosine_deviations)
    norm_rank = rank_normalize(norm_scores) if config.uses_norm else [0.0] * len(gradients)
    sign_rank = rank_normalize(sign_scores) if config.uses_sign else [0.0] * len(gradients)
    rep_rank = rank_normalize(rep_scores) if config.uses_reputation_age else [0.0] * len(gradients)
    age_rank = rank_normalize(age_scores) if config.uses_reputation_age else [0.0] * len(gradients)

    weights: list[tuple[float, list[float]]] = [(config.cos_weight, cos_rank)]
    if config.uses_norm:
        weights.append((config.norm_weight, norm_rank))
    if config.uses_sign:
        weights.append((config.sign_weight, sign_rank))
    if config.uses_reputation_age:
        weights.append((config.reputation_weight, rep_rank))
        weights.append((config.audit_age_weight, age_rank))
    weight_total = sum(weight for weight, _ in weights) or 1.0

    scores: list[L1Scores] = []
    for idx, vehicle_id in enumerate(vehicle_ids):
        risk_score = sum(weight * ranks[idx] for weight, ranks in weights) / weight_total
        scores.append(
            L1Scores(
                index=idx,
                client_id=vehicle_id,
                cosine_similarity=similarities[idx],
                cosine_deviation=cosine_deviations[idx],
                log_norm=log_norms[idx],
                norm_mad_score=norm_scores[idx] if config.uses_norm else None,
                sign_disagreement=sign_scores[idx] if config.uses_sign else None,
                reputation_risk=rep_scores[idx] if config.uses_reputation_age else None,
                audit_age_score=age_scores[idx] if config.uses_reputation_age else None,
                risk_score=float(risk_score),
            )
        )
    return scores


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(np.ravel(a), np.ravel(b)) / (norm_a * norm_b))
