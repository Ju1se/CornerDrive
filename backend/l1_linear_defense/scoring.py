"""Cheap L1 visibility scores for CornerDrive V4.1.

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
    pred_delta_main: Optional[float]
    pred_delta_corner: Optional[float]
    main_alignment: Optional[float]
    corner_alignment: Optional[float]
    main_harm_proxy: Optional[float]
    corner_harm_proxy: Optional[float]
    corner_benefit_proxy: Optional[float]
    dual_conflict_score: Optional[float]
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


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


def first_order_delta_scores(
    gradients: list[np.ndarray],
    validation_gradient: np.ndarray | None,
    learning_rate: float,
) -> list[float]:
    """Estimate validation loss drift with a first-order Taylor proxy.

    L2 measures ΔL = L(W - ηg_i; D) - L(W; D). V4.1 approximates this as
    -η <∇L(W; D), g_i>, which is cheap once the validation gradient is known.
    """

    if validation_gradient is None:
        return [0.0 for _ in gradients]
    ref = np.ravel(validation_gradient)
    return [
        _safe_float(-float(learning_rate) * float(np.dot(ref, np.ravel(gradient))))
        for gradient in gradients
    ]


def compute_l1_scores(
    gradients: list[np.ndarray],
    vehicle_ids: list[str],
    reference: np.ndarray,
    config: L1RouterConfig,
    *,
    main_validation_gradient: np.ndarray | None = None,
    corner_validation_gradient: np.ndarray | None = None,
    learning_rate: float = 1.0,
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
    pred_delta_main = first_order_delta_scores(
        gradients,
        main_validation_gradient if config.uses_dual_proxy else None,
        learning_rate,
    )
    pred_delta_corner = first_order_delta_scores(
        gradients,
        corner_validation_gradient if config.uses_dual_proxy else None,
        learning_rate,
    )
    main_alignments = [
        _safe_float(_cosine_similarity(gradient, main_validation_gradient))
        if config.uses_dual_proxy and main_validation_gradient is not None
        else 0.0
        for gradient in gradients
    ]
    corner_alignments = [
        _safe_float(_cosine_similarity(gradient, corner_validation_gradient))
        if config.uses_dual_proxy and corner_validation_gradient is not None
        else 0.0
        for gradient in gradients
    ]
    main_harm_scores = [
        max(0.0, value - config.theta_main_proxy)
        for value in pred_delta_main
    ]
    corner_harm_scores = [
        max(0.0, value - config.theta_corner_harm_proxy)
        for value in pred_delta_corner
    ]
    corner_benefit_scores = [
        max(0.0, -value)
        for value in pred_delta_corner
    ]
    dual_conflict_scores = [
        max(0.0, -main_delta) * max(0.0, corner_delta)
        for main_delta, corner_delta in zip(pred_delta_main, pred_delta_corner)
    ]

    cos_rank = rank_normalize(cosine_deviations)
    norm_rank = rank_normalize(norm_scores) if config.uses_norm else [0.0] * len(gradients)
    sign_rank = rank_normalize(sign_scores) if config.uses_sign else [0.0] * len(gradients)
    main_harm_rank = (
        rank_normalize(main_harm_scores) if config.uses_dual_proxy else [0.0] * len(gradients)
    )
    corner_harm_rank = (
        rank_normalize(corner_harm_scores) if config.uses_dual_proxy else [0.0] * len(gradients)
    )
    corner_benefit_rank = (
        rank_normalize(corner_benefit_scores) if config.uses_dual_proxy else [0.0] * len(gradients)
    )

    weights: list[tuple[float, list[float]]] = [(config.cos_weight, cos_rank)]
    if config.uses_norm:
        weights.append((config.norm_weight, norm_rank))
    if config.uses_sign:
        weights.append((config.sign_weight, sign_rank))
    if config.uses_dual_proxy:
        weights.append((config.dual_main_weight, main_harm_rank))
        weights.append((config.dual_corner_harm_weight, corner_harm_rank))
        weights.append((config.dual_corner_benefit_weight, corner_benefit_rank))
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
                pred_delta_main=pred_delta_main[idx] if config.uses_dual_proxy else None,
                pred_delta_corner=pred_delta_corner[idx] if config.uses_dual_proxy else None,
                main_alignment=main_alignments[idx] if config.uses_dual_proxy else None,
                corner_alignment=corner_alignments[idx] if config.uses_dual_proxy else None,
                main_harm_proxy=main_harm_scores[idx] if config.uses_dual_proxy else None,
                corner_harm_proxy=corner_harm_scores[idx] if config.uses_dual_proxy else None,
                corner_benefit_proxy=corner_benefit_scores[idx] if config.uses_dual_proxy else None,
                dual_conflict_score=dual_conflict_scores[idx] if config.uses_dual_proxy else None,
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
