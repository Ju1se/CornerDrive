"""
L1: Linear Defense - Byzantine-Robust Aggregation
Implements Geometric Median and Cosine Deviation Filter.
"""

import logging
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from common.config import (
    L1_GEOMETRIC_MEDIAN_MAX_ITER,
    L1_GEOMETRIC_MEDIAN_EPS,
    L1_SUSPECT_THRESHOLD,
)
from .config import L1RouterConfig
from .router import route_l1
from .scoring import compute_l1_scores

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of L1 aggregation."""
    aggregated_gradient: np.ndarray
    clean_indices: List[int]
    suspect_indices: List[int]
    cosine_scores: Dict[int, float]
    iterations: int
    routing_reasons: Dict[int, str] = field(default_factory=dict)
    l1_score_details: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    router_mode: str = "v25_cosine_fixed"
    quarantine_indices: List[int] = field(default_factory=list)
    low_weight_indices: List[int] = field(default_factory=list)
    route_actions: Dict[int, str] = field(default_factory=dict)
    aggregation_weights: Dict[int, float] = field(default_factory=dict)


def geometric_median(
    gradients: List[np.ndarray],
    max_iter: int = L1_GEOMETRIC_MEDIAN_MAX_ITER,
    eps: float = L1_GEOMETRIC_MEDIAN_EPS,
) -> Tuple[np.ndarray, int]:
    """
    Compute geometric median using Weiszfeld algorithm.

    Formula:
        w* = argmin_w Σᵢ ||w - ĝᵢ||₂

    Update rule:
        w_{t+1} = (Σᵢ ĝᵢ/||w_t - ĝᵢ||₂) / (Σᵢ 1/||w_t - ĝᵢ||₂)

    Args:
        gradients: List of gradient vectors from vehicles
        max_iter: Maximum iterations for convergence
        eps: Convergence threshold and numerical stability

    Returns:
        Tuple of (geometric_median, iterations_used)
    """
    if len(gradients) == 0:
        raise ValueError("Cannot compute geometric median of empty list")

    if len(gradients) == 1:
        return gradients[0].copy(), 1

    # Stack gradients into matrix
    G = np.stack(gradients)  # Shape: (n_vehicles, gradient_dim)
    n_vehicles = G.shape[0]

    # Initialize with arithmetic mean
    w = np.mean(G, axis=0)

    for iteration in range(max_iter):
        # Compute distances from current estimate to all gradients
        distances = np.linalg.norm(G - w, axis=1)  # Shape: (n_vehicles,)

        # Add epsilon for numerical stability (avoid division by zero)
        distances = np.maximum(distances, eps)

        # Compute weights: 1 / ||w - ĝᵢ||
        weights = 1.0 / distances  # Shape: (n_vehicles,)

        # Weiszfeld update: w_new = Σ(ĝᵢ * wᵢ) / Σ(wᵢ)
        w_new = np.sum(G * weights[:, np.newaxis], axis=0) / np.sum(weights)

        # Check convergence
        change = np.linalg.norm(w_new - w)
        w = w_new

        if change < eps:
            logger.debug(f"Geometric median converged in {iteration + 1} iterations")
            return w, iteration + 1

    logger.warning(f"Geometric median did not converge in {max_iter} iterations")
    return w, max_iter


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: cos(θ) = (a · b) / (||a||₂ · ||b||₂)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)


def cosine_deviation_score(gradient: np.ndarray, median: np.ndarray) -> float:
    """
    Compute cosine deviation score for outlier detection.

    Formula: Score_i = 1 - (ĝᵢ · w*) / (||ĝᵢ||₂ · ||w*||₂)

    Interpretation:
        - Score ≈ 0: Gradient aligned with median (likely honest)
        - Score ≈ 1: Gradient orthogonal to median (suspicious)
        - Score ≈ 2: Gradient opposite to median (likely malicious)
    """
    similarity = cosine_similarity(gradient, median)
    return 1.0 - similarity


def filter_suspects(
    gradients: List[np.ndarray],
    vehicle_ids: List[str],
    threshold: float = L1_SUSPECT_THRESHOLD,
    recheck_probability: float = 0.0,
    rng: Optional[random.Random] = None,
    router_config: Optional[L1RouterConfig] = None,
    main_validation_gradient: Optional[np.ndarray] = None,
    corner_validation_gradient: Optional[np.ndarray] = None,
    learning_rate: float = 1.0,
    theta_tol: Optional[float] = None,
    theta_corner_harm: Optional[float] = None,
    client_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
    current_round: int = 0,
) -> AggregationResult:
    """
    Main L1 pipeline: Aggregate gradients and identify suspects.

    Process:
        1. Compute geometric median of all gradients
        2. Calculate cosine deviation score for each gradient
        3. Flag gradients with score > threshold as suspects
        4. Optionally route a low-probability sample of apparent-clean
           gradients for L2 recheck
        5. Return clean aggregation and suspect list for L2

    Args:
        gradients: List of gradient vectors
        vehicle_ids: Corresponding vehicle identifiers
        threshold: Cosine deviation threshold for suspect detection
        recheck_probability: Probability of routing an apparent-clean
            gradient to L2 for blind recheck
        rng: Optional deterministic RNG for reproducible benchmarks
        router_config: Optional L1 router mode/config. Defaults to the
            V2.5 cosine + fixed recheck behavior.
        client_states: Optional cross-round state retained for API
            compatibility. The canonical V4.1 router does not use reputation
            or audit-age features.
        current_round: Current round id retained for API compatibility.

    Returns:
        AggregationResult with clean gradients and suspect list
    """
    if len(gradients) != len(vehicle_ids):
        raise ValueError("Gradients and vehicle_ids must have same length")

    if len(gradients) == 0:
        raise ValueError("Cannot filter empty gradient list")

    # Step 1: Compute geometric median
    median, iterations = geometric_median(gradients)
    router_mode = router_config.mode if router_config is not None else "v25_cosine_fixed"
    if router_config is not None:
        replace_payload: Dict[str, Any] = {"cos_deviation_threshold": threshold}
        if theta_tol is not None:
            replace_payload["theta_main_proxy"] = float(theta_tol)
        if theta_corner_harm is not None:
            replace_payload["theta_corner_harm_proxy"] = float(theta_corner_harm)
        router_config = replace(router_config, **replace_payload)

    # Step 2: Compute cosine deviation scores
    scores = {}
    routing_reasons = {}
    clean_indices = []
    suspect_indices = []
    p_recheck = max(0.0, min(1.0, float(recheck_probability)))
    random_draw = rng.random if rng is not None else random.random

    l1_score_details: Dict[int, Dict[str, Any]] = {}
    if router_config is None or router_config.mode == "v25_cosine_fixed":
        for i, (gradient, vid) in enumerate(zip(gradients, vehicle_ids)):
            score = cosine_deviation_score(gradient, median)
            scores[i] = score

            if score > threshold:
                suspect_indices.append(i)
                routing_reasons[i] = "cosine_screening"
                logger.info(f"Vehicle {vid} flagged as suspect (score={score:.4f})")
            elif p_recheck > 0.0 and random_draw() < p_recheck:
                suspect_indices.append(i)
                routing_reasons[i] = "probabilistic_recheck"
                logger.info(
                    "Vehicle %s routed for probabilistic L2 recheck "
                    "(score=%.4f, p=%.3f)",
                    vid,
                    score,
                    p_recheck,
                )
            else:
                clean_indices.append(i)
                routing_reasons[i] = "bypass"
    else:
        l1_scores = compute_l1_scores(
            gradients,
            vehicle_ids,
            median,
            router_config,
            main_validation_gradient=main_validation_gradient,
            corner_validation_gradient=corner_validation_gradient,
            learning_rate=learning_rate,
            client_states=client_states,
            current_round=current_round,
        )
        routed = route_l1(
            l1_scores,
            router_config,
            recheck_probability=recheck_probability,
            rng=rng,
        )
        clean_indices = routed.clean_indices
        suspect_indices = routed.suspect_indices
        routing_reasons = routed.routing_reasons
        quarantine_indices = list(routed.quarantine_indices or [])
        low_weight_indices = list(routed.low_weight_indices or [])
        route_actions = dict(routed.route_actions or {})
        aggregation_weights = dict(routed.aggregation_weights or {})
        scores = {
            score.index: score.cosine_deviation
            for score in l1_scores
        }
        l1_score_details = {
            score.index: score.to_dict()
            for score in l1_scores
        }
        for idx in suspect_indices:
            logger.info(
                "Vehicle %s routed to L2 by %s (risk=%.4f, deviation=%.4f)",
                vehicle_ids[idx],
                routing_reasons.get(idx, "unknown"),
                l1_score_details[idx]["risk_score"],
                scores[idx],
            )
    if router_config is None or router_config.mode == "v25_cosine_fixed":
        quarantine_indices = []
        low_weight_indices = []
        route_actions = {
            idx: "AUDIT" if idx in suspect_indices else "SAFE_ACCEPT"
            for idx in range(len(gradients))
        }
        aggregation_weights = {
            idx: 0.0 if idx in suspect_indices else 1.0
            for idx in range(len(gradients))
        }

    # Step 3: Compute clean aggregation (only from non-suspects)
    if clean_indices:
        clean_gradients = [gradients[i] for i in clean_indices]
        if router_config is not None and router_config.uses_dual_proxy:
            weights = np.array(
                [aggregation_weights.get(i, 1.0) for i in clean_indices],
                dtype=np.float64,
            )
            total_weight = float(np.sum(weights))
            if total_weight > 1e-12:
                stacked = np.stack(clean_gradients)
                aggregated = np.sum(stacked * weights[:, np.newaxis], axis=0) / total_weight
            else:
                aggregated = np.zeros_like(gradients[0])
        else:
            aggregated, _ = geometric_median(clean_gradients)
    else:
        # If all are suspects, use median of all (fallback)
        logger.warning("All gradients flagged as suspects, using full median")
        aggregated = median

    logger.info(
        f"L1 Filter: {len(clean_indices)} clean, {len(suspect_indices)} suspects "
        f"out of {len(gradients)} total"
    )

    return AggregationResult(
        aggregated_gradient=aggregated,
        clean_indices=clean_indices,
        suspect_indices=suspect_indices,
        cosine_scores=scores,
        routing_reasons=routing_reasons,
        l1_score_details=l1_score_details,
        router_mode=router_mode,
        quarantine_indices=quarantine_indices,
        low_weight_indices=low_weight_indices,
        route_actions=route_actions,
        aggregation_weights=aggregation_weights,
        iterations=iterations,
    )


def weighted_aggregation(
    gradients: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted average aggregation (for reputation-weighted updates).

    Formula: w_agg = Σ(wᵢ · ĝᵢ) / Σ(wᵢ)
    """
    if len(gradients) != len(weights):
        raise ValueError("Gradients and weights must have same length")

    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize

    G = np.stack(gradients)
    return np.sum(G * weights[:, np.newaxis], axis=0)
