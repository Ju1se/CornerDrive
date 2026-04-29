"""Routing policy for CornerDrive L1 visibility.

The router decides which updates receive L2 attention. It deliberately returns
only routing decisions and reasons; audit verdicts remain owned by L2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil

from .config import L1RouterConfig
from .scoring import L1Scores


@dataclass(frozen=True)
class L1RouteResult:
    suspect_indices: list[int]
    clean_indices: list[int]
    routing_reasons: dict[int, str]


def _threshold_reason(score: L1Scores, config: L1RouterConfig) -> str | None:
    if score.cosine_deviation > config.cos_deviation_threshold:
        return "cosine_screening"
    if (
        score.norm_mad_score is not None
        and score.norm_mad_score > config.norm_mad_threshold
    ):
        return "norm_mad_screening"
    if (
        score.sign_disagreement is not None
        and score.sign_disagreement > config.sign_threshold
    ):
        return "sign_screening"
    return None


def stratified_sample_by_risk(
    scores: list[L1Scores],
    quota: int,
    rng: random.Random | None,
) -> list[L1Scores]:
    if quota <= 0 or not scores:
        return []
    if quota >= len(scores):
        return list(scores)

    random_draw = rng.random if rng is not None else random.random
    ordered = sorted(scores, key=lambda score: score.risk_score)
    strata: list[list[L1Scores]] = [[], [], []]
    for position, score in enumerate(ordered):
        bucket = min(2, int(position * 3 / max(len(ordered), 1)))
        strata[bucket].append(score)

    selected: list[L1Scores] = []
    selected_indices: set[int] = set()
    per_stratum = max(1, quota // len(strata))
    for stratum in strata:
        if len(selected) >= quota or not stratum:
            continue
        shuffled = list(stratum)
        for idx in range(len(shuffled) - 1, 0, -1):
            swap_idx = int(random_draw() * (idx + 1))
            shuffled[idx], shuffled[swap_idx] = shuffled[swap_idx], shuffled[idx]
        for score in shuffled[:per_stratum]:
            if score.index not in selected_indices:
                selected.append(score)
                selected_indices.add(score.index)
                if len(selected) >= quota:
                    return selected

    remainder = [score for score in scores if score.index not in selected_indices]
    for idx in range(len(remainder) - 1, 0, -1):
        swap_idx = int(random_draw() * (idx + 1))
        remainder[idx], remainder[swap_idx] = remainder[swap_idx], remainder[idx]
    selected.extend(remainder[: max(0, quota - len(selected))])
    return selected


def route_l1(
    scores: list[L1Scores],
    config: L1RouterConfig,
    *,
    recheck_probability: float = 0.0,
    rng: random.Random | None = None,
) -> L1RouteResult:
    """Route updates to L2 without assigning any final verdict."""

    n = len(scores)
    if n == 0:
        return L1RouteResult([], [], {})

    random_draw = rng.random if rng is not None else random.random
    routed: dict[int, str] = {}

    if not config.uses_budget:
        p_recheck = max(0.0, min(1.0, float(recheck_probability)))
        for score in scores:
            threshold_reason = _threshold_reason(score, config)
            if threshold_reason is not None:
                routed[score.index] = threshold_reason
            elif p_recheck > 0.0 and random_draw() < p_recheck:
                routed[score.index] = "probabilistic_recheck"
        return _build_route_result(scores, routed)

    hard_candidates = [
        (score, _threshold_reason(score, config))
        for score in scores
        if _threshold_reason(score, config) is not None
    ]
    hard_candidates.sort(key=lambda item: item[0].risk_score, reverse=True)

    budget = max(1, min(n, int(ceil(config.queue_budget_ratio * n))))
    for score, reason in hard_candidates[:budget]:
        routed[score.index] = reason or "risk_topB"

    remaining_budget = max(0, budget - len(routed))
    if remaining_budget > 0:
        remaining = [
            score for score in scores
            if score.index not in routed
        ]
        remaining.sort(key=lambda score: score.risk_score, reverse=True)
        for score in remaining[:remaining_budget]:
            routed[score.index] = "risk_topB"

    non_routed = [
        score for score in scores
        if score.index not in routed
    ]
    random_quota = int(config.random_recheck_ratio * n)
    if config.random_recheck_ratio > 0.0 and random_quota == 0 and non_routed:
        random_quota = 1
    for score in stratified_sample_by_risk(non_routed, random_quota, rng):
        routed[score.index] = "stratified_random"

    return _build_route_result(scores, routed)


def _build_route_result(
    scores: list[L1Scores],
    routed: dict[int, str],
) -> L1RouteResult:
    suspect_indices = sorted(routed)
    clean_indices = sorted(score.index for score in scores if score.index not in routed)
    routing_reasons = {
        score.index: routed.get(score.index, "bypass")
        for score in scores
    }
    return L1RouteResult(
        suspect_indices=suspect_indices,
        clean_indices=clean_indices,
        routing_reasons=routing_reasons,
    )
