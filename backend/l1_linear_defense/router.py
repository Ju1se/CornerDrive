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
    quarantine_indices: list[int] | None = None
    low_weight_indices: list[int] | None = None
    route_actions: dict[int, str] | None = None
    aggregation_weights: dict[int, float] | None = None


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
    if (
        score.main_harm_proxy is not None
        and score.main_harm_proxy > 0.0
    ):
        return "main_harm_proxy"
    if (
        score.corner_harm_proxy is not None
        and score.corner_harm_proxy > 0.0
    ):
        return "corner_harm_proxy"
    return None


def _harm_priority(score: L1Scores) -> float:
    return (
        score.risk_score
        + float(score.main_harm_proxy or 0.0)
        + float(score.corner_harm_proxy or 0.0)
        + float(score.dual_conflict_score or 0.0)
    )


def _rarity_priority(score: L1Scores) -> float:
    return (
        float(score.corner_benefit_proxy or 0.0)
        + 0.25 * score.risk_score
        - float(score.main_harm_proxy or 0.0)
    )


def _uncertainty_priority(score: L1Scores) -> float:
    middle_risk = 1.0 - min(1.0, abs(score.risk_score - 0.5) * 2.0)
    return middle_risk + float(score.dual_conflict_score or 0.0)


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

    if config.uses_dual_proxy:
        return _route_dual_proxy(scores, config, rng)

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


def _route_dual_proxy(
    scores: list[L1Scores],
    config: L1RouterConfig,
    rng: random.Random | None,
) -> L1RouteResult:
    n = len(scores)
    budget = max(1, min(n, int(ceil(config.queue_budget_ratio * n))))
    harm_budget = max(1, int(ceil(budget * 0.60)))
    rarity_budget = max(0, int(ceil(budget * 0.20)))
    uncertainty_budget = max(0, budget - harm_budget - rarity_budget)

    routed: dict[int, str] = {}

    def add_ranked(
        candidates: list[L1Scores],
        quota: int,
        reason: str,
        *,
        require_positive: bool = False,
    ) -> int:
        added = 0
        for score in candidates:
            if added >= quota or len(routed) >= budget:
                break
            if score.index in routed:
                continue
            if require_positive and float(score.corner_benefit_proxy or 0.0) <= 0.0:
                continue
            routed[score.index] = f"audit:{reason}"
            added += 1
        return added

    hard_candidates = [
        (score, _threshold_reason(score, config))
        for score in scores
        if _threshold_reason(score, config) is not None
    ]
    hard_candidates.sort(key=lambda item: _harm_priority(item[0]), reverse=True)
    for score, reason in hard_candidates[:harm_budget]:
        routed[score.index] = f"audit:{reason or 'harm_proxy'}"

    harm_candidates = sorted(scores, key=_harm_priority, reverse=True)
    add_ranked(harm_candidates, max(0, harm_budget - len(routed)), "harm_topB")

    rarity_candidates = sorted(scores, key=_rarity_priority, reverse=True)
    add_ranked(rarity_candidates, rarity_budget, "rarity_proxy", require_positive=True)

    uncertainty_candidates = sorted(scores, key=_uncertainty_priority, reverse=True)
    add_ranked(uncertainty_candidates, uncertainty_budget, "uncertainty")

    remaining_budget = max(0, budget - len(routed))
    if remaining_budget > 0:
        add_ranked(harm_candidates, remaining_budget, "risk_topB")

    non_routed = [
        score for score in scores
        if score.index not in routed
    ]
    random_quota = int(config.random_recheck_ratio * n)
    if config.random_recheck_ratio > 0.0 and random_quota == 0 and non_routed:
        random_quota = 1
    for score in stratified_sample_by_risk(non_routed, random_quota, rng):
        routed[score.index] = "audit:stratified_random"

    route_actions: dict[int, str] = {}
    routing_reasons: dict[int, str] = {}
    weights: dict[int, float] = {}
    for score in scores:
        route = routed.get(score.index)
        if route is not None:
            route_actions[score.index] = "AUDIT"
            routing_reasons[score.index] = route
            weights[score.index] = 0.0
            continue

        has_corner_benefit = float(score.corner_benefit_proxy or 0.0) > 0.0
        has_harm_proxy = (
            float(score.main_harm_proxy or 0.0) > 0.0
            or float(score.corner_harm_proxy or 0.0) > 0.0
        )
        if (
            has_harm_proxy
            or (
                score.risk_score >= config.quarantine_risk_threshold
                and not has_corner_benefit
            )
        ):
            route_actions[score.index] = "QUARANTINE"
            routing_reasons[score.index] = "quarantine:harm_or_high_risk"
            weights[score.index] = 0.0
        elif score.risk_score >= config.low_weight_risk_threshold and not has_corner_benefit:
            route_actions[score.index] = "LOW_WEIGHT"
            routing_reasons[score.index] = "low_weight:moderate_risk"
            weights[score.index] = config.low_weight
        else:
            route_actions[score.index] = "SAFE_ACCEPT"
            routing_reasons[score.index] = "safe_accept"
            weights[score.index] = config.safe_weight

    suspect_indices = sorted(idx for idx, action in route_actions.items() if action == "AUDIT")
    quarantine_indices = sorted(idx for idx, action in route_actions.items() if action == "QUARANTINE")
    low_weight_indices = sorted(idx for idx, action in route_actions.items() if action == "LOW_WEIGHT")
    clean_indices = sorted(
        idx for idx, action in route_actions.items()
        if action in {"SAFE_ACCEPT", "LOW_WEIGHT"}
    )
    return L1RouteResult(
        suspect_indices=suspect_indices,
        clean_indices=clean_indices,
        routing_reasons=routing_reasons,
        quarantine_indices=quarantine_indices,
        low_weight_indices=low_weight_indices,
        route_actions=route_actions,
        aggregation_weights=weights,
    )


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
    route_actions = {
        score.index: "AUDIT" if score.index in routed else "SAFE_ACCEPT"
        for score in scores
    }
    aggregation_weights = {
        score.index: 0.0 if score.index in routed else 1.0
        for score in scores
    }
    return L1RouteResult(
        suspect_indices=suspect_indices,
        clean_indices=clean_indices,
        routing_reasons=routing_reasons,
        quarantine_indices=[],
        low_weight_indices=[],
        route_actions=route_actions,
        aggregation_weights=aggregation_weights,
    )
