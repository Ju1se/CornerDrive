"""Configuration for L1 visibility routing modes.

L1 routes updates to L2 for evidence-backed judgement. It does not assign
Fraud/Rarity/Noise verdicts and must not perform settlement actions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


L1RouterMode = Literal[
    "v25_cosine_fixed",
    "v3_m1_norm_fixed",
    "v3_m2_norm_sign_fixed",
    "v3_m3_budgeted",
    "v3_m4_reputation_age",
]


MODE_ALIASES: dict[str, L1RouterMode] = {
    "m0": "v25_cosine_fixed",
    "baseline": "v25_cosine_fixed",
    "v25": "v25_cosine_fixed",
    "v25_cosine_fixed": "v25_cosine_fixed",
    "m1": "v3_m1_norm_fixed",
    "norm": "v3_m1_norm_fixed",
    "v3_m1_norm_fixed": "v3_m1_norm_fixed",
    "m2": "v3_m2_norm_sign_fixed",
    "norm_sign": "v3_m2_norm_sign_fixed",
    "v3_m2_norm_sign_fixed": "v3_m2_norm_sign_fixed",
    "m3": "v3_m3_budgeted",
    "budgeted": "v3_m3_budgeted",
    "v3_m3_budgeted": "v3_m3_budgeted",
    "m4": "v3_m4_reputation_age",
    "full": "v3_m4_reputation_age",
    "l1v3": "v3_m4_reputation_age",
    "v3_m4_reputation_age": "v3_m4_reputation_age",
}


@dataclass(frozen=True)
class L1RouterConfig:
    mode: L1RouterMode = "v25_cosine_fixed"

    # Direction
    cos_weight: float = 0.35
    cos_deviation_threshold: float = 0.70

    # Norm / magnitude
    use_norm_mad: bool = True
    norm_weight: float = 0.20
    norm_mad_threshold: float = 3.0
    eps: float = 1e-12

    # Sign
    use_sign_score: bool = True
    sign_weight: float = 0.15
    sign_topk_ratio: float = 0.10
    sign_threshold: float = 0.65

    # Reputation / history
    use_reputation: bool = True
    reputation_weight: float = 0.20
    audit_age_weight: float = 0.10
    audit_age_cap: int = 10

    # Routing budget
    queue_budget_ratio: float = 0.35
    random_recheck_ratio: float = 0.05
    min_recheck_prob: float = 0.02
    max_recheck_prob: float = 0.30

    @property
    def uses_budget(self) -> bool:
        return self.mode in {"v3_m3_budgeted", "v3_m4_reputation_age"}

    @property
    def uses_norm(self) -> bool:
        return self.mode in {
            "v3_m1_norm_fixed",
            "v3_m2_norm_sign_fixed",
            "v3_m3_budgeted",
            "v3_m4_reputation_age",
        } and self.use_norm_mad

    @property
    def uses_sign(self) -> bool:
        return self.mode in {
            "v3_m2_norm_sign_fixed",
            "v3_m3_budgeted",
            "v3_m4_reputation_age",
        } and self.use_sign_score

    @property
    def uses_reputation_age(self) -> bool:
        return self.mode == "v3_m4_reputation_age" and self.use_reputation


def normalize_l1_mode(raw: str | None) -> L1RouterMode:
    key = (raw or "v25_cosine_fixed").strip().lower()
    if key not in MODE_ALIASES:
        supported = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Unsupported L1 router mode {raw!r}. Supported: {supported}")
    return MODE_ALIASES[key]


def make_l1_router_config(mode: str | None = None, **overrides) -> L1RouterConfig:
    payload = {"mode": normalize_l1_mode(mode)}
    payload.update(overrides)
    return L1RouterConfig(**payload)


def l1_router_config_from_env() -> L1RouterConfig:
    return make_l1_router_config(
        os.getenv("L1_ROUTER_MODE", "v25_cosine_fixed"),
        queue_budget_ratio=float(os.getenv("L1_QUEUE_BUDGET_RATIO", "0.35")),
        random_recheck_ratio=float(os.getenv("L1_RANDOM_RECHECK_RATIO", "0.05")),
    )
