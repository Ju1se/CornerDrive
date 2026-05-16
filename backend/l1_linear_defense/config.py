"""Configuration for L1 visibility routing modes.

L1 routes updates to L2 for evidence-backed judgement. It does not assign
Fraud/Rarity/Noise verdicts and must not perform settlement actions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


L1RouterMode = Literal[
    "cosine_recheck",
    "dual_proxy_budgeted",
]


MODE_ALIASES: dict[str, L1RouterMode] = {
    "baseline": "cosine_recheck",
    "cosine": "cosine_recheck",
    "cosine_recheck": "cosine_recheck",
    "calibrated": "dual_proxy_budgeted",
    "dual_proxy": "dual_proxy_budgeted",
    "dual_proxy_budgeted": "dual_proxy_budgeted",
}


@dataclass(frozen=True)
class L1RouterConfig:
    mode: L1RouterMode = "cosine_recheck"

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

    # Routing budget
    queue_budget_ratio: float = 0.35
    random_recheck_ratio: float = 0.05
    min_recheck_prob: float = 0.02
    max_recheck_prob: float = 0.30

    # L1V4 dual validation-gradient proxy
    dual_main_weight: float = 0.15
    dual_corner_harm_weight: float = 0.25
    dual_corner_benefit_weight: float = 0.10
    theta_main_proxy: float = 0.02
    theta_corner_harm_proxy: float = 0.005

    # L1V4 non-audit routing actions
    quarantine_risk_threshold: float = 0.75
    low_weight_risk_threshold: float = 0.45
    safe_weight: float = 0.80
    low_weight: float = 0.20

    @property
    def uses_budget(self) -> bool:
        return self.mode == "dual_proxy_budgeted"

    @property
    def uses_norm(self) -> bool:
        return self.mode == "dual_proxy_budgeted" and self.use_norm_mad

    @property
    def uses_sign(self) -> bool:
        return self.mode == "dual_proxy_budgeted" and self.use_sign_score

    @property
    def uses_dual_proxy(self) -> bool:
        return self.mode == "dual_proxy_budgeted"


def normalize_l1_mode(raw: str | None) -> L1RouterMode:
    key = (raw or "cosine_recheck").strip().lower()
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
        os.getenv("L1_ROUTER_MODE", "cosine_recheck"),
        queue_budget_ratio=float(os.getenv("L1_QUEUE_BUDGET_RATIO", "0.35")),
        random_recheck_ratio=float(os.getenv("L1_RANDOM_RECHECK_RATIO", "0.05")),
    )
