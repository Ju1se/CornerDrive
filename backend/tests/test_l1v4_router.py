import random

import numpy as np
import pytest

from l1_linear_defense.aggregation import filter_suspects
from l1_linear_defense.config import make_l1_router_config
from l1_linear_defense.scoring import rank_normalize


def test_rank_normalize_collapses_uninformative_signal() -> None:
    assert rank_normalize([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


def test_cosine_recheck_mode_preserves_default_router() -> None:
    gradients = [np.array([1.0, 1.0, 1.0]) for _ in range(4)]
    gradients.append(np.array([-10.0, -10.0, -10.0]))
    vehicle_ids = [f"0x{i:040x}" for i in range(5)]

    baseline = filter_suspects(
        gradients,
        vehicle_ids,
        threshold=0.3,
        recheck_probability=0.0,
        rng=random.Random(1),
    )
    explicit_cosine = filter_suspects(
        gradients,
        vehicle_ids,
        threshold=0.3,
        recheck_probability=0.0,
        rng=random.Random(1),
        router_config=make_l1_router_config("cosine_recheck"),
    )

    assert explicit_cosine.suspect_indices == baseline.suspect_indices
    assert explicit_cosine.clean_indices == baseline.clean_indices
    assert explicit_cosine.routing_reasons == baseline.routing_reasons
    assert explicit_cosine.l1_score_details == {}


def test_deprecated_router_modes_are_not_supported() -> None:
    for mode in ("norm_fixed", "norm_sign_fixed", "budgeted_legacy"):
        with pytest.raises(ValueError):
            make_l1_router_config(mode)


def test_v4_dual_proxy_routes_corner_harm_before_default_accept() -> None:
    gradients = [
        np.array([1.0, 0.0]),
        np.array([0.0, -1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.1]),
    ]
    vehicle_ids = [f"0x{i:040x}" for i in range(4)]
    config = make_l1_router_config(
        "dual_proxy_budgeted",
        cos_deviation_threshold=3.0,
        norm_mad_threshold=999.0,
        sign_threshold=2.0,
        queue_budget_ratio=0.25,
        random_recheck_ratio=0.0,
        theta_corner_harm_proxy=0.01,
        safe_weight=0.8,
        low_weight=0.2,
    )

    result = filter_suspects(
        gradients,
        vehicle_ids,
        threshold=3.0,
        recheck_probability=0.0,
        router_config=config,
        main_validation_gradient=np.array([1.0, 0.0]),
        corner_validation_gradient=np.array([0.0, 1.0]),
        learning_rate=1.0,
        theta_tol=0.1,
        theta_corner_harm=0.01,
        rng=random.Random(7),
    )

    assert result.suspect_indices == [1]
    assert result.route_actions[1] == "AUDIT"
    assert result.routing_reasons[1] == "audit:corner_harm_proxy"
    assert result.aggregation_weights[1] == 0.0
    assert result.l1_score_details[1]["pred_delta_corner"] > 0.0
