import random

import numpy as np

from l1_linear_defense.aggregation import filter_suspects
from l1_linear_defense.config import make_l1_router_config
from l1_linear_defense.scoring import rank_normalize


def test_rank_normalize_collapses_uninformative_signal() -> None:
    assert rank_normalize([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


def test_v25_mode_preserves_default_cosine_router() -> None:
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
    explicit_m0 = filter_suspects(
        gradients,
        vehicle_ids,
        threshold=0.3,
        recheck_probability=0.0,
        rng=random.Random(1),
        router_config=make_l1_router_config("v25_cosine_fixed"),
    )

    assert explicit_m0.suspect_indices == baseline.suspect_indices
    assert explicit_m0.clean_indices == baseline.clean_indices
    assert explicit_m0.routing_reasons == baseline.routing_reasons
    assert explicit_m0.l1_score_details == {}


def test_budgeted_router_uses_top_risk_when_thresholds_do_not_fire() -> None:
    gradients = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.9, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([3.0, 3.0, 3.0]),
        np.array([1.0, -1.0, 1.0]),
    ]
    vehicle_ids = [f"0x{i:040x}" for i in range(5)]
    config = make_l1_router_config(
        "m3",
        cos_deviation_threshold=3.0,
        norm_mad_threshold=999.0,
        sign_threshold=2.0,
        queue_budget_ratio=0.40,
        random_recheck_ratio=0.0,
    )

    result = filter_suspects(
        gradients,
        vehicle_ids,
        threshold=3.0,
        recheck_probability=0.0,
        router_config=config,
        rng=random.Random(4),
    )

    assert len(result.suspect_indices) == 2
    assert set(result.routing_reasons[idx] for idx in result.suspect_indices) == {
        "risk_topB"
    }
    assert result.router_mode == "v3_m3_budgeted"
    assert result.l1_score_details


def test_full_router_can_route_low_reputation_client_without_verdict() -> None:
    gradients = [np.array([1.0, 1.0, 1.0]) for _ in range(3)]
    vehicle_ids = [f"0x{i:040x}" for i in range(3)]
    client_states = {
        vehicle_ids[1]: {
            "reputation": 0.2,
            "recent_fraud_count": 1,
            "last_audit_round": 3,
        },
        vehicle_ids[0]: {"reputation": 1.0, "last_audit_round": 3},
        vehicle_ids[2]: {"reputation": 1.0, "last_audit_round": 3},
    }
    config = make_l1_router_config(
        "m4",
        cos_deviation_threshold=3.0,
        norm_mad_threshold=999.0,
        sign_threshold=2.0,
        queue_budget_ratio=0.20,
        random_recheck_ratio=0.0,
    )

    result = filter_suspects(
        gradients,
        vehicle_ids,
        threshold=3.0,
        recheck_probability=0.0,
        router_config=config,
        client_states=client_states,
        current_round=4,
        rng=random.Random(5),
    )

    assert result.suspect_indices == [1]
    assert result.routing_reasons[1] == "risk_topB"
    assert "verdict" not in result.l1_score_details[1]
