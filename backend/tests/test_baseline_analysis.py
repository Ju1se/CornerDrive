import pytest

from policy_agent.analysis.baselines import build_baseline_analysis


@pytest.mark.asyncio
async def test_baseline_analysis_includes_fedavg():
    payload = await build_baseline_analysis(None, rounds=4)

    assert payload["classification_rounds"] == 4
    assert [entry["id"] for entry in payload["baselines"]] == [
        "fedavg",
        "l1_only",
        "static_l2",
        "adaptive",
    ]

    for baseline in payload["baselines"]:
        assert len(baseline["rounds"]) == 4
        assert 0.0 <= baseline["summary"]["main_accuracy_avg"] <= 1.0
        assert 0.0 <= baseline["summary"]["corner_accuracy_avg"] <= 1.0
        assert 0.0 <= baseline["summary"]["false_slash_estimate_avg"] <= 1.0
        assert 0.0 <= baseline["summary"]["rarity_retention_rate_avg"] <= 1.0


@pytest.mark.asyncio
async def test_fedavg_accepts_all_rarity_without_false_slash():
    payload = await build_baseline_analysis(None, rounds=4)
    fedavg = next(entry for entry in payload["baselines"] if entry["id"] == "fedavg")

    for round_item in fedavg["rounds"]:
        assert round_item["rarity_retention_rate"] == pytest.approx(1.0)
        assert round_item["false_slash_estimate"] == pytest.approx(0.0)
