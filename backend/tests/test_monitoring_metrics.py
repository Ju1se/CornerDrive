from fastapi.testclient import TestClient

from l1_linear_defense.server import app as l1_app
from l4_settlement.dashboard_api import app as l4_app


def test_l1_metrics_returns_prometheus_text():
    response = TestClient(l1_app).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "l1_pending_gradients" in response.text
    assert "l1_batches_processed_total" in response.text


def test_l4_metrics_returns_prometheus_text():
    response = TestClient(l4_app).get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "l4_total_vehicles" in response.text
    assert "l4_settlement_requests_total" in response.text
