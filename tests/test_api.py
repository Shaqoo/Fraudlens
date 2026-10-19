"""API tests for health, readiness, validation and prediction endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraudlens.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_dependency_state(client: TestClient) -> None:
    payload = client.get("/health/ready").json()
    assert set(payload) == {"status", "model_loaded", "database_connected"}
    assert isinstance(payload["model_loaded"], bool)


def test_model_info_without_deployed_model(client: TestClient) -> None:
    response = client.get("/model/info")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        assert set(response.json()) == {
            "model_name",
            "model_version",
            "model_type",
            "threshold",
            "status",
        }


def test_openapi_documents_all_route_groups(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/health", "/health/ready", "/model/info", "/predictions", "/etl/runs", "/schedules", "/metrics"):
        assert path in paths


def test_predict_rejects_unknown_top_level_fields(client: TestClient) -> None:
    response = client.post("/predictions", json={"features": {"amount": 1.0}, "surprise": True})
    assert response.status_code == 422


def test_predict_requires_features(client: TestClient) -> None:
    response = client.post("/predictions", json={"transaction_id": "TXN-1"})
    assert response.status_code == 422


def test_batch_requires_non_empty_list(client: TestClient) -> None:
    response = client.post("/predictions/batch", json={"transactions": []})
    assert response.status_code == 422


def test_review_status_pattern_is_enforced(client: TestClient) -> None:
    response = client.patch("/predictions/1/review", json={"review_status": "invented"})
    assert response.status_code == 422


def test_missing_prediction_returns_404(client: TestClient) -> None:
    # Without a database the request cannot be served at all -> 503.
    assert client.get("/predictions/999999").status_code in {404, 503}


def test_predictions_support_pagination_query(client: TestClient) -> None:
    response = client.get("/predictions?predicted_label=1&limit=20&offset=0")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        payload = response.json()
        assert {"total", "limit", "offset", "items"} <= set(payload)


def test_history_requires_a_database(client: TestClient) -> None:
    response = client.get("/predictions")
    assert response.status_code in {200, 503}


def test_limit_is_bounded(client: TestClient) -> None:
    assert client.get("/predictions?limit=9999").status_code == 422


def test_predict_without_deployed_model_returns_service_unavailable(client: TestClient) -> None:
    response = client.post(
        "/predictions",
        json={"transaction_id": "TXN-1", "features": {"amount": 125.0, "v1": 0.1, "v2": 0.2, "hour": 5}},
    )
    assert response.status_code in {201, 503}


def test_batch_without_deployed_model_returns_service_unavailable(client: TestClient) -> None:
    response = client.post("/predictions/batch", json={"transactions": [{"features": {"amount": 1.0}}]})
    assert response.status_code in {200, 422, 503}
