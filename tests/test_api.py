"""HTTP behavior of the forecasting service."""

from fastapi.testclient import TestClient

from taxi_mlops.api import create_app


def test_health_is_unavailable_without_resources():
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Forecasting resources are not ready"}
