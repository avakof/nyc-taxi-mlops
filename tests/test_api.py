"""HTTP behavior of the forecasting service."""

from fastapi.testclient import TestClient

from taxi_mlops.api import create_app


def test_health_is_unavailable_without_resources(tmp_path):
    missing_model = tmp_path / "missing-model"

    with TestClient(create_app(release_dir=missing_model)) as client:
        response = client.get("/health")
        assert client.app.state.model is None

    assert response.status_code == 503
    assert response.json() == {"detail": "Forecasting resources are not ready"}
