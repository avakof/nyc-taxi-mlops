"""Real model, hashed release, lifespan and HTTP contract checks."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from taxi_mlops.api import create_app
from taxi_mlops.release import forecast_features

pytestmark = pytest.mark.usefixtures("isolated_mlflow")
REQUEST = {"zone_id": 237, "forecast_origin": "2020-03-08T07:00:00Z"}


def test_ready_and_model_identity(api_client):
    assert api_client.get("/health").json() == {"status": "ready"}
    response = api_client.get("/model-info")
    assert response.status_code == 200
    identity = response.json()
    assert identity["model_name"] == "test-taxi-demand"
    assert identity["model_version"] == "1"
    assert identity["run_id"]
    assert identity["release_id"].startswith("sha256:")


def test_forecast_uses_elapsed_week_and_next_hour(api_client):
    response = api_client.post("/forecast", json=REQUEST)
    assert response.status_code == 200
    prediction = response.json()
    assert prediction["expected_pickups"] == 8.0
    assert prediction["target_start"] == "2020-03-08T07:00:00Z"
    assert prediction["target_end"] == "2020-03-08T08:00:00Z"
    assert (
        prediction["release_id"] == api_client.get("/model-info").json()["release_id"]
    )
    assert prediction["request_id"]


@pytest.mark.parametrize(
    "changes",
    [
        {"zone_id": 999},
        {"zone_id": 237.0},
        {"zone_id": True},
        {"zone_id": -1},
        {"forecast_origin": 123},
        {"forecast_origin": "2020-03-08T07:00:00"},
        {"forecast_origin": "2020-03-08T07:30:00Z"},
        {"forecast_origin": "not-a-date"},
        {"unexpected": 1},
    ],
)
def test_invalid_requests_return_422(api_client, changes):
    response = api_client.post("/forecast", json={**REQUEST, **changes})
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["zone_id", "forecast_origin"])
def test_missing_request_fields_return_422(api_client, field):
    body = {key: value for key, value in REQUEST.items() if key != field}
    assert api_client.post("/forecast", json=body).status_code == 422


@pytest.mark.parametrize(
    "change", ["missing-hour", "incomplete", "late", "missing-count"]
)
def test_unavailable_observation_returns_503_without_zero_filling(
    release_copy, reseal_release, change
):
    path = release_copy / "history.parquet"
    history = pd.read_parquet(path)
    row = history.index[
        (history.zone_id == 237)
        & (
            history.hour_utc
            == pd.Timestamp(REQUEST["forecast_origin"]) - pd.Timedelta(hours=5)
        )
    ][0]
    if change == "missing-hour":
        history = history.drop(index=row)
    elif change == "incomplete":
        history.loc[row, "is_complete"] = False
    elif change == "late":
        history.loc[row, "available_at"] = pd.Timestamp(
            REQUEST["forecast_origin"]
        ) + pd.Timedelta(hours=1)
    else:
        history["pickups"] = history.pickups.astype(float)
        history.loc[row, "pickups"] = np.nan
        history.loc[row, "is_complete"] = False
    history.to_parquet(path, index=False)
    reseal_release(release_copy)
    with TestClient(create_app(release_dir=release_copy)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/forecast", json=REQUEST).status_code == 503


def test_complete_zero_is_a_valid_observation(release_copy, reseal_release):
    path = release_copy / "history.parquet"
    history = pd.read_parquet(path)
    row = (history.zone_id == 237) & (
        history.hour_utc
        == pd.Timestamp(REQUEST["forecast_origin"]) - pd.Timedelta(hours=168)
    )
    history.loc[row, "pickups"] = 0
    history.to_parquet(path, index=False)
    reseal_release(release_copy)
    with TestClient(create_app(release_dir=release_copy)) as client:
        response = client.post("/forecast", json=REQUEST)
        assert response.status_code == 200
        assert response.json()["expected_pickups"] == 0.0


def test_future_counts_cannot_change_current_forecast(api_client):
    bundle = api_client.app.state.release
    before = api_client.post("/forecast", json=REQUEST).json()["expected_pickups"]
    bundle.history.loc[
        bundle.history.hour_utc >= pd.Timestamp(REQUEST["forecast_origin"]), "pickups"
    ] = 999999
    after = api_client.post("/forecast", json=REQUEST).json()["expected_pickups"]
    assert before == after == 8.0


def test_dst_calendar_and_elapsed_lags(api_client):
    bundle = api_client.app.state.release
    features = forecast_features(
        bundle.history,
        bundle.params,
        bundle.zones,
        237,
        pd.Timestamp(REQUEST["forecast_origin"]),
    )
    assert features.hour_of_day.iloc[0] == 3
    assert features.lag_168.iloc[0] == 8
    assert features.lag_1.iloc[0] == 175


@pytest.mark.parametrize(
    "kind",
    [
        "changed-file",
        "extra-file",
        "missing-file",
        "wrong-run",
        "duplicate-hour",
        "negative-count",
    ],
)
def test_invalid_release_is_not_ready(release_copy, reseal_release, kind):
    if kind == "changed-file":
        with (release_copy / "features.json").open("a") as stream:
            stream.write(" ")
    elif kind == "extra-file":
        (release_copy / "unexpected.txt").write_text("not in manifest")
    elif kind == "missing-file":
        (release_copy / "zones.json").unlink()
    elif kind == "wrong-run":
        p = release_copy / "release.json"
        manifest = json.loads(p.read_text())
        manifest["run_id"] = "wrong-source-run"
        p.write_text(json.dumps(manifest))
        reseal_release(release_copy)
    else:
        p = release_copy / "history.parquet"
        history = pd.read_parquet(p)
        if kind == "duplicate-hour":
            history = pd.concat([history, history.iloc[[0]]], ignore_index=True)
        else:
            history.loc[0, "pickups"] = -1
        history.to_parquet(p, index=False)
        reseal_release(release_copy)
    with TestClient(create_app(release_dir=release_copy)) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/model-info").status_code == 503
        assert client.post("/forecast", json=REQUEST).status_code == 503


@pytest.mark.parametrize(
    "output", [[float("inf")], [float("nan")], [-1], [1, 2], [True], [1 + 2j]]
)
def test_invalid_model_output_returns_500(api_client, output):
    class BrokenModel:
        def predict(self, inputs):
            return output

    api_client.app.state.release.model = BrokenModel()
    response = api_client.post("/forecast", json=REQUEST)
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed"}


def test_unknown_history_period_returns_503(api_client):
    response = api_client.post(
        "/forecast", json={**REQUEST, "forecast_origin": "2020-01-01T00:00:00Z"}
    )
    assert response.status_code == 503


def test_equivalent_timezone_origin_is_normalized_to_utc(api_client):
    response = api_client.post(
        "/forecast", json={**REQUEST, "forecast_origin": "2020-03-08T03:00:00-04:00"}
    )
    assert response.status_code == 200
    assert response.json()["target_start"] == REQUEST["forecast_origin"]
    assert response.json()["expected_pickups"] == 8.0
