"""Export regressions: explicit connections, pinned versions and source identity."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from export_release import export_release

from taxi_mlops.release import load_release, prepare_replay_history

pytestmark = pytest.mark.usefixtures("isolated_mlflow")


def test_export_overrides_conflicting_global_connections(
    serving_project, registered_test_models, tmp_path, monkeypatch
):
    monkeypatch.chdir(serving_project)
    mlflow.set_tracking_uri("http://127.0.0.1:1")
    mlflow.set_registry_uri("http://127.0.0.1:1")
    model = registered_test_models
    output = tmp_path / "release"
    manifest = export_release(output, model["uri"], model["name"])
    assert mlflow.get_tracking_uri() == model["uri"]
    assert mlflow.get_registry_uri() == model["uri"]
    assert manifest["model_version"] == "1"
    assert load_release(output).manifest["run_id"] == model["versions"][0].run_id


def test_alias_is_resolved_once_and_numeric_version_is_exported(
    serving_project, registered_test_models, tmp_path, monkeypatch
):
    monkeypatch.chdir(serving_project)
    model = registered_test_models
    original = mlflow.MlflowClient.get_model_version_by_alias
    client = mlflow.MlflowClient(tracking_uri=model["uri"], registry_uri=model["uri"])
    calls = []

    def resolve_then_move(self, name, alias):
        version = original(self, name, alias)
        calls.append(version.version)
        client.set_registered_model_alias(name, alias, model["versions"][1].version)
        return version

    monkeypatch.setattr(
        mlflow.MlflowClient, "get_model_version_by_alias", resolve_then_move
    )
    try:
        manifest = export_release(tmp_path / "release", model["uri"], model["name"])
        assert calls == ["1"]
        assert manifest["model_version"] == "1"
        assert manifest["run_id"] == model["versions"][0].run_id
    finally:
        client.set_registered_model_alias(
            model["name"], "champion", model["versions"][0].version
        )


def test_existing_release_is_never_overwritten(serving_release, registered_test_models):
    sentinel = (serving_release / "release.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_release(
            serving_release,
            registered_test_models["uri"],
            registered_test_models["name"],
        )
    assert (serving_release / "release.json").read_bytes() == sentinel


def test_changed_source_bytes_rejected(
    serving_project, registered_test_models, tmp_path, monkeypatch
):
    monkeypatch.chdir(serving_project)
    source = serving_project / "data/raw/synthetic.csv"
    previous = source.read_bytes()
    source.write_bytes(previous + b"changed")
    try:
        with pytest.raises(ValueError, match="Source data checksum mismatch"):
            export_release(
                tmp_path / "release",
                registered_test_models["uri"],
                registered_test_models["name"],
            )
    finally:
        source.write_bytes(previous)
    assert not (tmp_path / "release").exists()


def test_snapshot_completeness_requires_explicit_acceptance():
    hours = pd.date_range("2020-01-01T00:00:00Z", periods=2, freq="h")
    raw = pd.DataFrame({"zone_id": [237, 237], "hour_utc": hours, "pickups": [0, 2]})
    with pytest.raises(ValueError, match="explicit replay acceptance"):
        prepare_replay_history(raw, [237])


def test_explicit_incomplete_metadata_is_preserved():
    hours = pd.date_range("2020-01-01T00:00:00Z", periods=2, freq="h")
    history = pd.DataFrame(
        {
            "zone_id": [237, 237],
            "hour_utc": hours,
            "pickups": [0.0, np.nan],
            "is_complete": [True, False],
            "available_at": hours + pd.Timedelta(hours=1),
        }
    )
    result = prepare_replay_history(history, [237], accept_complete_snapshot=True)
    assert not result.is_complete.iloc[1]
    assert pd.isna(result.pickups.iloc[1])
    assert result.pickups.iloc[0] == 0


def test_native_sklearn_release_serves(
    serving_project, registered_test_models, tmp_path, monkeypatch
):
    import mlflow.sklearn
    from fastapi.testclient import TestClient
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    from taxi_mlops.api import create_app
    from taxi_mlops.config import load_params
    from taxi_mlops.features import build_feature_table, feature_columns

    with threadpool_limits(limits=1):
        monkeypatch.chdir(serving_project)
        uri = registered_test_models["uri"]
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        params = load_params()
        history = pd.read_parquet("data/processed/zone_hour_counts.parquet")
        features = build_feature_table(history, params)
        estimator = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=3,
            max_leaf_nodes=3,
            min_samples_leaf=2,
            early_stopping=False,
        )
        estimator.fit(features[feature_columns(params)], features.target)
        name = "test-native-hgb"
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(
                estimator,
                name="model",
                pip_requirements=[f"mlflow=={mlflow.__version__}"],
                skops_trusted_types=[
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                    "functools.partial",
                    "sklearn.utils.validation.check_array",
                ],
            )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
        mlflow.MlflowClient().set_registered_model_alias(
            name, "champion", version.version
        )
        output = tmp_path / "native-release"
        manifest = export_release(output, uri, name)
        assert manifest["loader"] == "sklearn"
        mlflow.set_tracking_uri("http://127.0.0.1:1")
        mlflow.set_registry_uri("http://127.0.0.1:1")
        with TestClient(create_app(release_dir=output)) as client:
            assert client.get("/health").status_code == 200
            response = client.post(
                "/forecast",
                json={"zone_id": 237, "forecast_origin": "2020-03-08T07:00:00Z"},
            )
            assert response.status_code == 200
            assert np.isfinite(response.json()["expected_pickups"])
            assert response.json()["expected_pickups"] >= 0
