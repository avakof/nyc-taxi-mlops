"""Test helpers and isolated real MLflow/release fixtures. No TLC downloads required."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.models import infer_signature

from taxi_mlops.api import create_app
from taxi_mlops.release import release_id, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def isolated_mlflow(monkeypatch):
    """Restore process-level MLflow settings after each serving test."""
    previous = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous[0])
        mlflow.set_registry_uri(previous[1])


@pytest.fixture(scope="session")
def test_mlflow_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("isolated-mlflow")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uri = f"http://127.0.0.1:{port}"
    environment = {
        **os.environ,
        "MLFLOW_TRACKING_URI": uri,
        "MLFLOW_REGISTRY_URI": uri,
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    log_path = directory / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
                "--backend-store-uri",
                f"sqlite:///{directory / 'mlflow.db'}",
                "--artifacts-destination",
                str(directory / "artifacts"),
            ],
            cwd=directory,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail(
                            "Test MLflow exited: " + log_path.read_text()[-4000:]
                        )
                    try:
                        if client.get(uri + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(
                        "Test MLflow startup timed out: " + log_path.read_text()[-4000:]
                    )
            yield uri
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.fixture(scope="session")
def registered_test_models(test_mlflow_server):
    """Two real serialized models and Registry versions on the isolated server."""

    class SeasonalBaseline(mlflow.pyfunc.PythonModel):
        def __init__(self, multiplier):
            self.multiplier = multiplier

        def predict(self, context, model_input, params=None):
            return model_input["lag_168"].to_numpy() * self.multiplier

    previous = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    mlflow.set_tracking_uri(test_mlflow_server)
    mlflow.set_registry_uri(test_mlflow_server)
    try:
        mlflow.set_experiment("serving-integration-tests")
        name = "test-taxi-demand"
        versions = []
        example = pd.DataFrame({"lag_168": [1.0, 2.0]})
        for multiplier in (1, 2):
            with mlflow.start_run() as run:
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=SeasonalBaseline(multiplier),
                    signature=infer_signature(example, np.array([1.0, 2.0])),
                    input_example=example,
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}",
                        f"pandas=={pd.__version__}",
                    ],
                )
            versions.append(
                mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
            )
        client = mlflow.MlflowClient(
            tracking_uri=test_mlflow_server, registry_uri=test_mlflow_server
        )
        client.set_registered_model_alias(name, "champion", versions[0].version)
        client.set_registered_model_alias(name, "candidate", versions[1].version)
        return {"uri": test_mlflow_server, "name": name, "versions": versions}
    finally:
        mlflow.set_tracking_uri(previous[0])
        mlflow.set_registry_uri(previous[1])


@pytest.fixture(scope="session")
def serving_project(tmp_path_factory):
    """A tiny synthetic source snapshot and Git repo, independent of user data."""
    directory = tmp_path_factory.mktemp("serving-project")
    for name in ["data/raw", "data/processed", "config"]:
        (directory / name).mkdir(parents=True)
    source = directory / "data/raw/synthetic.csv"
    source.write_text("synthetic pickup source for integration tests\n")
    (directory / "data/manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"file": "data/raw/synthetic.csv", "sha256": sha256_file(source)}
                ]
            }
        )
    )
    (directory / "params.yaml").write_text((ROOT / "params.yaml").read_text())
    (directory / "uv.lock").write_text((ROOT / "uv.lock").read_text())
    (directory / "config/selected_zones.json").write_text(
        json.dumps([{"zone_id": 237}, {"zone_id": 161}])
    )
    hours = pd.date_range("2020-03-01T00:00:00Z", periods=240, freq="h")
    history = pd.concat(
        [
            pd.DataFrame(
                {
                    "zone_id": zone,
                    "hour_utc": hours,
                    "pickups": np.arange(1, 241) + extra,
                }
            )
            for zone, extra in [(237, 0), (161, 1000)]
        ],
        ignore_index=True,
    )
    history.to_parquet(
        directory / "data/processed/zone_hour_counts.parquet", index=False
    )
    subprocess.run(["git", "init", "-q", str(directory)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(directory),
            "-c",
            "user.name=Integration Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "Synthetic test fixture",
        ],
        check=True,
    )
    return directory


@pytest.fixture(scope="session")
def serving_release(serving_project, registered_test_models, tmp_path_factory):
    from export_release import export_release

    output = tmp_path_factory.mktemp("release-output") / "release"
    previous_directory = Path.cwd()
    previous = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    try:
        os.chdir(serving_project)
        export_release(
            output, registered_test_models["uri"], registered_test_models["name"]
        )
    finally:
        os.chdir(previous_directory)
        mlflow.set_tracking_uri(previous[0])
        mlflow.set_registry_uri(previous[1])
    return output


@pytest.fixture
def api_client(serving_release, isolated_mlflow):
    """Exercise real startup and inference while the runtime Registry is unreachable."""
    mlflow.set_tracking_uri("http://127.0.0.1:1")
    mlflow.set_registry_uri("http://127.0.0.1:1")
    app = create_app(release_dir=serving_release)
    with TestClient(app) as client:
        yield client
    assert app.state.model is None
    assert app.state.release is None
    assert not app.state.ready


@pytest.fixture
def release_copy(serving_release, tmp_path):
    import shutil

    path = tmp_path / "release"
    shutil.copytree(serving_release, path)
    return path


@pytest.fixture
def reseal_release():
    """Update synthetic fixture hashes after intentional valid data modifications."""

    def reseal(path):
        manifest_path = path / "release.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"] = {
            str(p.relative_to(path)): sha256_file(p)
            for p in sorted(path.rglob("*"))
            if p.is_file() and p != manifest_path
        }
        manifest["release_id"] = release_id(manifest)
        manifest_path.write_text(json.dumps(manifest))

    return reseal
