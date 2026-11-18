"""Reproduce the pinned baseline service from public source files in a clean checkout."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import mlflow
from mlflow.exceptions import MlflowException

from taxi_mlops.release import load_release, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DVC_FILES = ["data/raw.dvc", "data/external.dvc", "data/processed.dvc"]


def run(args: list[str], *, capture: bool = False, environment: dict | None = None):
    return subprocess.run(
        args, cwd=ROOT, env=environment, text=True, capture_output=capture, check=True
    )


def clean_checkout() -> None:
    if run(["git", "status", "--porcelain"], capture=True).stdout.strip():
        raise SystemExit(
            "Commit project changes before building a final release. Generated files belong in ignored directories."
        )


def status_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from status_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from status_values(child)


def bootstrap(output: Path) -> dict:
    clean_checkout()
    commit = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    if output.exists():
        previous = load_release(output).manifest
        if previous["git_commit"] != commit or previous["git_dirty"]:
            raise SystemExit(
                "Existing output belongs to another build. Retain it and choose a new --output directory."
            )
        return {
            "result": "ALREADY_PREPARED",
            "directory": str(output),
            "release_id": previous["release_id"],
        }
    print("1/4 Downloading and verifying source snapshots", flush=True)
    run([sys.executable, str(ROOT / "scripts/download_data.py")])
    print("2/4 Rebuilding hourly counts and verifying DVC snapshots", flush=True)
    run([sys.executable, str(ROOT / "scripts/aggregate_and_split.py")])
    before = {name: (ROOT / name).read_bytes() for name in DVC_FILES}
    status = json.loads(run(["dvc", "status", "--json"], capture=True).stdout)
    if set(status_values(status)) - {"not in cache"}:
        raise SystemExit(
            "Regenerated data differs from the committed DVC snapshot. Do not accept new hashes silently; inspect dvc status."
        )
    run(["dvc", "commit", "--no-relink", *DVC_FILES])
    if any((ROOT / name).read_bytes() != content for name, content in before.items()):
        raise SystemExit(
            "DVC metadata changed unexpectedly; inspect before continuing."
        )
    clean_checkout()
    state = ROOT / ".runtime/mlflow"
    state.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uri = f"http://127.0.0.1:{port}"
    environment = {
        **os.environ,
        "MLFLOW_TRACKING_URI": uri,
        "MLFLOW_REGISTRY_URI": uri,
        "MLFLOW_EXPERIMENT_NAME": "nyc-taxi-demand",
        "MLFLOW_MODEL_NAME": "nyc-taxi-demand-forecaster",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    log_path = state / "bootstrap-server.log"
    print("3/4 Preparing the persistent local MLflow Registry", flush=True)
    with log_path.open("w") as log:
        server = subprocess.Popen(
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
                f"sqlite:///{state / 'mlflow.db'}",
                "--artifacts-destination",
                str(state / "artifacts"),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(
                            "Bootstrap MLflow exited; inspect " + str(log_path)
                        )
                    try:
                        if client.get(uri + "/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError(
                        "Bootstrap MLflow startup timed out; inspect " + str(log_path)
                    )
            mlflow.set_tracking_uri(uri)
            mlflow.set_registry_uri(uri)
            client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
            name = "nyc-taxi-demand-forecaster"
            try:
                client.get_model_version_by_alias(name, "champion")
            except MlflowException as error:
                if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/register_baseline_champion.py"),
                    ],
                    environment=environment,
                )
            print("4/4 Exporting and validating the pinned release", flush=True)
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/export_release.py"),
                    "--tracking-uri",
                    uri,
                    "--output",
                    str(output),
                    "--accept-complete-snapshot",
                ],
                environment=environment,
            )
            manifest = load_release(output).manifest
            model_run = client.get_run(manifest["run_id"])
            summary = {
                "result": "PASS",
                "directory": str(output),
                "release_id": manifest["release_id"],
                "model_version": manifest["model_version"],
                "run_id": manifest["run_id"],
                "git_commit": manifest["git_commit"],
                "git_dirty": manifest["git_dirty"],
                "validation_metrics": model_run.data.metrics,
                "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
                "registry_database": str(state / "mlflow.db"),
            }
            clean_checkout()
            return summary
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/final-serving")
    parser.add_argument(
        "--accept-complete-snapshot",
        action="store_true",
        required=True,
        help="Accept the documented historical replay completeness and hourly availability assumption",
    )
    args = parser.parse_args()
    summary = bootstrap(args.output.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
