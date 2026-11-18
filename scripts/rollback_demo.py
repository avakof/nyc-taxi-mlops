"""Prove full image-and-release recovery in an isolated Compose project."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from taxi_mlops.release import load_release, release_id

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {"zone_id": 237, "forecast_origin": "2020-02-20T15:00:00Z"}
FAULTY_APP = """from fastapi.responses import JSONResponse
from taxi_mlops.api import create_app


def create_broken_app():
    app = create_app()

    @app.middleware("http")
    async def inject_failure(request, call_next):
        if request.url.path == "/forecast":
            return JSONResponse({"detail": "Injected deployment failure"}, status_code=500)
        return await call_next(request)

    return app
"""


def command(args: list[str], **kwargs) -> str:
    result = subprocess.run(args, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )
    return result.stdout.strip()


def image_identity(image: str) -> str:
    return json.loads(command(["docker", "image", "inspect", image]))[0]["Id"]


def deploy(
    image: str,
    release_path: Path,
    project: str,
    port: int,
    *,
    root: Path = ROOT,
    probe_request: dict | None = None,
) -> dict:
    """Recreate the service from pinned local artifacts and verify its actual identity."""
    expected = load_release(release_path).manifest
    pinned = image_identity(image)
    environment = {
        **os.environ,
        "TAXI_DEPLOY_IMAGE": pinned,
        "TAXI_RELEASE_PATH": str(release_path.resolve()),
        "TAXI_API_PORT": str(port),
    }
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(root / "compose.yaml"),
        "-f",
        str(root / "compose.deploy.yaml"),
    ]
    command(
        compose
        + [
            "up",
            "--no-build",
            "--pull",
            "never",
            "--force-recreate",
            "--detach",
            "--wait",
            "--wait-timeout",
            "120",
        ],
        cwd=root,
        env=environment,
    )
    container_id = command(compose + ["ps", "-q", "api"], cwd=root, env=environment)
    container = json.loads(command(["docker", "inspect", container_id]))[0]
    if container["Config"]["Image"] != pinned:
        raise RuntimeError(
            "The running container does not reference the requested pinned image"
        )
    mount = next(
        value for value in container["Mounts"] if value["Destination"] == "/opt/release"
    )
    if mount["RW"]:
        raise RuntimeError("The running release mount is writable")
    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        identity = client.get("/model-info")
        identity.raise_for_status()
        info = identity.json()
        if any(
            info[key] != expected[key]
            for key in ("release_id", "model_name", "model_version", "run_id")
        ):
            raise RuntimeError(
                "The served release identity differs from the requested manifest"
            )
        probe_start = time.monotonic()
        forecast = client.post(
            "/forecast", json=REQUEST if probe_request is None else probe_request
        )
        probe_latency_ms = (time.monotonic() - probe_start) * 1000
    descriptor = container.get("ImageManifestDescriptor") or {}
    return {
        "pinned_image_id": pinned,
        "actual_image": container["Image"],
        "platform_manifest": descriptor.get("digest"),
        "container_id": container_id,
        "model_info": info,
        "health_status": health.status_code,
        "forecast_status": forecast.status_code,
        "probe_latency_ms": probe_latency_ms,
        "forecast": forecast.json(),
    }


def restore_deployment(
    record_path: Path, project: str, port: int, *, root: Path = ROOT
) -> dict:
    """Restore a retained image and package, then check identity and a known forecast."""
    record = json.loads(record_path.read_text())
    package = load_release(record["release_path"])
    if package.manifest["release_id"] != record["release_id"]:
        raise RuntimeError("Retained package differs from the deployment record")
    started = time.monotonic()
    restored = deploy(
        record["image_id"],
        Path(record["release_path"]),
        project,
        port,
        root=root,
        probe_request=record["forecast_request"],
    )
    if restored["forecast_status"] != 200:
        raise RuntimeError("Restored deployment failed its forecast probe")
    if not math.isclose(
        restored["forecast"]["expected_pickups"],
        record["expected_pickups"],
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Restored forecast differs from the known-good forecast")
    for key in ("actual_image", "platform_manifest"):
        if restored[key] != record[key]:
            raise RuntimeError(f"Restored deployment differs in {key}")
    restored["recovery_seconds_wall"] = time.monotonic() - started
    restored["recovery_within_120_seconds"] = restored["recovery_seconds_wall"] <= 120
    return restored


def run_demo(source: Path, image: str, port: int, *, root: Path = ROOT) -> dict:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"Port {port} is already occupied; choose --port with a free port"
            ) from error
    token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    project = "taxi-rollback-" + token.lower()
    retained = root / "models" / ("rollback-" + token)
    evidence = root / "reports" / "rollback" / token
    retained.mkdir(parents=True)
    evidence.mkdir(parents=True)
    good = retained / "good"
    bad = retained / "bad"
    load_release(source)
    shutil.copytree(source, good)
    shutil.copytree(source, bad)
    manifest_path = bad / "release.json"
    bad_manifest = json.loads(manifest_path.read_text())
    bad_manifest["incident_scenario"] = "injected-serving-code-failure"
    bad_manifest["release_id"] = release_id(bad_manifest)
    manifest_path.write_text(json.dumps(bad_manifest, indent=2) + "\n")
    pinned_good = image_identity(image)
    good_tag = "nyc-taxi-mlops:retained-" + token.lower()
    bad_tag = "nyc-taxi-mlops:fault-" + token.lower()
    command(["docker", "image", "tag", pinned_good, good_tag])
    with tempfile.TemporaryDirectory(prefix="taxi-fault-image-") as temporary:
        build = Path(temporary)
        (build / "faulty_app.py").write_text(FAULTY_APP)
        (build / "Dockerfile").write_text(
            f'FROM {good_tag}\nCOPY faulty_app.py /app/faulty_app.py\nCMD ["/app/.venv/bin/uvicorn", "faulty_app:create_broken_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]\n'
        )
        command(["docker", "build", "--tag", bad_tag, str(build)])
    pinned_bad = image_identity(bad_tag)
    if pinned_good == pinned_bad:
        raise RuntimeError("Fault injection did not create a distinct image")
    report = {
        "scenario": "isolated-bad-code-rollback",
        "project": project,
        "port": port,
        "forecast_origin_simulation": REQUEST["forecast_origin"],
        "retained_good_tag": good_tag,
        "retained_bad_tag": bad_tag,
        "good_release_path": str(good),
        "bad_release_path": str(bad),
        "timeline": [],
    }
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(root / "compose.yaml"),
        "-f",
        str(root / "compose.deploy.yaml"),
    ]

    def record(phase, details):
        report["timeline"].append(
            {"phase": phase, "wall_time_utc": datetime.now(UTC).isoformat(), **details}
        )
        (evidence / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        baseline = deploy(pinned_good, good, project, port, root=root)
        if baseline["forecast_status"] != 200:
            raise RuntimeError("Known-good release failed its forecast probe")
        record("healthy-baseline", baseline)
        snapshot = {
            "image_id": baseline["pinned_image_id"],
            "actual_image": baseline["actual_image"],
            "platform_manifest": baseline["platform_manifest"],
            "release_path": str(good),
            "release_id": baseline["model_info"]["release_id"],
            "expected_pickups": baseline["forecast"]["expected_pickups"],
            "forecast_request": REQUEST,
        }
        rollback_record = evidence / "known-good-deployment.json"
        rollback_record.write_text(json.dumps(snapshot, indent=2) + "\n")
        report["rollback_record"] = str(rollback_record)
        failing = deploy(pinned_bad, bad, project, port, root=root)
        if failing["health_status"] != 200 or failing["forecast_status"] != 500:
            raise RuntimeError(
                "Expected a healthy process with an injected forecast failure"
            )
        record("failure-detected", failing)
        recovery_start = time.monotonic()
        try:
            recovered = restore_deployment(rollback_record, project, port, root=root)
            duration = time.monotonic() - recovery_start
            if recovered["forecast_status"] != 200:
                raise RuntimeError("Rollback did not restore forecasts")
            if (
                recovered["forecast"]["expected_pickups"]
                != baseline["forecast"]["expected_pickups"]
            ):
                raise RuntimeError(
                    "Rollback prediction differs from the known-good prediction"
                )
            for key in (
                "pinned_image_id",
                "actual_image",
                "platform_manifest",
                "model_info",
            ):
                if recovered[key] != baseline[key]:
                    raise RuntimeError(f"Rollback did not restore {key}")
            record("recovered", {**recovered, "recovery_seconds_wall": duration})
        except Exception as error:
            record("recovery-failed", {"error": str(error)})
            raise
        report["result"] = "PASS"
        report["recovery_seconds_wall"] = duration
        report["unchanged_history_sha256"] = json.loads(
            (good / "release.json").read_text()
        )["files"]["history.parquet"]
        record(
            "verification-complete",
            {
                "note": "Data history was identical across releases; no observations were rewound"
            },
        )
        report["evidence_directory"] = str(evidence)
        (evidence / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        environment = {
            **os.environ,
            "TAXI_DEPLOY_IMAGE": pinned_good,
            "TAXI_RELEASE_PATH": str(good.resolve()),
            "TAXI_API_PORT": str(port),
        }
        logs = subprocess.run(
            compose + ["logs", "--no-color", "api"],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
        )
        (evidence / "final-container.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(
            compose + ["down", "--timeout", "10"],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "models/final-serving")
    parser.add_argument("--image", default="nyc-taxi-mlops:serving")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    report = run_demo(args.release.resolve(), args.image, args.port)
    print(
        json.dumps(
            {
                "result": report["result"],
                "evidence_directory": report["evidence_directory"],
                "recovery_seconds_wall": report["recovery_seconds_wall"],
                "phases": [
                    {
                        "phase": entry["phase"],
                        "health_status": entry.get("health_status"),
                        "forecast_status": entry.get("forecast_status"),
                    }
                    for entry in report["timeline"]
                    if "health_status" in entry
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
