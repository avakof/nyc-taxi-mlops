"""Export and validate one immutable, self-contained historical replay release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
import mlflow
import pandas as pd
from dotenv import load_dotenv

from taxi_mlops.config import load_params
from taxi_mlops.release import (
    HISTORY_POLICY,
    load_release,
    prepare_replay_history,
    release_id,
    sha256_file,
)


def export_release(
    output: Path, tracking_uri: str, model_name: str, alias: str = "champion"
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {output}")
    try:
        response = httpx.get(
            tracking_uri.rstrip("/") + "/health", timeout=5, trust_env=False
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise SystemExit(
            "MLflow is unreachable. Start your configured tracking server and retry."
        ) from error
    params = load_params()
    source_manifest = json.loads(Path("data/manifest.json").read_text())
    for entry in source_manifest["files"]:
        if sha256_file(Path(entry["file"])) != entry["sha256"]:
            raise ValueError(f"Source data checksum mismatch: {entry['file']}")
    zones = [
        row["zone_id"]
        for row in json.loads(Path("config/selected_zones.json").read_text())
    ]
    history = prepare_replay_history(
        pd.read_parquet("data/processed/zone_hour_counts.parquet"),
        zones,
        accept_complete_snapshot=True,
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(version.run_id)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".taxi-release-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        bundle.mkdir()
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{version.version}",
            dst_path=str(root / "download"),
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
        shutil.copytree(downloaded, bundle / "model")
        metadata = mlflow.models.Model.load(str(bundle / "model" / "MLmodel"))
        history.to_parquet(bundle / "history.parquet", index=False)
        (bundle / "features.json").write_text(
            json.dumps({"features": params["features"]}, indent=2) + "\n"
        )
        (bundle / "zones.json").write_text(json.dumps(zones) + "\n")
        manifest = {
            "format_version": 1,
            "model_name": model_name,
            "model_version": str(version.version),
            "run_id": version.run_id,
            "loader": "sklearn" if "sklearn" in metadata.flavors else "pyfunc",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "model_source_git_commit": run.data.tags.get("git_commit"),
            "validation_metrics": dict(run.data.metrics),
            "model_source_data_manifest_sha256": run.data.tags.get(
                "data_manifest_sha256"
            ),
            "source_data_manifest_sha256": sha256_file(Path("data/manifest.json")),
            "source_history_sha256": sha256_file(
                Path("data/processed/zone_hour_counts.parquet")
            ),
            "environment_lock_sha256": sha256_file(Path("uv.lock")),
            "serving_source_sha256": {
                str(p.relative_to(Path(__file__).resolve().parents[1])): sha256_file(p)
                for p in sorted(
                    (Path(__file__).resolve().parents[1] / "src" / "taxi_mlops").glob(
                        "*.py"
                    )
                )
            },
            "history_policy": HISTORY_POLICY,
            "files": {
                str(path.relative_to(bundle)): sha256_file(path)
                for path in sorted(bundle.rglob("*"))
                if path.is_file()
            },
        }
        manifest["release_id"] = release_id(manifest)
        (bundle / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_release(bundle)
        bundle.rename(output)
    return manifest


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/final-serving"))
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
    )
    parser.add_argument("--model-name", default="nyc-taxi-demand-forecaster")
    parser.add_argument("--alias", default="champion")
    parser.add_argument(
        "--accept-complete-snapshot",
        action="store_true",
        required=True,
        help="Explicitly accept complete monthly snapshots with simulated availability at hour closure",
    )
    args = parser.parse_args()
    manifest = export_release(
        args.output, args.tracking_uri, args.model_name, args.alias
    )
    print(
        json.dumps(
            {
                "directory": str(args.output),
                "release_id": manifest["release_id"],
                "model_version": manifest["model_version"],
                "run_id": manifest["run_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
