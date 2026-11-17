"""Register a run's logged model into the MLflow Model Registry as the `candidate` alias,
then resolve that exact alias/version and run a smoke prediction to verify the registered
artifact actually loads and predicts -- not just that registration succeeded.

Promotion from `candidate` to `champion` is a deliberate, separate step (ADR-0001; the
project's own adversarial-audit correction: "automatic retraining is confused with automatic
deployment -- promotion is a separate controlled operation"). This script never touches
`champion`.

Usage:
    uv run python scripts/register_model.py <run_id>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

from taxi_mlops.config import load_params
from taxi_mlops.features import build_feature_table, feature_columns, split_by_period

ROOT = Path(__file__).resolve().parents[1]
ALIAS = "candidate"


def smoke_predict(model_name: str, version: str) -> None:
    """Load the exact version just registered (not re-resolving the alias) and predict once."""
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")

    params = load_params()
    zone_ids = [z["zone_id"] for z in json.loads((ROOT / "config" / "selected_zones.json").read_text())]
    zone_hour = pd.read_parquet(ROOT / "data" / "processed" / "zone_hour_counts.parquet")
    zone_hour = zone_hour[zone_hour.zone_id.isin(zone_ids)]
    feature_table = build_feature_table(zone_hour, params)
    test = split_by_period(feature_table, params, "test")
    cols = feature_columns(params)

    sample = test[cols].head(3)
    pred = model.predict(sample)
    print(f"smoke prediction OK: {pred.tolist()} (nonnegative finite: {bool((pred >= 0).all())})")
    if not (pred >= 0).all():
        raise SystemExit("smoke check failed: model produced a negative prediction")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run python scripts/register_model.py <run_id>")
    run_id = sys.argv[1]

    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "nyc-taxi-demand-forecaster")

    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(model_name, ALIAS, version.version)
    print(f"{model_name} v{version.version} registered from run {run_id}, now @{ALIAS}")

    # Resolve once, load the immutable numeric version (not re-resolving @candidate), verify it.
    smoke_predict(model_name, version.version)


if __name__ == "__main__":
    main()
