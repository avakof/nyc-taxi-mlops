"""Run the candidate promotion gates (ADR-0001 Sec. 8) for real: champion vs. candidate,
loaded from the Registry by alias, evaluated fresh on the frozen test split. Records a
pass/fail decision to config/ -- never moves the `champion` alias itself. Promotion is a
separate, deliberate action (ADR-0005), not something this script performs.

Usage:
    uv run python scripts/evaluate_promotion.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

from taxi_mlops.config import load_params
from taxi_mlops.features import build_feature_table, feature_columns, split_by_period

ROOT = Path(__file__).resolve().parents[1]
MIN_MAE_IMPROVEMENT = 0.05
MAX_ZONE_REGRESSION = 0.10


def load_selected_zone_ids() -> list[int]:
    zones = json.loads((ROOT / "config" / "selected_zones.json").read_text())
    return [z["zone_id"] for z in zones]


def mae_wape(actual: pd.Series, predicted) -> tuple[float, float | None]:
    error = actual.to_numpy() - np.asarray(predicted)
    mae = float(np.abs(error).mean())
    total = float(actual.sum())
    wape = float(np.abs(error).sum() / total) if total > 0 else None
    return mae, wape


def main() -> None:
    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "nyc-taxi-demand-forecaster")

    params = load_params()
    zone_ids = load_selected_zone_ids()
    zone_hour = pd.read_parquet(
        ROOT / "data" / "processed" / "zone_hour_counts.parquet"
    )
    zone_hour = zone_hour[zone_hour.zone_id.isin(zone_ids)]
    feature_table = build_feature_table(zone_hour, params)
    test = split_by_period(feature_table, params, "test")
    cols = feature_columns(params)

    # Coverage: verify, don't assume -- every selected zone should have every test-period hour.
    test_start = pd.Timestamp(params["periods"]["test"]["start"])
    test_end = pd.Timestamp(params["periods"]["test"]["end"])
    expected_hours = int((test_end - test_start).total_seconds() // 3600)
    expected_rows = len(zone_ids) * expected_hours
    coverage = len(test) / expected_rows

    client = MlflowClient()
    champion_version = client.get_model_version_by_alias(model_name, "champion")
    candidate_version = client.get_model_version_by_alias(model_name, "candidate")
    # Champion (the baseline pyfunc wrapper) loads fine via the generic pyfunc interface.
    # The HGB candidate must load via its native sklearn flavor: pyfunc's strict schema
    # enforcement rejects the zone_id category dtype (recorded as plain "long" in the
    # signature), even though the values are identical -- a schema-validation quirk, not
    # a real data problem. mlflow.sklearn.load_model bypasses that layer entirely.
    champion = mlflow.pyfunc.load_model(
        f"models:/{model_name}/{champion_version.version}"
    )
    candidate = mlflow.sklearn.load_model(
        f"models:/{model_name}/{candidate_version.version}"
    )

    champion_pred = champion.predict(test[["lag_168"]])
    candidate_pred = candidate.predict(test[cols])

    champion_mae, champion_wape = mae_wape(test["target"], champion_pred)
    candidate_mae, candidate_wape = mae_wape(test["target"], candidate_pred)
    mae_improvement = (champion_mae - candidate_mae) / champion_mae

    by_zone = pd.DataFrame(
        {
            "zone_id": test["zone_id"],
            "champion_abs_err": (test["target"] - champion_pred).abs(),
            "candidate_abs_err": (test["target"] - candidate_pred).abs(),
        }
    )
    zone_mae = by_zone.groupby("zone_id", observed=True).mean()
    zone_mae["regression_pct"] = (
        zone_mae.candidate_abs_err - zone_mae.champion_abs_err
    ) / zone_mae.champion_abs_err
    worst_zone_regression = float(zone_mae["regression_pct"].max())
    regressed_zones = zone_mae[zone_mae.regression_pct > MAX_ZONE_REGRESSION]

    gates = {
        "mae_improvement_pct": mae_improvement * 100,
        "mae_gate_pass": mae_improvement >= MIN_MAE_IMPROVEMENT,
        "wape_gate_pass": candidate_wape is None
        or champion_wape is None
        or candidate_wape <= champion_wape,
        "zone_regression_gate_pass": bool(len(regressed_zones) == 0),
        "worst_zone_regression_pct": worst_zone_regression * 100,
        "coverage_pass": coverage >= 1.0,
        "coverage_pct": coverage * 100,
    }
    overall_pass = all(
        [
            gates["mae_gate_pass"],
            gates["wape_gate_pass"],
            gates["zone_regression_gate_pass"],
            gates["coverage_pass"],
        ]
    )

    decision = {
        "champion_version": champion_version.version,
        "candidate_version": candidate_version.version,
        "champion_mae": champion_mae,
        "candidate_mae": candidate_mae,
        "champion_wape": champion_wape,
        "candidate_wape": candidate_wape,
        **gates,
        "overall_pass": overall_pass,
    }
    print(json.dumps(decision, indent=2))

    out_path = ROOT / "config" / f"promotion_decision_v{candidate_version.version}.json"
    out_path.write_text(json.dumps(decision, indent=2) + "\n")
    print(
        f"\nwrote {out_path.relative_to(ROOT)} -- decision recorded, champion alias NOT moved"
    )


if __name__ == "__main__":
    main()
