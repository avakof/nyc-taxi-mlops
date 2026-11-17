"""Evaluate naive prev-day (lag_24) and prev-week (lag_168) baselines on calibration and test.

No fitting involved: a baseline's prediction for a target hour is just the actual pickups
observed 24 (or 168) hours earlier -- an eligible feature per the forecast contract. This
exists to answer whether a trained model is worth deploying at all, not as a formality
(see ADR-0001, Modeling). Logged to MLflow with the naming conventions from
taxi_mlops.tracking / ADR-0002 so both baselines line up against the HGB model later.

Usage:
    uv run python scripts/baselines.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from taxi_mlops import tracking
from taxi_mlops.config import load_params

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

BASELINES = {
    "baseline_prev_day": 24,
    "baseline_prev_week": 168,
}


def load_selected_zone_ids() -> list[int]:
    zones = json.loads((ROOT / "config" / "selected_zones.json").read_text())
    return [z["zone_id"] for z in zones]


def predictions_for_lag(zone_hour: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """Shift each zone's complete hourly series by lag_hours to get a causal "prediction"."""
    df = zone_hour.sort_values(["zone_id", "hour_utc"]).copy()
    df["prediction"] = df.groupby("zone_id")["pickups"].shift(lag_hours)
    return df


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    error = actual - predicted
    mae = float(error.abs().mean())
    rmse = float(np.sqrt((error**2).mean()))
    total_actual = float(actual.sum())
    wape = float(error.abs().sum() / total_actual) if total_actual > 0 else None
    return {"mae": mae, "rmse": rmse, "wape": wape, "total_actual": total_actual}


def evaluate_split(
    predicted_full: pd.DataFrame, split_df: pd.DataFrame, zone_ids: list[int]
) -> dict:
    merged = split_df.merge(
        predicted_full[["zone_id", "hour_utc", "prediction"]],
        on=["zone_id", "hour_utc"],
        how="left",
    )
    merged = merged[merged.zone_id.isin(zone_ids)]
    # Count missing predictions BEFORE dropping them -- a matched key with a NaN prediction
    # (e.g. not enough history yet for this zone-hour's lag) must not disappear uncounted.
    rows_missing_prediction = int(merged.prediction.isna().sum())
    merged = merged.dropna(subset=["prediction"])

    overall = compute_metrics(merged.pickups, merged.prediction)

    by_zone = (
        merged.groupby("zone_id")
        .apply(
            lambda g: compute_metrics(g.pickups, g.prediction)["mae"],
            include_groups=False,
        )
        .sort_values(ascending=False)
    )
    overall["worst_zone_id"] = int(by_zone.index[0])
    overall["worst_zone_mae"] = float(by_zone.iloc[0])
    overall["by_zone_mae"] = {int(k): float(v) for k, v in by_zone.items()}
    overall["rows_evaluated"] = int(len(merged))
    overall["rows_missing_prediction"] = rows_missing_prediction
    return overall


def main() -> None:
    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-demand"))

    params = load_params()
    zone_ids = load_selected_zone_ids()
    zone_hour = pd.read_parquet(PROCESSED_DIR / "zone_hour_counts.parquet")
    calibration = pd.read_parquet(PROCESSED_DIR / "calibration.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    REPORTS_DIR.mkdir(exist_ok=True)

    for model_type, lag_hours in BASELINES.items():
        predicted_full = predictions_for_lag(zone_hour, lag_hours)

        with mlflow.start_run(
            run_name=tracking.run_name(model_type, "calibration_test")
        ):
            mlflow.log_params(
                {
                    tracking.PARAM_MODEL_TYPE: model_type,
                    "lag_hours": lag_hours,
                    tracking.PARAM_N_ZONES: len(zone_ids),
                    tracking.PARAM_DATA_MONTHS: ",".join(params["data"]["months"]),
                }
            )
            mlflow.set_tags(
                {
                    tracking.TAG_GIT_COMMIT: tracking.git_commit(),
                    tracking.TAG_GIT_DIRTY: str(tracking.git_dirty()),
                    tracking.TAG_DATA_MANIFEST_SHA256: tracking.data_manifest_sha256(
                        ROOT / "data" / "manifest.json"
                    ),
                    tracking.TAG_SCENARIO: tracking.SCENARIO_HEALTHY,
                }
            )

            for split_name, split_df in (("calibration", calibration), ("test", test)):
                result = evaluate_split(predicted_full, split_df, zone_ids)

                mlflow.log_metrics(
                    {
                        tracking.mae_key(split_name): result["mae"],
                        tracking.rmse_key(split_name): result["rmse"],
                        tracking.worst_zone_mae_key(split_name): result[
                            "worst_zone_mae"
                        ],
                    }
                )
                if result["wape"] is not None:
                    mlflow.log_metric(tracking.wape_key(split_name), result["wape"])
                mlflow.set_tag(
                    tracking.worst_zone_tag(split_name), result["worst_zone_id"]
                )

                report_path = (
                    REPORTS_DIR / f"{model_type}_{split_name}_metrics_by_zone.json"
                )
                report_path.write_text(
                    json.dumps(result["by_zone_mae"], indent=2) + "\n"
                )
                mlflow.log_artifact(str(report_path))

                print(
                    f"{model_type} / {split_name}: mae={result['mae']:.2f} rmse={result['rmse']:.2f} "
                    f"wape={result['wape']} worst_zone={result['worst_zone_id']} "
                    f"(missing_prediction_rows={result['rows_missing_prediction']})"
                )


if __name__ == "__main__":
    main()
