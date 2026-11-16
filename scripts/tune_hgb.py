"""Small grid search over HistGradientBoostingRegressor configs, selected by calibration MAE.

Fits on train only; never touches test (test is the frozen final evaluation, per the
forecast contract). Per ADR-0001, tuning stays limited to a few recorded configurations --
this is a small grid, not an open-ended optimizer. Every trial is logged as a nested MLflow
run; the winning config is written to config/hgb_best_params.json for scripts/train_hgb.py.

Usage:
    uv run python scripts/tune_hgb.py
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from taxi_mlops import tracking
from taxi_mlops.config import load_params
from taxi_mlops.features import build_feature_table, feature_columns, split_by_period

ROOT = Path(__file__).resolve().parents[1]

GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [4, 6, 8],
    "max_iter": [150, 300],
}


def load_selected_zone_ids() -> list[int]:
    zones = json.loads((ROOT / "config" / "selected_zones.json").read_text())
    return [z["zone_id"] for z in zones]


def main() -> None:
    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-demand"))

    params = load_params()
    seed = params["random_seed"]
    zone_ids = load_selected_zone_ids()

    zone_hour = pd.read_parquet(ROOT / "data" / "processed" / "zone_hour_counts.parquet")
    zone_hour = zone_hour[zone_hour.zone_id.isin(zone_ids)]
    feature_table = build_feature_table(zone_hour, params)

    train = split_by_period(feature_table, params, "train")
    calibration = split_by_period(feature_table, params, "calibration")
    cols = feature_columns(params)
    X_train, y_train = train[cols], train["target"]
    X_cal, y_cal = calibration[cols], calibration["target"]

    combos = [dict(zip(GRID.keys(), values, strict=True)) for values in itertools.product(*GRID.values())]
    results = []

    with mlflow.start_run(run_name="hgb_tuning"):
        for i, hp in enumerate(combos):
            with mlflow.start_run(run_name=f"hgb_trial_{i}", nested=True):
                model = HistGradientBoostingRegressor(
                    loss="poisson",
                    random_state=seed,
                    categorical_features="from_dtype",
                    **hp,
                )
                model.fit(X_train, y_train)
                mae = mean_absolute_error(y_cal, model.predict(X_cal))

                mlflow.log_params({**hp, tracking.PARAM_MODEL_TYPE: "hgb", tracking.PARAM_SEED: seed})
                mlflow.log_metric(tracking.mae_key("calibration"), mae)
                results.append({**hp, "calibration_mae": mae})
                print(f"trial {i}: {hp} -> calibration MAE {mae:.3f}")

        best = min(results, key=lambda r: r["calibration_mae"])
        mlflow.log_dict({"grid": GRID, "results": results, "best": best}, "tuning_results.json")

    best_params = {k: v for k, v in best.items() if k != "calibration_mae"}
    config_dir = ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "hgb_best_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
    print(f"best: {best}")


if __name__ == "__main__":
    main()
