"""Log the prev-week baseline as a pyfunc model and register it as the initial `champion`.

Resolves the open question from ADR-0005: there's no existing champion for a first
candidate to be measured against, so the gate "beat champion by >=5% MAE" can't apply yet.
Bootstrapping champion with the best-evidenced baseline (prev-week lag, which beat prev-day
decisively on test: MAE 47.87 vs 63.46) gives that gate a real, defined target from day one,
consistent with the project's own stated principle -- a well-operated simple model beats an
unreproducible one.

scripts/baselines.py only logged metrics, never a loadable model (a lag lookup has nothing
to fit/pickle). This wraps it as an mlflow.pyfunc model so the Registry's alias mechanics
(resolve-once, load-by-version) work identically whether champion is this baseline or a
trained model -- serving code never needs to special-case it.

Usage:
    uv run python scripts/register_baseline_champion.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature

from taxi_mlops import tracking
from taxi_mlops.config import load_params

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
LAG_HOURS = 168
MODEL_TYPE = "baseline_prev_week"
ALIAS = "champion"


class PrevWeekBaseline(mlflow.pyfunc.PythonModel):
    """Predicted pickups = actual pickups 168 hours (7 days) earlier. Parameter-free, not fit."""

    LAG_COLUMN = "lag_168"

    def predict(self, context, model_input, params=None):
        return model_input[self.LAG_COLUMN].to_numpy()


def load_selected_zone_ids() -> list[int]:
    zones = json.loads((ROOT / "config" / "selected_zones.json").read_text())
    return [z["zone_id"] for z in zones]


def build_lag_168_table(params: dict) -> pd.DataFrame:
    zone_ids = load_selected_zone_ids()
    zone_hour = pd.read_parquet(PROCESSED_DIR / "zone_hour_counts.parquet")
    zone_hour = zone_hour[zone_hour.zone_id.isin(zone_ids)]

    df = zone_hour.sort_values(["zone_id", "hour_utc"]).reset_index(drop=True).copy()
    df["lag_168"] = df.groupby("zone_id")["pickups"].shift(LAG_HOURS)
    df = df.rename(columns={"pickups": "target"}).dropna(subset=["lag_168"])
    return df


def main() -> None:
    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-demand"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "nyc-taxi-demand-forecaster")

    params = load_params()
    df = build_lag_168_table(params)

    test_start = pd.Timestamp(params["periods"]["test"]["start"], tz="America/New_York").tz_convert("UTC")
    test_end = pd.Timestamp(params["periods"]["test"]["end"], tz="America/New_York").tz_convert("UTC")
    test = df[(df.hour_utc >= test_start) & (df.hour_utc < test_end)]

    model = PrevWeekBaseline()
    sample_input = test[["lag_168"]].head(5)
    signature = infer_signature(sample_input, model.predict(None, sample_input))

    with mlflow.start_run(run_name=tracking.run_name(MODEL_TYPE, "champion_bootstrap")) as run:
        mlflow.set_tags(
            {
                tracking.TAG_GIT_COMMIT: tracking.git_commit(),
                tracking.TAG_GIT_DIRTY: str(tracking.git_dirty()),
                tracking.TAG_SCENARIO: tracking.SCENARIO_HEALTHY,
            }
        )
        mlflow.log_params({tracking.PARAM_MODEL_TYPE: MODEL_TYPE, "lag_hours": LAG_HOURS})
        mlflow.pyfunc.log_model(python_model=model, name="model", signature=signature, input_example=sample_input)
        run_id = run.info.run_id

    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, ALIAS, version.version)
    print(f"{model_name} v{version.version} registered from run {run_id}, now @{ALIAS}")

    # Resolve once, load the immutable numeric version (not re-resolving @champion), verify it.
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")
    pred = loaded.predict(test[["lag_168"]].head(3))
    print(f"smoke prediction OK: {pred.tolist()}")


if __name__ == "__main__":
    main()
