"""Batch forecast report for the dashboard's "Forecasted vs. Live Pickups" diagnostic.
Loads the tuned HistGradientBoostingRegressor (Registry alias `candidate`, the real
model -- the served `champion` alias is the previous-week baseline, not what this is
for) and predicts every causal feature row in zone_hour_counts.parquet, writing actual
and forecasted pickups side by side. Precomputed and read-only in the dashboard, same
as reports/drift_report*.parquet -- never run model inference live on page load.

Resolves the model's local artifact path directly from mlflow.db via sqlite3 and loads
it from disk -- no MLflow tracking/registry server needed. (A running server is only
required to resolve a `models:/name/version` URI through its `mlflow-artifacts:/`
proxy; since we already know the on-disk path, that indirection is unnecessary here.)

Usage:
    uv run python scripts/compute_forecasts.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import mlflow
import pandas as pd

from taxi_mlops.config import load_params
from taxi_mlops.features import build_feature_table, feature_columns

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"
MODEL_NAME = "nyc-taxi-demand-forecaster"
MODEL_ALIAS = "candidate"


def resolve_local_model_path(model_name: str, alias: str) -> tuple[str, int, str]:
    """(local_path, version, run_id) for `model_name`@`alias`, read straight from the
    SQLite backend store -- no tracking server, no mlflow.set_tracking_uri."""
    con = sqlite3.connect(MLFLOW_DB)
    try:
        row = con.execute(
            """
            SELECT mv.version, mv.run_id, mv.storage_location
            FROM registered_model_aliases AS a
            JOIN model_versions AS mv
              ON mv.name = a.name AND mv.version = a.version
            WHERE a.name = ? AND a.alias = ?
            """,
            (model_name, alias),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise ValueError(f"No registry row for {model_name}@{alias} in {MLFLOW_DB}")
    version, run_id, storage_location = row
    # storage_location looks like "mlflow-artifacts:/1/models/m-<id>/artifacts" --
    # mlartifacts/ is exactly where this repo's server was configured to land that
    # scheme (--artifacts-destination mlartifacts).
    relative = storage_location.removeprefix("mlflow-artifacts:/")
    local_path = ROOT / "mlartifacts" / relative
    if not local_path.exists():
        raise FileNotFoundError(f"Resolved model path does not exist: {local_path}")
    return str(local_path), version, run_id


def main() -> None:
    params = load_params()
    zone_hour = pd.read_parquet(
        ROOT / "data" / "processed" / "zone_hour_counts.parquet"
    )

    local_path, version, run_id = resolve_local_model_path(MODEL_NAME, MODEL_ALIAS)
    print(f"loading {MODEL_NAME}@{MODEL_ALIAS} (version {version}, run {run_id})")
    print(f"  from {local_path}")
    # Native sklearn flavor, not pyfunc -- pyfunc's strict schema enforcement rejects the
    # zone_id category dtype (same quirk noted in scripts/evaluate_promotion.py).
    model = mlflow.sklearn.load_model(local_path)

    feature_table = build_feature_table(zone_hour, params)
    cols = feature_columns(params)
    predictions = model.predict(feature_table[cols])

    forecast = pd.DataFrame(
        {
            "zone_id": feature_table["zone_id"].astype(int),
            "hour_utc": feature_table["hour_utc"],
            "actual_pickups": feature_table["target"],
            "forecasted_pickups": predictions,
        }
    )

    out_path = ROOT / "reports" / "forecast_vs_actual.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path.relative_to(ROOT)}: {len(forecast)} rows, "
        f"{forecast['hour_utc'].min()} to {forecast['hour_utc'].max()}, "
        f"model_version={version}"
    )


if __name__ == "__main__":
    main()
