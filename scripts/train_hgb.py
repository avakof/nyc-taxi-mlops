"""Train the final HGB model with scripts/tune_hgb.py's winning hyperparameters, evaluate on
calibration and test, and log everything -- including SHAP values -- to MLflow via
taxi_mlops.tracking, so it lines up directly against the baseline runs.

Usage:
    uv run python scripts/train_hgb.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor

from taxi_mlops import tracking
from taxi_mlops.config import load_params
from taxi_mlops.features import build_feature_table, feature_columns, split_by_period

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODEL_TYPE = "hgb"


def compute_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    error = actual.to_numpy() - predicted
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt((error**2).mean()))
    total_actual = float(actual.sum())
    wape = float(np.abs(error).sum() / total_actual) if total_actual > 0 else None
    return {"mae": mae, "rmse": rmse, "wape": wape}


def load_selected_zone_ids() -> list[int]:
    zones = json.loads((ROOT / "config" / "selected_zones.json").read_text())
    return [z["zone_id"] for z in zones]


def main() -> None:
    load_dotenv(ROOT / ".env")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-demand"))

    params = load_params()
    seed = params["random_seed"]

    best_params_path = ROOT / "config" / "hgb_best_params.json"
    best_params = json.loads(best_params_path.read_text()) if best_params_path.exists() else {}
    if not best_params:
        print("no config/hgb_best_params.json -- run scripts/tune_hgb.py first; using sklearn defaults")

    zone_ids = load_selected_zone_ids()
    zone_hour = pd.read_parquet(ROOT / "data" / "processed" / "zone_hour_counts.parquet")
    zone_hour = zone_hour[zone_hour.zone_id.isin(zone_ids)]
    feature_table = build_feature_table(zone_hour, params)

    train = split_by_period(feature_table, params, "train")
    calibration = split_by_period(feature_table, params, "calibration")
    test = split_by_period(feature_table, params, "test")
    cols = feature_columns(params)
    X_train, y_train = train[cols], train["target"]

    model = HistGradientBoostingRegressor(
        loss="poisson",
        random_state=seed,
        categorical_features="from_dtype",
        **best_params,
    )
    model.fit(X_train, y_train)

    REPORTS_DIR.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=tracking.run_name(MODEL_TYPE, "train_eval")):
        mlflow.log_params(
            {
                tracking.PARAM_MODEL_TYPE: MODEL_TYPE,
                tracking.PARAM_SEED: seed,
                tracking.PARAM_FEATURE_LAGS_HOURS: ",".join(map(str, params["features"]["lags_hours"])),
                tracking.PARAM_FEATURE_ROLLING_WINDOWS_HOURS: ",".join(
                    map(str, params["features"]["rolling_windows_hours"])
                ),
                tracking.PARAM_N_ZONES: len(zone_ids),
                tracking.PARAM_DATA_MONTHS: ",".join(params["data"]["months"]),
                **best_params,
            }
        )
        for period in ("train", "calibration", "test"):
            start_key, end_key = tracking.split_param_names(period)
            mlflow.log_params(
                {start_key: params["periods"][period]["start"], end_key: params["periods"][period]["end"]}
            )

        mlflow.set_tags(
            {
                tracking.TAG_GIT_COMMIT: tracking.git_commit(),
                tracking.TAG_GIT_DIRTY: str(tracking.git_dirty()),
                tracking.TAG_DATA_MANIFEST_SHA256: tracking.data_manifest_sha256(ROOT / "data" / "manifest.json"),
                tracking.TAG_SCENARIO: tracking.SCENARIO_HEALTHY,
            }
        )

        for split_name, split_df in (("train", train), ("calibration", calibration), ("test", test)):
            X, y = split_df[cols], split_df["target"]
            pred = model.predict(X)
            result = compute_metrics(y, pred)

            mlflow.log_metrics(
                {tracking.mae_key(split_name): result["mae"], tracking.rmse_key(split_name): result["rmse"]}
            )
            if result["wape"] is not None:
                mlflow.log_metric(tracking.wape_key(split_name), result["wape"])

            abs_err = (split_df["target"] - pred).abs()
            by_zone_mae = abs_err.groupby(split_df["zone_id"], observed=True).mean().sort_values(ascending=False)
            worst_zone_id = int(by_zone_mae.index[0])
            mlflow.log_metric(tracking.worst_zone_mae_key(split_name), float(by_zone_mae.iloc[0]))
            mlflow.set_tag(tracking.worst_zone_tag(split_name), worst_zone_id)

            report_path = REPORTS_DIR / f"{MODEL_TYPE}_{split_name}_metrics_by_zone.json"
            report_path.write_text(json.dumps({int(k): float(v) for k, v in by_zone_mae.items()}, indent=2) + "\n")
            mlflow.log_artifact(str(report_path))

            print(f"{MODEL_TYPE} / {split_name}: mae={result['mae']:.2f} rmse={result['rmse']:.2f} wape={result['wape']}")

        # SHAP, computed once here so later feature-importance work (ADR-0004) doesn't need
        # a retrain. Model-agnostic Permutation explainer, not TreeExplainer: ADR-0004 found
        # TreeExplainer broken (both path-dependent and interventional modes) for this model's
        # native categorical splits (categorical_features="from_dtype") -- it disagreed sharply
        # with permutation importance, while the model-agnostic explainer matched it closely.
        # Slower per-sample than TreeExplainer, so kept small on purpose.
        background = train[cols].sample(n=min(200, len(train)), random_state=seed)
        masker = shap.maskers.Independent(background, max_samples=200)
        sample = test[cols].sample(n=min(200, len(test)), random_state=seed)
        explainer = shap.explainers.Permutation(model.predict, masker)
        shap_values = explainer(sample).values

        shap_values_path = REPORTS_DIR / tracking.SHAP_VALUES_FILENAME
        np.savez_compressed(shap_values_path, shap_values=shap_values, feature_names=np.array(cols))
        mlflow.log_artifact(str(shap_values_path), artifact_path=tracking.SHAP_ARTIFACT_PATH)

        mean_abs_shap = {col: float(np.abs(shap_values[:, i]).mean()) for i, col in enumerate(cols)}
        mean_abs_shap = dict(sorted(mean_abs_shap.items(), key=lambda kv: -kv[1]))
        shap_summary_path = REPORTS_DIR / tracking.SHAP_SUMMARY_FILENAME
        shap_summary_path.write_text(json.dumps(mean_abs_shap, indent=2) + "\n")
        mlflow.log_artifact(str(shap_summary_path), artifact_path=tracking.SHAP_ARTIFACT_PATH)
        print("mean |SHAP| by feature:", json.dumps(mean_abs_shap, indent=2))

        signature = infer_signature(X_train, model.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            # Our own freshly-trained model, not a file loaded from an untrusted source.
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "functools.partial",
                "sklearn.utils.validation.check_array",
            ],
        )


if __name__ == "__main__":
    main()
