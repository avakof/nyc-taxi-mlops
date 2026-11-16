"""Single source of truth for MLflow naming: params, tags, and metric keys.

Import these from every script that logs to MLflow (training, registration,
monitoring) instead of hardcoding key names. See
docs/adr/0002-mlflow-logging-conventions.md for the rationale.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# --- Params (flat, snake_case) ----------------------------------------------

PARAM_MODEL_TYPE = "model_type"
PARAM_SEED = "seed"
PARAM_FEATURE_LAGS_HOURS = "feature_lags_hours"
PARAM_FEATURE_ROLLING_WINDOWS_HOURS = "feature_rolling_windows_hours"
PARAM_N_ZONES = "n_zones"
PARAM_DATA_MONTHS = "data_months"


def split_param_names(period: str) -> tuple[str, str]:
    """Param keys for a period's boundaries, e.g. ('split_train_start', 'split_train_end')."""
    return f"split_{period}_start", f"split_{period}_end"


# --- Tags ---------------------------------------------------------------------

TAG_GIT_COMMIT = "git_commit"
TAG_GIT_DIRTY = "git_dirty"
TAG_DATA_MANIFEST_SHA256 = "data_manifest_sha256"
TAG_SCENARIO = "scenario"

SCENARIO_HEALTHY = "healthy"
SCENARIO_DRIFT_NATURAL = "drift_natural"
SCENARIO_DRIFT_INJECTED = "drift_injected"
SCENARIO_ROLLBACK_DRILL = "rollback_drill"


def worst_zone_tag(split: str) -> str:
    return f"{split}_worst_zone_id"


def run_name(model_type: str, split_role: str) -> str:
    return f"{model_type}_{split_role}"


# --- Metrics (namespaced by split: train / calibration / test / replay) -------


def metric_key(split: str, name: str) -> str:
    """e.g. metric_key("test", "mae") -> "test/mae"."""
    return f"{split}/{name}"


def mae_key(split: str) -> str:
    return metric_key(split, "mae")


def rmse_key(split: str) -> str:
    return metric_key(split, "rmse")


def wape_key(split: str) -> str:
    return metric_key(split, "wape")


def mae_vs_baseline_pct_key(split: str) -> str:
    return metric_key(split, "mae_vs_baseline_pct")


def worst_zone_mae_key(split: str) -> str:
    return metric_key(split, "worst_zone_mae")


# --- Provenance helpers ---------------------------------------------------------


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_dirty() -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(status.strip())


def data_manifest_sha256(manifest_path: str | Path = "data/manifest.json") -> str:
    """Fingerprint of the whole data manifest (every tracked raw file), not just one."""
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


# --- SHAP / feature importance artifacts ---------------------------------------
# Logged once per training run so later feature-importance work (ADR-0001 addendum:
# defer feature pruning to this, not the EDA correlation matrix) doesn't need a retrain.

SHAP_ARTIFACT_PATH = "shap"
SHAP_VALUES_FILENAME = "shap_values.npz"
SHAP_SUMMARY_FILENAME = "shap_summary.json"
