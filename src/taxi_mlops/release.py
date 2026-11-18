"""Pinned, hashed local releases for historical taxi-demand replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from taxi_mlops.features import build_feature_table, feature_columns

HISTORY_POLICY = "accepted-monthly-snapshot; available-at-hour-end"


class HistoryUnavailable(ValueError):
    """A forecast cannot obtain complete observations available by its origin."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_id(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "release_id"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def validate_history(history: pd.DataFrame, zones: list[int]) -> None:
    required = {"zone_id", "hour_utc", "pickups", "is_complete", "available_at"}
    if not required.issubset(history.columns) or history.empty:
        raise ValueError("History is empty or lacks required columns")
    if not pd.api.types.is_integer_dtype(history.zone_id):
        raise ValueError("History zone IDs must be integers")
    if history.zone_id.isna().any() or set(history.zone_id.unique()) != set(zones):
        raise ValueError("History must cover exactly the frozen zone list")
    for column in ("hour_utc", "available_at"):
        values = history[column]
        if not isinstance(values.dtype, pd.DatetimeTZDtype) or values.isna().any():
            raise ValueError(f"{column} must contain timezone-aware timestamps")
        if str(values.dt.tz) != "UTC":
            raise ValueError(f"{column} must use UTC")
    if (history.hour_utc != history.hour_utc.dt.floor("h")).any():
        raise ValueError("History intervals must start on UTC hours")
    if (history.available_at < history.hour_utc + pd.Timedelta(hours=1)).any():
        raise ValueError("An observation cannot be available before its hour ends")
    if history.duplicated(["zone_id", "hour_utc"]).any():
        raise ValueError("Duplicate zone-hour observations")
    if (
        not pd.api.types.is_bool_dtype(history.is_complete)
        or history.is_complete.isna().any()
    ):
        raise ValueError("Completeness must be explicitly boolean")
    if not pd.api.types.is_numeric_dtype(history.pickups):
        raise ValueError("Pickup counts must be numeric")
    counts = history.loc[history.is_complete, "pickups"].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Complete pickup counts must be finite nonnegative integers")


def prepare_replay_history(
    raw: pd.DataFrame, zones: list[int], *, accept_complete_snapshot: bool = False
) -> pd.DataFrame:
    """Preserve explicit observation metadata; only infer it under the stated replay assumption."""
    flags = {"is_complete", "available_at"}
    present = flags.intersection(raw.columns)
    if present and present != flags:
        raise ValueError("History needs both completeness and availability metadata")
    columns = ["zone_id", "hour_utc", "pickups"] + sorted(present)
    history = raw.loc[raw.zone_id.isin(zones), columns].copy()
    if not present:
        if not accept_complete_snapshot:
            raise ValueError(
                "Unmarked monthly snapshots require explicit replay acceptance"
            )
        history["is_complete"] = True
        history["available_at"] = history.hour_utc + pd.Timedelta(hours=1)
    validate_history(history, zones)
    return history


def forecast_features(
    history: pd.DataFrame, params: dict, zones: list[int], zone_id: int, origin
) -> pd.DataFrame:
    origin = pd.Timestamp(origin).tz_convert("UTC")
    hours = max(
        params["features"]["lags_hours"] + params["features"]["rolling_windows_hours"]
    )
    expected = pd.date_range(
        origin - pd.Timedelta(hours=hours), periods=hours, freq="h"
    )
    window = history.loc[
        (history.zone_id == zone_id)
        & (history.hour_utc >= expected[0])
        & (history.hour_utc < origin)
    ].sort_values("hour_utc")
    if not pd.DatetimeIndex(window.hour_utc).equals(expected):
        raise HistoryUnavailable("Required history contains missing or duplicate hours")
    if not window.is_complete.all() or not (window.available_at <= origin).all():
        raise HistoryUnavailable(
            "Required observations are incomplete or not yet available"
        )
    if not np.isfinite(window.pickups.to_numpy(dtype=float)).all():
        raise HistoryUnavailable("Required pickup counts are unavailable")
    # The target is unknown at issuance. NaN is a placeholder, never an observed zero.
    target = pd.DataFrame(
        {"zone_id": [zone_id], "hour_utc": [origin], "pickups": [np.nan]}
    )
    inputs = pd.concat(
        [window[["zone_id", "hour_utc", "pickups"]], target], ignore_index=True
    )
    features = build_feature_table(inputs, params)
    features = features.loc[features.hour_utc == origin, feature_columns(params)].copy()
    if len(features) != 1:
        raise HistoryUnavailable("Insufficient history to construct forecast features")
    features["zone_id"] = pd.Categorical(features.zone_id, categories=sorted(zones))
    return features


def checked_prediction(model, inputs: pd.DataFrame) -> float:
    if isinstance(model, mlflow.pyfunc.PyFuncModel):
        schema = model.metadata.get_input_schema()
        names = schema.input_names() if schema is not None else []
        if names and all(isinstance(name, str) for name in names):
            inputs = inputs.loc[:, names]
    predictions = np.asarray(model.predict(inputs))
    if predictions.shape != (1,) or not np.issubdtype(predictions.dtype, np.number):
        raise ValueError("Model must return one numeric prediction")
    if np.iscomplexobj(predictions):
        raise ValueError("Complex predictions are invalid")
    value = float(predictions[0])
    if not np.isfinite(value) or value < 0:
        raise ValueError("Model prediction must be finite and nonnegative")
    return value


@dataclass
class Release:
    manifest: dict
    model: Any
    params: dict
    zones: list[int]
    history: pd.DataFrame


def load_release(directory: str | Path) -> Release:
    root = Path(directory).resolve()
    manifest = json.loads((root / "release.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("release_id") != release_id(
        manifest
    ):
        raise ValueError("Invalid release manifest or identity")
    if manifest.get("history_policy") != HISTORY_POLICY:
        raise ValueError("Unsupported historical availability policy")
    for field in ("model_name", "model_version", "run_id"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"Release lacks {field}")
    if not manifest["model_version"].isdigit() or int(manifest["model_version"]) < 1:
        raise ValueError("Model version must be a positive numeric Registry version")
    files = manifest.get("files", {})
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "release.json"
    }
    if not files or set(files) != actual:
        raise ValueError("Release file inventory differs from its manifest")
    for relative, expected_hash in files.items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or sha256_file(path) != expected_hash
        ):
            raise ValueError(f"Release checksum mismatch: {relative}")
    params = json.loads((root / "features.json").read_text())
    for key in ("lags_hours", "rolling_windows_hours"):
        values = params["features"][key]
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v <= 0 for v in values)
        ):
            raise ValueError("Feature windows must be positive integer hours")
    zones = json.loads((root / "zones.json").read_text())
    if (
        not isinstance(zones, list)
        or not zones
        or any(type(z) is not int or z <= 0 for z in zones)
        or len(set(zones)) != len(zones)
    ):
        raise ValueError("Invalid frozen zone list")
    history = pd.read_parquet(root / "history.parquet")
    validate_history(history, zones)
    metadata = mlflow.models.Model.load(str(root / "model" / "MLmodel"))
    if metadata.run_id != manifest["run_id"]:
        raise ValueError("Model source run differs from release manifest")
    if manifest["loader"] == "sklearn" and "sklearn" in metadata.flavors:
        model = mlflow.sklearn.load_model(str(root / "model"))
    elif manifest["loader"] == "pyfunc" and "python_function" in metadata.flavors:
        model = mlflow.pyfunc.load_model(str(root / "model"))
    else:
        raise ValueError("Model flavor differs from release manifest")
    # Readiness requires at least one usable history window and a real smoke prediction.
    hours = max(
        params["features"]["lags_hours"] + params["features"]["rolling_windows_hours"]
    )
    usable = False
    for zone, group in history.groupby("zone_id", sort=False):
        complete = group.sort_values("hour_utc").reset_index(drop=True)
        for end in range(hours, len(complete) + 1):
            origin = complete.hour_utc.iloc[end - 1] + pd.Timedelta(hours=1)
            try:
                features = forecast_features(history, params, zones, int(zone), origin)
            except HistoryUnavailable:
                continue
            checked_prediction(model, features)
            usable = True
            break
        if usable:
            break
    if not usable:
        raise ValueError("History has no usable complete forecast window")
    return Release(manifest, model, params, zones, history)
