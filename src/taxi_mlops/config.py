"""Load and validate the forecast timing configuration."""

from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

PERIOD_NAMES = ("warmup", "train", "calibration", "test", "replay")


def load_params(path: str | Path = "params.yaml") -> dict:
    """Read parameters and reject invalid timing or history settings."""
    try:
        params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError("Configuration must contain valid YAML") from error

    if not isinstance(params, dict):
        raise ValueError("Configuration must be a mapping")

    try:
        horizon = params["forecast"]["horizon_hours"]
        zone = params["forecast"]["timezone"]
        periods = params["periods"]
        features = params["features"]
    except (KeyError, TypeError) as error:
        raise ValueError("Missing required configuration sections") from error

    if type(horizon) is not int or horizon != 1:
        raise ValueError("horizon_hours must be the integer 1")
    if zone != "America/New_York":
        raise ValueError("forecast.timezone must be America/New_York")
    tz = ZoneInfo(zone)

    windows = []
    for name in PERIOD_NAMES:
        try:
            start, end = [
                datetime.combine(
                    date.fromisoformat(periods[name][edge]),
                    time.min,
                    tzinfo=tz,
                ).astimezone(UTC)
                for edge in ("start", "end")
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} needs quoted ISO start/end dates") from error
        if start >= end:
            raise ValueError(f"{name} must start before it ends")
        if windows and windows[-1][1] != start:
            raise ValueError("Periods must be contiguous without overlap")
        windows.append((start, end))

    history = []
    for field in ("lags_hours", "rolling_windows_hours"):
        if not isinstance(features, dict):
            raise ValueError("features must be a mapping")
        values = features.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value <= 0 for value in values)
        ):
            raise ValueError(f"{field} must contain positive integer hours")
        history.extend(values)

    warmup_hours = (windows[0][1] - windows[0][0]).total_seconds() / 3600
    if warmup_hours < max(history):
        raise ValueError("Insufficient elapsed hours in warmup")

    return params
