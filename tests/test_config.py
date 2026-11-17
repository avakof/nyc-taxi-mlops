"""taxi_mlops.config.load_params has zero test coverage despite being the one thing that
rejects a broken forecast-timing configuration before it can leak into a training run."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from taxi_mlops.config import load_params

VALID = {
    "forecast": {"horizon_hours": 1, "timezone": "America/New_York"},
    "periods": {
        "warmup": {"start": "2020-01-01", "end": "2020-01-08"},
        "train": {"start": "2020-01-08", "end": "2020-02-03"},
        "calibration": {"start": "2020-02-03", "end": "2020-02-17"},
        "test": {"start": "2020-02-17", "end": "2020-03-01"},
        "replay": {"start": "2020-03-01", "end": "2020-05-01"},
    },
    "features": {"lags_hours": [1, 2, 3, 24, 168], "rolling_windows_hours": [3, 24, 168]},
}


def write_params(tmp_path: Path, overrides: dict) -> Path:
    params = copy.deepcopy(VALID)
    for section, values in overrides.items():
        params[section] = {**params.get(section, {}), **values}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.dump(params))
    return path


def test_valid_config_loads(tmp_path: Path) -> None:
    path = write_params(tmp_path, {})
    params = load_params(path)
    assert params["forecast"]["horizon_hours"] == 1


def test_wrong_horizon_rejected(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"forecast": {"horizon_hours": 2}})
    with pytest.raises(ValueError, match="horizon_hours"):
        load_params(path)


def test_wrong_timezone_rejected(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"forecast": {"timezone": "UTC"}})
    with pytest.raises(ValueError, match="timezone"):
        load_params(path)


def test_overlapping_periods_rejected(tmp_path: Path) -> None:
    # Train now overlaps calibration by a day instead of ending where calibration starts.
    path = write_params(tmp_path, {"periods": {**VALID["periods"], "train": {"start": "2020-01-08", "end": "2020-02-04"}}})
    with pytest.raises(ValueError, match="contiguous"):
        load_params(path)


def test_non_positive_lag_rejected(tmp_path: Path) -> None:
    path = write_params(tmp_path, {"features": {"lags_hours": [1, 0, 24], "rolling_windows_hours": [3]}})
    with pytest.raises(ValueError, match="lags_hours"):
        load_params(path)


def test_warmup_shorter_than_longest_lag_rejected(tmp_path: Path) -> None:
    # Warmup is only 7 days (168h) but a 200h lag needs more history than that.
    path = write_params(tmp_path, {"features": {"lags_hours": [1, 200], "rolling_windows_hours": [3]}})
    with pytest.raises(ValueError, match="warmup"):
        load_params(path)
