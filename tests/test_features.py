"""build_feature_table is the one place a leakage bug could creep in silently. Verify its
lags/rolling means are genuinely causal (never include the current or a future hour), and
that calendar features come from NY local time, not raw UTC."""

from __future__ import annotations

import pandas as pd
import pytest

from taxi_mlops.features import build_feature_table, feature_columns, split_by_period

PARAMS = {"features": {"lags_hours": [1, 2], "rolling_windows_hours": [2]}}


@pytest.fixture
def zone_hour() -> pd.DataFrame:
    hours = pd.date_range("2020-01-01 00:00", periods=10, freq="h", tz="UTC")
    zone_a = pd.DataFrame(
        {"zone_id": "A", "hour_utc": hours, "pickups": range(0, 100, 10)}
    )
    zone_b = pd.DataFrame(
        {"zone_id": "B", "hour_utc": hours, "pickups": range(0, 50, 5)}
    )
    return pd.concat([zone_a, zone_b], ignore_index=True)


def test_insufficient_history_rows_dropped(zone_hour: pd.DataFrame) -> None:
    feat = build_feature_table(zone_hour, PARAMS)
    # lag_2/roll_2 need 2 hours of prior history -- the first 2 rows per zone can't have it.
    assert len(feat) == len(zone_hour) - 2 * zone_hour.zone_id.nunique()


def test_lags_are_strictly_from_the_past(zone_hour: pd.DataFrame) -> None:
    feat = build_feature_table(zone_hour, PARAMS)
    zone_a = feat[feat.zone_id == "A"].sort_values("hour_utc").reset_index(drop=True)
    # Zone A pickups are 0,10,20,...,90 -- after dropping, row 0 is the original hour index 2.
    assert zone_a.loc[0, "target"] == 20
    assert zone_a.loc[0, "lag_1"] == 10
    assert zone_a.loc[0, "lag_2"] == 0
    assert zone_a.loc[1, "target"] == 30
    assert zone_a.loc[1, "lag_1"] == 20
    assert zone_a.loc[1, "lag_2"] == 10


def test_rolling_mean_excludes_current_hour(zone_hour: pd.DataFrame) -> None:
    feat = build_feature_table(zone_hour, PARAMS)
    zone_a = feat[feat.zone_id == "A"].sort_values("hour_utc").reset_index(drop=True)
    # roll_2 at the first valid row averages the two PRIOR hours (0, 10), never the current
    # target (20) -- this is exactly the check that would catch a leakage bug.
    assert zone_a.loc[0, "roll_2"] == pytest.approx(5.0)
    assert zone_a.loc[1, "roll_2"] == pytest.approx(15.0)


def test_calendar_features_use_ny_local_time_not_utc(zone_hour: pd.DataFrame) -> None:
    feat = build_feature_table(zone_hour, PARAMS)
    expected_local = feat.hour_utc.dt.tz_convert("America/New_York")
    assert (feat.hour_of_day == expected_local.dt.hour).all()
    assert (feat.day_of_week == expected_local.dt.day_of_week).all()
    assert (feat.is_weekend == (expected_local.dt.day_of_week >= 5).astype(int)).all()


def test_zone_id_is_categorical(zone_hour: pd.DataFrame) -> None:
    feat = build_feature_table(zone_hour, PARAMS)
    assert feat.zone_id.dtype.name == "category"


def test_feature_columns_matches_actual_columns(zone_hour: pd.DataFrame) -> None:
    feat = build_feature_table(zone_hour, PARAMS)
    assert set(feature_columns(PARAMS)).issubset(feat.columns)


def test_split_by_period_is_non_overlapping() -> None:
    # 2020-01-01 00:00 America/New_York == 05:00 UTC (EST, UTC-5 in January) -- start exactly
    # on that boundary so 48 hours covers both periods with nothing left over.
    hours = pd.date_range("2020-01-01 05:00", periods=48, freq="h", tz="UTC")
    feat = pd.DataFrame({"hour_utc": hours, "target": range(48)})
    params = {
        "periods": {
            "train": {"start": "2020-01-01", "end": "2020-01-02"},
            "calibration": {"start": "2020-01-02", "end": "2020-01-03"},
        }
    }
    train = split_by_period(feat, params, "train")
    calibration = split_by_period(feat, params, "calibration")
    assert set(train.hour_utc).isdisjoint(set(calibration.hour_utc))
    assert len(train) + len(calibration) == len(feat)
