"""Tests for the Drift Monitoring Dashboard data layer and payload assembly.

Skips (not fails) when reports/drift_report*.parquet or data/processed/*.parquet are
absent -- both are gitignored generated artifacts (scripts/compute_drift.py,
scripts/aggregate_and_split.py), so a fresh checkout or CI runner won't have them yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard.data import (
    get_available_weeks,
    get_nearest_drift_row,
    load_drift_report,
    load_drift_report_by_zone,
    load_params,
    load_selected_zones,
    load_zone_hour_counts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED_ARTIFACTS = [
    REPO_ROOT / "reports" / "drift_report.parquet",
    REPO_ROOT / "reports" / "drift_report_by_zone.parquet",
    REPO_ROOT / "data" / "processed" / "zone_hour_counts.parquet",
]
requires_generated_artifacts = pytest.mark.skipif(
    not all(path.exists() for path in _REQUIRED_ARTIFACTS),
    reason=(
        "needs reports/drift_report*.parquet and data/processed/zone_hour_counts.parquet -- "
        "run scripts/aggregate_and_split.py and scripts/compute_drift.py first"
    ),
)


def test_params_and_zones_loading():
    params = load_params()
    assert "periods" in params
    assert "train" in params["periods"]
    assert "test" in params["periods"]

    zones = load_selected_zones()
    assert len(zones) == 20
    assert any(z["zone_id"] == 237 for z in zones)


@requires_generated_artifacts
def test_drift_reports_read_only_schema():
    drift_df = load_drift_report()
    assert not drift_df.empty
    expected_cols = [
        "window_end",
        "weighted_score",
        "max_state",
        "persistent_alert",
        "state_pickups_residual",
        "state_zone_mix",
        "state_hour_of_day_mix",
        "state_weekend_mix",
    ]
    for col in expected_cols:
        assert col in drift_df.columns

    zone_df = load_drift_report_by_zone()
    assert not zone_df.empty
    assert "zone_id" in zone_df.columns
    assert "state_pickups_residual" in zone_df.columns
    assert "persistent_alert" in zone_df.columns


@requires_generated_artifacts
def test_available_weeks_and_nearest_row():
    zone_df = load_drift_report_by_zone()
    weeks = get_available_weeks(zone_df)
    assert len(weeks) >= 4

    drift_df = load_drift_report()
    nearest = get_nearest_drift_row(drift_df, weeks[0])
    assert isinstance(nearest, pd.Series)
    assert "max_state" in nearest
    assert "weighted_score" in nearest


@requires_generated_artifacts
def test_zone_hour_counts_and_dow_calculation():
    raw_df = load_zone_hour_counts()
    assert "zone_id" in raw_df.columns
    assert "hour_utc" in raw_df.columns
    assert "pickups" in raw_df.columns

    # Check day of week computation for zone 237
    z237 = raw_df[raw_df["zone_id"] == 237].head(48).copy()
    z237["hour_ny"] = z237["hour_utc"].dt.tz_convert("America/New_York")
    z237["day_name"] = z237["hour_ny"].dt.day_name()
    dow_means = z237.groupby("day_name")["pickups"].mean()
    assert not dow_means.empty
