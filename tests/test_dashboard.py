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


def test_rendered_weekday_charts_use_modeled_zones_and_preserve_gaps(monkeypatch):
    from dashboard import app

    # The first UTC timestamp is still Wednesday evening in New York.
    hours = pd.to_datetime(
        ["2020-01-09T04:00Z"] * 3
        + ["2020-01-09T06:00Z"] * 2
        + ["2020-01-10T06:00Z"]
        + ["2020-02-18T06:00Z"] * 2
    )
    counts = pd.DataFrame(
        {
            "hour_utc": hours,
            "zone_id": [237, 161, 999, 237, 161, 237, 237, 161],
            "pickups": [10, 4, 1000, 0, 0, 8, 9, 42],
            "is_complete": [True] * 7 + [False],
        }
    )
    zones = [
        {"zone_id": zone, "name": str(zone), "borough": "Manhattan"}
        for zone in [237, 161]
    ]
    week = pd.Timestamp("2020-02-18T06:00Z")
    report = pd.DataFrame({"window_end": [week]})
    monkeypatch.setattr(app, "load_selected_zones", lambda: zones)
    monkeypatch.setattr(app, "load_zone_hour_counts", lambda: counts)
    monkeypatch.setattr(app, "load_drift_report", lambda: report)
    monkeypatch.setattr(
        app, "load_drift_report_by_zone", lambda: report.assign(zone_id=237)
    )
    monkeypatch.setattr(app, "drift_dashboard_controls", lambda **kwargs: {})
    figures = []
    captions = []
    monkeypatch.setattr(
        app.st, "plotly_chart", lambda fig, **kwargs: figures.append(fig)
    )
    monkeypatch.setattr(app.st, "caption", captions.append)
    app.main()

    _, zone_figure, city_figure = figures
    zone_train, zone_test = [pd.Series(bar.y, index=bar.x) for bar in zone_figure.data]
    city_train, city_test = [pd.Series(bar.y, index=bar.x) for bar in city_figure.data]
    assert city_train["Wednesday"] == 14
    assert zone_train["Wednesday"] == 10
    assert city_train["Thursday"] == zone_train["Thursday"] == 0
    assert pd.isna(city_train["Monday"]) and pd.isna(zone_train["Monday"])
    assert zone_train["Friday"] == 8 and pd.isna(city_train["Friday"])
    assert zone_test["Tuesday"] == 9 and pd.isna(city_test["Tuesday"])
    assert any("No data" in note and "Monday" in note for note in captions)
