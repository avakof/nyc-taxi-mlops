"""ADR-0009's math, verified the same way features.py is: hand-computable small cases for
the pure statistics, an explicit check of the 2x2 state table, the persistence rule, and
-- the one that would catch a real bug -- that a drift window never sees data at or after
its own evaluation hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxi_mlops import monitoring as m

CFG = {
    "window_hours": 168,
    "min_window_hours": 72,
    "thresholds": {
        "psi": {"warning": 0.10, "retrain": 0.25},
        "cohens_d": {"warning": 0.2, "retrain": 0.5},
        "cohens_w": {"warning": 0.1, "retrain": 0.3},
        "cohens_h": {"warning": 0.2, "retrain": 0.5},
    },
    "adf": {"alpha": 0.05, "min_observations": 30},
    "persistence": {"min_state": 2, "required_consecutive_windows": 3},
}


def _zone_hour(
    n_hours: int, zones: tuple[int, ...] = (1, 2), start: str = "2020-01-01 05:00"
) -> pd.DataFrame:
    """Dense zone x hour grid (matches zone_hour_counts.parquet's shape), start at 05:00 UTC
    == 2020-01-01 00:00 America/New_York so local-day boundaries land cleanly."""
    hours = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    local = hours.tz_convert("America/New_York")
    base = 20 + 10 * np.sin(local.hour * 2 * np.pi / 24)
    frames = []
    for zone in zones:
        zone_offset = zone * 5
        is_weekend = (local.day_of_week >= 5).astype(int)
        pickups = np.clip(base + zone_offset - 8 * is_weekend, 0, None)
        frames.append(
            pd.DataFrame({"zone_id": zone, "hour_utc": hours, "pickups": pickups})
        )
    return pd.concat(frames, ignore_index=True)


# --- eta_squared / dimension_weights -------------------------------------------------


def test_eta_squared_known_decomposition():
    # Two groups, each perfectly constant -> all variance is between-group.
    values = pd.Series([10.0, 10.0, 20.0, 20.0])
    groups = pd.Series(["a", "a", "b", "b"])
    assert m.eta_squared(values, groups) == pytest.approx(1.0)


def test_eta_squared_zero_when_groups_dont_differ():
    values = pd.Series([10.0, 20.0, 10.0, 20.0])
    groups = pd.Series(["a", "a", "b", "b"])
    assert m.eta_squared(values, groups) == pytest.approx(0.0)


def test_dimension_weights_sum_to_one_and_floor_weekend():
    # Weekend has (close to) zero effect here -- the floor must still kick in.
    zone_hour = _zone_hour(24 * 14)
    weights = m.dimension_weights(zone_hour, weekend_floor=0.05)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["weekend_mix"] >= 0.05 - 1e-9
    assert set(weights) == set(m.DIMENSIONS)


# --- PSI / categorical PSI ------------------------------------------------------------


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(0, 1, 2000)
    bin_edges = m.quantile_bin_edges(reference, 10)
    assert m.psi(reference, current, bin_edges) < 0.05


def test_psi_detects_a_mean_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(3, 1, 2000)
    bin_edges = m.quantile_bin_edges(reference, 10)
    assert m.psi(reference, current, bin_edges) > 0.25


def test_categorical_psi_detects_share_shift():
    reference_share = pd.Series({"a": 0.5, "b": 0.5})
    same_share = pd.Series({"a": 50, "b": 50})
    shifted_share = pd.Series({"a": 95, "b": 5})
    assert m.categorical_psi(reference_share, same_share) == pytest.approx(
        0.0, abs=1e-6
    )
    assert m.categorical_psi(reference_share, shifted_share) > 0.25


# --- Effect sizes ----------------------------------------------------------------------


def test_cohens_d_log1p_zero_for_identical_samples():
    sample = np.array([5.0, 10.0, 15.0, 20.0] * 10)
    assert m.cohens_d_log1p(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_log1p_sign_matches_direction_of_shift():
    reference = np.full(50, 10.0)
    higher = np.full(50, 40.0)
    lower = np.full(50, 2.0)
    # Pooled SD is 0 for two constant-but-different-mean samples only once log1p is
    # applied to BOTH -- use a touch of spread so pooled_sd isn't degenerate.
    rng = np.random.default_rng(1)
    reference = reference + rng.normal(0, 0.5, 50)
    higher = higher + rng.normal(0, 0.5, 50)
    lower = lower + rng.normal(0, 0.5, 50)
    assert m.cohens_d_log1p(reference, higher) > 0
    assert m.cohens_d_log1p(reference, lower) < 0


def test_cohens_h_direction_and_zero_at_equal_shares():
    assert m.cohens_h(0.3, 0.3) == pytest.approx(0.0, abs=1e-9)
    assert m.cohens_h(0.1, 0.5) > 0
    assert m.cohens_h(0.5, 0.1) < 0


def test_cohens_w_zero_when_shares_match_and_positive_when_they_dont():
    reference_share = pd.Series({"a": 0.5, "b": 0.5})
    assert m.cohens_w(reference_share, pd.Series({"a": 50, "b": 50})) == pytest.approx(
        0.0, abs=1e-6
    )
    assert m.cohens_w(reference_share, pd.Series({"a": 90, "b": 10})) > 0.3


# --- 2x2 state table ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "detection,impact,expected",
    [
        (0.0, 0.0, 0),
        (0.15, 0.0, 1),  # detection warns alone
        (0.0, 0.3, 1),  # impact warns alone
        (0.15, 0.3, 2),  # both warn, neither retrains
        (0.30, 0.0, 3),  # detection alone retrains
        (0.0, 0.6, 3),  # impact alone retrains
        (0.30, 0.6, 4),  # both retrain
    ],
)
def test_classify_state_matches_the_adr_0009_table(detection, impact, expected):
    state = m.classify_state(
        detection,
        impact,
        detection_warning=0.10,
        detection_retrain=0.25,
        impact_warning=0.2,
        impact_retrain=0.5,
    )
    assert state == expected


def test_classify_state_is_none_when_a_statistic_is_unavailable():
    assert (
        m.classify_state(
            None,
            0.3,
            detection_warning=0.1,
            detection_retrain=0.25,
            impact_warning=0.2,
            impact_retrain=0.5,
        )
        is None
    )


# --- Persistence -------------------------------------------------------------------------


def test_persistent_alert_requires_consecutive_windows_not_just_total_count():
    # Three flagged windows total, but not consecutive -> never alerts.
    scattered = pd.Series([2, 0, 2, 0, 2, 0, 0])
    assert not m.persistent_alert(scattered, min_state=2, required_consecutive=3).any()
    # Three IN A ROW -> alerts from the third one on.
    consecutive = pd.Series([0, 2, 2, 2, 0])
    alerts = m.persistent_alert(consecutive, min_state=2, required_consecutive=3)
    assert list(alerts) == [False, False, False, True, False]


def test_persistent_alert_treats_missing_state_as_not_flagged():
    with_gap = pd.Series([2, 2, np.nan, 2])
    assert not m.persistent_alert(with_gap, min_state=2, required_consecutive=3).iloc[
        -1
    ]


# --- End-to-end: reference profile, evaluate_window, and the causal report -----------------


def test_evaluate_window_is_calm_on_data_matching_the_reference():
    reference_data = _zone_hour(24 * 21)
    reference = m.build_reference_profile(reference_data)
    # A full week (168h), same as CFG's window_hours, so the weekday/weekend mix matches
    # the reference's own mix -- a shorter window would show a spurious weekend-share
    # "shift" purely from which days it happens to cover, not real drift.
    healthy_window = _zone_hour(24 * 7, start="2020-01-15 05:00")
    result = m.evaluate_window(healthy_window, reference, CFG)
    assert result["max_state"] <= 1


def test_evaluate_window_flags_an_injected_demand_collapse():
    reference_data = _zone_hour(24 * 21)
    reference = m.build_reference_profile(reference_data)
    collapsed = _zone_hour(24 * 7, start="2020-02-15 05:00")
    collapsed["pickups"] = collapsed["pickups"] * 0.2  # COVID-style thinning
    result = m.evaluate_window(collapsed, reference, CFG)
    assert result["state_pickups_residual"] == 4
    assert result["max_state"] == 4


def test_compute_drift_report_never_scores_a_window_using_future_data():
    # A spike placed well AFTER an early evaluation hour must not affect that hour's report
    # -- this is the leakage check: identical in spirit to test_features.py's lag tests.
    history = _zone_hour(24 * 10)
    reference = m.build_reference_profile(history)
    history_with_future_spike = history.copy()
    late_mask = history_with_future_spike["hour_utc"] >= history_with_future_spike[
        "hour_utc"
    ].max() - pd.Timedelta(hours=5)
    history_with_future_spike.loc[late_mask, "pickups"] *= 50

    early_eval_hour = history["hour_utc"].min() + pd.Timedelta(hours=96)
    report_without_spike = m.compute_drift_report(
        history, reference, CFG, evaluation_hours=pd.DatetimeIndex([early_eval_hour])
    )
    report_with_future_spike = m.compute_drift_report(
        history_with_future_spike,
        reference,
        CFG,
        evaluation_hours=pd.DatetimeIndex([early_eval_hour]),
    )
    pd.testing.assert_frame_equal(report_without_spike, report_with_future_spike)


def test_compute_zone_drift_report_isolates_a_single_collapsed_zone():
    # One zone collapses, the other stays healthy -- this is the "one construction-project
    # zone shouldn't retrain the whole model" scenario the per-zone report exists for.
    reference_data = _zone_hour(24 * 21, zones=(1, 2))
    reference = m.build_reference_profile(reference_data)
    week = _zone_hour(24 * 7, zones=(1, 2), start="2020-02-15 05:00")
    week.loc[week.zone_id == 1, "pickups"] *= 0.2

    weekly_hours = m.weekly_evaluation_hours(week["hour_utc"], CFG["window_hours"])
    report = m.compute_zone_drift_report(
        week, reference, CFG, zone_ids=[1, 2], evaluation_hours=weekly_hours
    )
    report = report[~report["insufficient_data"]]
    assert set(report.loc[report.state_pickups_residual == 4, "zone_id"]) == {1}
    assert (report.loc[report.zone_id == 2, "state_pickups_residual"] <= 1).all()


def test_compute_drift_report_marks_short_windows_insufficient_not_silently_healthy():
    history = _zone_hour(10)  # far short of CFG's min_window_hours=24
    reference = m.build_reference_profile(_zone_hour(24 * 21))
    eval_hour = history["hour_utc"].max()
    report = m.compute_drift_report(
        history, reference, CFG, evaluation_hours=pd.DatetimeIndex([eval_hour])
    )
    assert bool(report.loc[0, "insufficient_data"]) is True
    assert "max_state" not in report.columns or pd.isna(report.loc[0].get("max_state"))
