"""Regression test for the missing-prediction undercount Ali's review caught: a row whose
prediction is NaN (not enough history for this lag yet) must be counted as missing, not
silently dropped and forgotten."""

from __future__ import annotations

import pandas as pd
from baselines import evaluate_split, predictions_for_lag


def test_missing_predictions_are_counted_not_silently_dropped() -> None:
    # 3 hours for one zone, lag_hours=2 -- shift(2) leaves rows 0 AND 1 without a value
    # (only row 2 has something 2 positions back), so both must show up as missing, not
    # just vanish from "rows_evaluated" without a trace.
    hours = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    zone_hour = pd.DataFrame(
        {"zone_id": [1, 1, 1], "hour_utc": hours, "pickups": [10, 20, 30]}
    )
    predicted_full = predictions_for_lag(zone_hour, lag_hours=2)

    split_df = zone_hour.copy()
    result = evaluate_split(predicted_full, split_df, zone_ids=[1])

    assert result["rows_missing_prediction"] == 2
    assert result["rows_evaluated"] == 1


def test_no_missing_predictions_when_full_history_available() -> None:
    hours = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    zone_hour = pd.DataFrame(
        {"zone_id": [1] * 5, "hour_utc": hours, "pickups": [10, 20, 30, 40, 50]}
    )
    predicted_full = predictions_for_lag(zone_hour, lag_hours=1)

    # Only evaluate the rows that do have a full lag_1 history available.
    split_df = zone_hour.iloc[1:].copy()
    result = evaluate_split(predicted_full, split_df, zone_ids=[1])

    assert result["rows_missing_prediction"] == 0
    assert result["rows_evaluated"] == 4
