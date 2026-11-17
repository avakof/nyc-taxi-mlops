"""aggregate_and_split.clean_trips is the quarantine step the EDA's findings depend on
(out-of-range timestamps, DST edge cases, placeholder zones). Small synthetic fixtures,
not real downloaded data."""

from __future__ import annotations

import pandas as pd
from aggregate_and_split import PLACEHOLDER_ZONES, clean_trips

WARMUP_START_UTC = pd.Timestamp("2020-01-01", tz="America/New_York").tz_convert("UTC")
TEST_END_UTC = pd.Timestamp("2020-03-01", tz="America/New_York").tz_convert("UTC")


def test_placeholder_zones_excluded() -> None:
    placeholder = next(iter(PLACEHOLDER_ZONES))
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2020-01-15 10:00", "2020-01-15 11:00"]),
            "PULocationID": [placeholder, 100],
        }
    )
    cleaned = clean_trips(trips, WARMUP_START_UTC, TEST_END_UTC)
    assert set(cleaned.PULocationID) == {100}


def test_out_of_window_rows_excluded() -> None:
    # A real TLC defect pattern: a few rows stamped years off from the actual data.
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2003-01-01 00:00", "2020-01-15 10:00", "2021-01-01 00:00"]),
            "PULocationID": [50, 50, 50],
        }
    )
    cleaned = clean_trips(trips, WARMUP_START_UTC, TEST_END_UTC)
    assert len(cleaned) == 1


def test_dst_nonexistent_time_quarantined() -> None:
    # 2020-03-08 02:30 America/New_York does not exist (spring-forward: 02:00 -> 03:00).
    trips = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2020-03-08 02:30", "2020-03-08 10:00"]),
            "PULocationID": [50, 50],
        }
    )
    window_end = pd.Timestamp("2020-04-01", tz="America/New_York").tz_convert("UTC")
    cleaned = clean_trips(trips, WARMUP_START_UTC, window_end)
    assert len(cleaned) == 1
    expected_hour = pd.Timestamp("2020-03-08 10:00", tz="America/New_York").tz_convert("UTC").floor("h")
    assert cleaned.iloc[0]["hour_utc"] == expected_hour


def test_valid_row_hour_utc_computed_correctly() -> None:
    trips = pd.DataFrame({"tpep_pickup_datetime": pd.to_datetime(["2020-01-15 10:30"]), "PULocationID": [50]})
    cleaned = clean_trips(trips, WARMUP_START_UTC, TEST_END_UTC)
    expected = pd.Timestamp("2020-01-15 10:00", tz="America/New_York").tz_convert("UTC")
    assert cleaned.iloc[0]["hour_utc"] == expected
