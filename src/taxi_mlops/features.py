"""Causal feature construction for the zone-hour forecasting target.

One shared implementation -- used by tuning and training (and later monitoring/serving)
so features are built identically everywhere. See docs/forecast_contract.md.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


def feature_columns(params: dict) -> list[str]:
    lags = params["features"]["lags_hours"]
    windows = params["features"]["rolling_windows_hours"]
    return (
        [f"lag_{lag}" for lag in lags]
        + [f"roll_{window}" for window in windows]
        + ["hour_of_day", "day_of_week", "is_weekend", "zone_id"]
    )


def build_feature_table(zone_hour: pd.DataFrame, params: dict) -> pd.DataFrame:
    """zone_hour: columns zone_id, hour_utc, pickups -- one row per zone-hour, no gaps.

    Returns one row per zone-hour with causal lag/rolling/calendar features plus the
    target (current pickups). Rows without enough history for the longest lag/window
    are dropped (this is exactly what the warmup period exists to cover).
    """
    lags = params["features"]["lags_hours"]
    windows = params["features"]["rolling_windows_hours"]

    df = zone_hour.sort_values(["zone_id", "hour_utc"]).reset_index(drop=True).copy()
    by_zone = df.groupby("zone_id")["pickups"]

    for lag in lags:
        df[f"lag_{lag}"] = by_zone.shift(lag)

    # Rolling means are over the preceding, already-shifted values -- never the current hour.
    df["_shifted_1"] = by_zone.shift(1)
    for window in windows:
        df[f"roll_{window}"] = (
            df.groupby("zone_id")["_shifted_1"]
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    df = df.drop(columns="_shifted_1")

    local_hour = df["hour_utc"].dt.tz_convert(NY)
    df["hour_of_day"] = local_hour.dt.hour
    df["day_of_week"] = local_hour.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["zone_id"] = df["zone_id"].astype("category")
    df = df.rename(columns={"pickups": "target"})

    cols = feature_columns(params)
    df = df.dropna(subset=[c for c in cols if c != "zone_id"])
    return df[["hour_utc", "target"] + cols].reset_index(drop=True)


def split_by_period(
    feature_table: pd.DataFrame, params: dict, period_name: str
) -> pd.DataFrame:
    start = pd.Timestamp(params["periods"][period_name]["start"], tz=NY).tz_convert(
        "UTC"
    )
    end = pd.Timestamp(params["periods"][period_name]["end"], tz=NY).tz_convert("UTC")
    mask = (feature_table.hour_utc >= start) & (feature_table.hour_utc < end)
    return feature_table.loc[mask].reset_index(drop=True)
