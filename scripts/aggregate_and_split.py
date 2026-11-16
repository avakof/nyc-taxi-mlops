"""Aggregate raw trips to zone-hour pickup counts, then split by target-hour timestamp
into train/calibration/test. Time-based split only -- this is time series, never random.

Usage:
    uv run python scripts/aggregate_and_split.py
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from taxi_mlops.config import load_params

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
NY = ZoneInfo("America/New_York")

# "Unknown" / "N/A" in the TLC zone lookup -- not real geography, excluded from aggregation.
PLACEHOLDER_ZONES = {264, 265}


def needed_months(params: dict) -> list[str]:
    periods = params["periods"]
    start = pd.Timestamp(periods["warmup"]["start"])
    end = pd.Timestamp(periods["test"]["end"]) - pd.Timedelta(days=1)
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def load_raw_pickups(params: dict) -> pd.DataFrame:
    taxi_type = params["data"]["taxi_type"]
    periods = params["periods"]
    warmup_start_utc = pd.Timestamp(periods["warmup"]["start"], tz=NY).tz_convert("UTC")
    test_end_utc = pd.Timestamp(periods["test"]["end"], tz=NY).tz_convert("UTC")

    frames = [
        pd.read_parquet(RAW_DIR / f"{taxi_type}_tripdata_{month}.parquet", columns=["tpep_pickup_datetime", "PULocationID"])
        for month in needed_months(params)
    ]
    trips = pd.concat(frames, ignore_index=True)

    before = len(trips)
    trips = trips[~trips.PULocationID.isin(PLACEHOLDER_ZONES)]
    dropped_placeholder = before - len(trips)

    # Naive source timestamps are NYC local wall time (documented assumption). Quarantine
    # anything that doesn't localize cleanly: nonexistent (spring-forward gap, e.g. 2020-03-08)
    # or ambiguous (fall-back) local times, rather than guessing.
    local = trips.tpep_pickup_datetime.dt.tz_localize(NY, nonexistent="NaT", ambiguous="NaT")
    dst_quarantined = int(local.isna().sum())
    trips = trips.assign(pickup_utc=local.dt.tz_convert("UTC")).dropna(subset=["pickup_utc"])

    # Quarantine rows whose pickup time falls outside the covered window -- TLC files contain
    # a small number of data-entry errors (timestamps years off), found during EDA.
    hour_utc = trips.pickup_utc.dt.floor("h")
    in_window = (hour_utc >= warmup_start_utc) & (hour_utc < test_end_utc)
    out_of_range = int((~in_window).sum())
    trips = trips.loc[in_window].assign(hour_utc=hour_utc[in_window])

    print(
        f"dropped {dropped_placeholder} placeholder-zone rows, {dst_quarantined} DST-invalid rows, "
        f"{out_of_range} out-of-window rows; {len(trips)} rows remain"
    )
    return trips[["PULocationID", "hour_utc"]]


def to_zone_hour_counts(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    periods = params["periods"]
    warmup_start_utc = pd.Timestamp(periods["warmup"]["start"], tz=NY).tz_convert("UTC")
    test_end_utc = pd.Timestamp(periods["test"]["end"], tz=NY).tz_convert("UTC")

    zones = pd.read_csv(ROOT / "data" / "external" / "taxi_zone_lookup.csv")
    zone_ids = sorted(set(zones.LocationID) - PLACEHOLDER_ZONES)
    all_hours = pd.date_range(warmup_start_utc, test_end_utc, freq="h", inclusive="left")

    counts = trips.groupby(["PULocationID", "hour_utc"]).size()
    full_index = pd.MultiIndex.from_product([zone_ids, all_hours], names=["zone_id", "hour_utc"])
    zone_hour = counts.reindex(full_index, fill_value=0).rename("pickups").reset_index()
    return zone_hour


def split_by_period(zone_hour: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "calibration", "test"):
        start = pd.Timestamp(params["periods"][name]["start"], tz=NY).tz_convert("UTC")
        end = pd.Timestamp(params["periods"][name]["end"], tz=NY).tz_convert("UTC")
        mask = (zone_hour.hour_utc >= start) & (zone_hour.hour_utc < end)
        out[name] = zone_hour.loc[mask].reset_index(drop=True)
    return out


def main() -> None:
    params = load_params()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    trips = load_raw_pickups(params)
    zone_hour = to_zone_hour_counts(trips, params)
    zone_hour.to_parquet(PROCESSED_DIR / "zone_hour_counts.parquet", index=False)
    print(
        f"wrote canonical aggregate: {len(zone_hour)} rows "
        f"({zone_hour.zone_id.nunique()} zones x {zone_hour.hour_utc.nunique()} hours), "
        "covers warmup through test (needed as lag history for early train rows)"
    )

    for name, df in split_by_period(zone_hour, params).items():
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"wrote {path.relative_to(ROOT)}: {len(df)} rows, {df.hour_utc.min()} to {df.hour_utc.max()}")


if __name__ == "__main__":
    main()
