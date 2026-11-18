"""Data access layer for the NYC Taxi Drift Monitoring Dashboard.

READ-ONLY against precomputed reports and raw diagnostics data.
Never recomputes drift statistics or alters underlying files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_params() -> dict[str, Any]:
    params_path = REPO_ROOT / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_selected_zones() -> list[dict[str, Any]]:
    zones_path = REPO_ROOT / "config" / "selected_zones.json"
    with open(zones_path) as f:
        return json.load(f)


@st.cache_data
def load_drift_report() -> pd.DataFrame:
    path = REPO_ROOT / "reports" / "drift_report.parquet"
    df = pd.read_parquet(path)
    # Ensure window_end is UTC timestamp
    if df["window_end"].dt.tz is None:
        df["window_end"] = df["window_end"].dt.tz_localize("UTC")
    return df.sort_values("window_end").reset_index(drop=True)


@st.cache_data
def load_drift_report_by_zone() -> pd.DataFrame:
    path = REPO_ROOT / "reports" / "drift_report_by_zone.parquet"
    df = pd.read_parquet(path)
    if df["window_end"].dt.tz is None:
        df["window_end"] = df["window_end"].dt.tz_localize("UTC")
    return df.sort_values(["window_end", "zone_id"]).reset_index(drop=True)


@st.cache_data
def load_zone_hour_counts() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "zone_hour_counts.parquet"
    df = pd.read_parquet(path)
    if df["hour_utc"].dt.tz is None:
        df["hour_utc"] = df["hour_utc"].dt.tz_localize("UTC")
    return df


def get_available_weeks(zone_report_df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return distinct sorted window_end timestamps available in the weekly zone report."""
    weeks = sorted(zone_report_df["window_end"].unique())
    return [pd.Timestamp(w) for w in weeks]


def get_nearest_drift_row(drift_df: pd.DataFrame, target_time: pd.Timestamp) -> pd.Series:
    """Find the exact or nearest drift_report row for the given target timestamp."""
    exact = drift_df[drift_df["window_end"] == target_time]
    if len(exact) > 0:
        return exact.iloc[0]
    # Fallback to nearest window_end
    diffs = (drift_df["window_end"] - target_time).abs()
    nearest_idx = diffs.idxmin()
    return drift_df.loc[nearest_idx]
