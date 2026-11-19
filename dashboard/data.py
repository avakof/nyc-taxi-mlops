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
    df["hour_ny"] = df["hour_utc"].dt.tz_convert("America/New_York")
    return df


@st.cache_data
def load_forecast_vs_actual() -> pd.DataFrame:
    """scripts/compute_forecasts.py's output: the real tuned HGB model (Registry
    `candidate`, not the served `champion` baseline) predicting every causal feature
    row. Starts ~168h after zone_hour_counts.parquet's own start (build_feature_table
    drops rows without full lag/rolling history), so it's shorter than the raw series."""
    path = REPO_ROOT / "reports" / "forecast_vs_actual.parquet"
    df = pd.read_parquet(path)
    if df["hour_utc"].dt.tz is None:
        df["hour_utc"] = df["hour_utc"].dt.tz_localize("UTC")
    df["hour_ny"] = df["hour_utc"].dt.tz_convert("America/New_York")
    return df


WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def modeled_zone_counts(zone_hour: pd.DataFrame, zone_ids: list[int]) -> pd.DataFrame:
    """One column per modeled zone; missing or incomplete counts stay unknown."""
    rows = zone_hour.loc[zone_hour["zone_id"].isin(zone_ids)]
    if "is_complete" in rows.columns:
        rows = rows.assign(pickups=rows["pickups"].where(rows["is_complete"].eq(True)))
    counts = rows.pivot(index="hour_utc", columns="zone_id", values="pickups")
    return counts.reindex(columns=zone_ids).tz_convert("America/New_York")


def weekday_means(
    hourly: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Mean of observed hourly counts per local weekday in the half-open period."""
    period = hourly.loc[(hourly.index >= start) & (hourly.index < end)]
    return period.groupby(period.index.day_name()).mean().reindex(WEEKDAYS)


def get_available_weeks(zone_report_df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return distinct sorted window_end timestamps available in the weekly zone report."""
    weeks = sorted(zone_report_df["window_end"].unique())
    return [pd.Timestamp(w) for w in weeks]


def get_nearest_drift_row(
    drift_df: pd.DataFrame, target_time: pd.Timestamp
) -> pd.Series:
    """Find the exact or nearest drift_report row for the given target timestamp."""
    exact = drift_df[drift_df["window_end"] == target_time]
    if len(exact) > 0:
        return exact.iloc[0]
    # Fallback to nearest window_end
    diffs = (drift_df["window_end"] - target_time).abs()
    nearest_idx = diffs.idxmin()
    return drift_df.loc[nearest_idx]


@st.cache_data
def load_weeks_payload() -> list[dict[str, Any]]:
    """Build and cache the complete weeks payload for the React component.

    Vectorized and cached to eliminate per-rerun iterrows overhead.
    """
    drift_df = load_drift_report()
    zone_drift_df = load_drift_report_by_zone()
    selected_zones_list = load_selected_zones()

    zones_by_id = {z["zone_id"]: z for z in selected_zones_list}
    available_weeks = get_available_weeks(zone_drift_df)

    # Index drift_df by window_end for fast O(1) lookup
    drift_indexed = drift_df.set_index("window_end")

    # Group zone_drift_df by window_end
    grouped_zones = {k: v for k, v in zone_drift_df.groupby("window_end")}

    weeks_payload = []
    for i, w_utc in enumerate(available_weeks):
        w_ny = w_utc.tz_convert("America/New_York")
        if w_utc in drift_indexed.index:
            kpi_match = drift_indexed.loc[w_utc]
            kpi_row = kpi_match.iloc[0] if isinstance(kpi_match, pd.DataFrame) else kpi_match
        else:
            kpi_row = get_nearest_drift_row(drift_df, w_utc)

        zw_df = grouped_zones.get(w_utc)
        zones_data = []
        if zw_df is not None:
            for zr in zw_df.to_dict(orient="records"):
                zid = int(zr["zone_id"])
                zmeta = zones_by_id.get(zid, {})
                zones_data.append(
                    {
                        "zone_id": zid,
                        "name": zmeta.get("name", f"Zone {zid}"),
                        "borough": zmeta.get("borough", "Unknown"),
                        "pickups_train": zmeta.get("pickups_train", 0),
                        "state_pickups_residual": int(zr.get("state_pickups_residual", 0)),
                        "z_pickups_residual": float(zr.get("z_pickups_residual", 0.0)),
                        "cohens_d_pickups_residual": float(
                            zr.get("cohens_d_pickups_residual", 0.0)
                        ),
                        "psi_pickups_residual": float(zr.get("psi_pickups_residual", 0.0)),
                        "persistent_alert": bool(zr.get("persistent_alert", False)),
                    }
                )

        weeks_payload.append(
            {
                "week_index": i,
                "window_end_utc": w_utc.isoformat(),
                "window_end_str": w_ny.strftime("%Y-%m-%d %H:%M EST"),
                "label": f"Week {i + 1} ({w_ny.strftime('%b %d, %Y')})",
                "kpi": {
                    "max_state": int(kpi_row.get("max_state", 0)),
                    "weighted_score": float(kpi_row.get("weighted_score", 0.0)),
                    "persistent_alert": bool(kpi_row.get("persistent_alert", False)),
                    "state_pickups_residual": int(
                        kpi_row.get("state_pickups_residual", 0)
                    ),
                    "psi_pickups_residual": float(
                        kpi_row.get("psi_pickups_residual", 0.0)
                    ),
                    "cohens_d_pickups_residual": float(
                        kpi_row.get("cohens_d_pickups_residual", 0.0)
                    ),
                    "adf_label": str(kpi_row.get("adf_label", "reverting")),
                    "state_zone_mix": int(kpi_row.get("state_zone_mix", 0)),
                    "psi_zone_mix": float(kpi_row.get("psi_zone_mix", 0.0)),
                    "cohens_w_zone_mix": float(kpi_row.get("cohens_w_zone_mix", 0.0)),
                    "state_hour_of_day_mix": int(
                        kpi_row.get("state_hour_of_day_mix", 0)
                    ),
                    "psi_hour_of_day_mix": float(
                        kpi_row.get("psi_hour_of_day_mix", 0.0)
                    ),
                    "cohens_w_hour_of_day_mix": float(
                        kpi_row.get("cohens_w_hour_of_day_mix", 0.0)
                    ),
                    "state_weekend_mix": int(kpi_row.get("state_weekend_mix", 0)),
                    "psi_weekend_mix": float(kpi_row.get("psi_weekend_mix", 0.0)),
                    "cohens_h_weekend_mix": float(
                        kpi_row.get("cohens_h_weekend_mix", 0.0)
                    ),
                },
                "zones": zones_data,
            }
        )
    return weeks_payload
