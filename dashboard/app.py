"""NYC Taxi Demand — Drift Monitoring Dashboard

Streamlit application providing a READ-ONLY visualization of precomputed
drift statistics and raw ground-truth diagnostic time series.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit's `streamlit run dashboard/app.py` puts this file's own directory
# (dashboard/) on sys.path, not the repo root -- without this, `from dashboard...`
# below raises ModuleNotFoundError no matter what directory streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import drift_dashboard_controls
from dashboard.data import (
    get_available_weeks,
    get_nearest_drift_row,
    load_drift_report,
    load_drift_report_by_zone,
    load_params,
    load_selected_zones,
    load_zone_hour_counts,
)

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Drift Monitor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# New York Knicks branding CSS
KNICKS_BLUE = "#006BB6"
KNICKS_ORANGE = "#F58426"
KNICKS_SILVER = "#BEC0C2"

st.markdown(
    f"""
    <style>
        .main-header {{
            background: linear-gradient(90deg, {KNICKS_BLUE} 0%, #004D84 100%);
            padding: 16px 24px;
            border-radius: 10px;
            color: #FFFFFF;
            margin-bottom: 20px;
            border-bottom: 4px solid {KNICKS_ORANGE};
        }}
        .main-header h1 {{
            margin: 0;
            font-size: 26px;
            font-weight: 800;
            color: #FFFFFF;
        }}
        .main-header p {{
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #E0E7FF;
        }}
        .diagnostics-container {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .diagnostics-title {{
            font-size: 16px;
            font-weight: 800;
            color: {KNICKS_BLUE};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .diagnostics-sub {{
            font-size: 12px;
            color: #6B7280;
            margin-bottom: 16px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1>NYC Taxi Demand — Drift Monitoring</h1>
                    <p>HistGradientBoostingRegressor • 20 Modeled NYC Zones • Trailing 168h Multi-Dimensional Drift Detection</p>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;">
                        READ-ONLY SIMULATION
                    </span>
                    <div style="font-size: 11px; color: #BFDBFE; margin-top: 4px;">Static Parquet Snapshot (Pre-COVID Healthy Window)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Load precomputed datasets and configuration
    params = load_params()
    selected_zones_list = load_selected_zones()
    drift_df = load_drift_report()
    zone_drift_df = load_drift_report_by_zone()
    zone_hour_df = load_zone_hour_counts()

    # Zone mapping dictionaries
    zones_by_id = {z["zone_id"]: z for z in selected_zones_list}
    # Default zone: highest pickups_train
    default_zone = max(selected_zones_list, key=lambda z: z.get("pickups_train", 0))
    default_zone_id = default_zone["zone_id"]

    # Available simulation weeks from weekly zone report
    available_weeks = get_available_weeks(zone_drift_df)
    if not available_weeks:
        st.error("No weekly evaluation windows found in reports/drift_report_by_zone.parquet.")
        return

    # 2. Build payload for the React component
    weeks_payload = []
    for i, w_utc in enumerate(available_weeks):
        w_ny = w_utc.tz_convert("America/New_York")
        # Nearest / exact citywide row
        kpi_row = get_nearest_drift_row(drift_df, w_utc)

        # Zone rows for this week
        zw_df = zone_drift_df[zone_drift_df["window_end"] == w_utc]
        zones_data = []
        for _, zr in zw_df.iterrows():
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
                    "cohens_d_pickups_residual": float(zr.get("cohens_d_pickups_residual", 0.0)),
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
                    "state_pickups_residual": int(kpi_row.get("state_pickups_residual", 0)),
                    "psi_pickups_residual": float(kpi_row.get("psi_pickups_residual", 0.0)),
                    "cohens_d_pickups_residual": float(kpi_row.get("cohens_d_pickups_residual", 0.0)),
                    "adf_label": str(kpi_row.get("adf_label", "reverting")),
                    "state_zone_mix": int(kpi_row.get("state_zone_mix", 0)),
                    "psi_zone_mix": float(kpi_row.get("psi_zone_mix", 0.0)),
                    "cohens_w_zone_mix": float(kpi_row.get("cohens_w_zone_mix", 0.0)),
                    "state_hour_of_day_mix": int(kpi_row.get("state_hour_of_day_mix", 0)),
                    "psi_hour_of_day_mix": float(kpi_row.get("psi_hour_of_day_mix", 0.0)),
                    "cohens_w_hour_of_day_mix": float(kpi_row.get("cohens_w_hour_of_day_mix", 0.0)),
                    "state_weekend_mix": int(kpi_row.get("state_weekend_mix", 0)),
                    "psi_weekend_mix": float(kpi_row.get("psi_weekend_mix", 0.0)),
                    "cohens_h_weekend_mix": float(kpi_row.get("cohens_h_weekend_mix", 0.0)),
                },
                "zones": zones_data,
            }
        )

    # 3. Mount Custom React Component
    ctrl_state = drift_dashboard_controls(
        weeks_data=weeks_payload,
        initial_week_index=0,
        initial_zone_id=default_zone_id,
        key="drift_controls_component",
    )

    active_week_idx = int(ctrl_state.get("weekIndex", 0))
    if active_week_idx < 0 or active_week_idx >= len(available_weeks):
        active_week_idx = 0
    active_week_utc = available_weeks[active_week_idx]
    active_week_ny = active_week_utc.tz_convert("America/New_York")

    # Selected zone from React component or fallback
    clicked_zone_id = ctrl_state.get("selectedZoneId")
    if clicked_zone_id is not None and int(clicked_zone_id) in zones_by_id:
        active_zone_id = int(clicked_zone_id)
    else:
        active_zone_id = default_zone_id

    # 4. Diagnostics Section (Ground Truth Raw Time Series)
    st.markdown(
        """
        <div class="diagnostics-container">
            <div class="diagnostics-title">Ground-Truth Diagnostics (Raw Activity Series)</div>
            <div class="diagnostics-sub">
                Raw pickup counts from <code>data/processed/zone_hour_counts.parquet</code> and period boundaries from <code>params.yaml</code>.
                This is diagnostic raw ground truth, completely decoupled from drift detection scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Zone selection dropdown
    zone_options = [z["zone_id"] for z in selected_zones_list]
    zone_labels = {
        z["zone_id"]: f"{z['name']} ({z['borough']}) — Train: {z.get('pickups_train', 0):,} pickups"
        for z in selected_zones_list
    }

    selected_idx = zone_options.index(active_zone_id) if active_zone_id in zone_options else 0
    chosen_zone_id = st.selectbox(
        "Select Zone for Diagnostics:",
        options=zone_options,
        index=selected_idx,
        format_func=lambda zid: zone_labels.get(zid, f"Zone {zid}"),
    )

    # Filter raw pickups for chosen zone
    zone_counts = zone_hour_df[zone_hour_df["zone_id"] == chosen_zone_id].copy()
    zone_counts["hour_ny"] = zone_counts["hour_utc"].dt.tz_convert("America/New_York")
    zone_counts = zone_counts.sort_values("hour_ny").reset_index(drop=True)

    # Periods from params.yaml
    periods = params.get("periods", {})
    train_start = pd.Timestamp(periods.get("train", {}).get("start", "2020-01-08"), tz="America/New_York")
    train_end = pd.Timestamp(periods.get("train", {}).get("end", "2020-02-03"), tz="America/New_York")
    calib_start = pd.Timestamp(periods.get("calibration", {}).get("start", "2020-02-03"), tz="America/New_York")
    calib_end = pd.Timestamp(periods.get("calibration", {}).get("end", "2020-02-17"), tz="America/New_York")
    test_start = pd.Timestamp(periods.get("test", {}).get("start", "2020-02-17"), tz="America/New_York")
    test_end = pd.Timestamp(periods.get("test", {}).get("end", "2020-03-01"), tz="America/New_York")

    # Diagnostic 1: Raw Pickups Time Series Line Chart
    fig_line = go.Figure()

    fig_line.add_trace(
        go.Scatter(
            x=zone_counts["hour_ny"],
            y=zone_counts["pickups"],
            mode="lines",
            name="Hourly Pickups",
            line={"color": KNICKS_BLUE, "width": 1.5},
            hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Pickups: %{y:,}<extra></extra>",
        )
    )

    # Shaded periods
    period_shading = [
        ("Warmup", "2020-01-01", train_start, "rgba(229, 231, 235, 0.4)"),
        ("Train Period", train_start, train_end, "rgba(0, 107, 182, 0.10)"),
        ("Calibration Period", calib_start, calib_end, "rgba(190, 192, 194, 0.20)"),
        ("Test Period", test_start, test_end, "rgba(245, 132, 38, 0.12)"),
    ]

    for label, start_dt, end_dt, color in period_shading:
        start_ts = pd.Timestamp(start_dt, tz="America/New_York") if isinstance(start_dt, str) else start_dt
        end_ts = pd.Timestamp(end_dt, tz="America/New_York") if isinstance(end_dt, str) else end_dt
        fig_line.add_vrect(
            x0=start_ts,
            x1=end_ts,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=f"<b>{label}</b>",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#4B5563",
        )

    # "You are here" vertical marker matching active simulation week
    fig_line.add_vline(
        x=active_week_ny,
        line_width=2.5,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="<b>📍 Current Evaluation (You are here)</b>",
        annotation_position="bottom right",
        annotation_font_size=11,
        annotation_font_color="#DC2626",
    )

    chosen_meta = zones_by_id.get(chosen_zone_id, {})
    fig_line.update_layout(
        title={
            "text": f"Raw Hourly Demand: {chosen_meta.get('name', f'Zone {chosen_zone_id}')} ({chosen_meta.get('borough', '')})",
            "font": {"size": 15, "color": KNICKS_BLUE},
        },
        xaxis_title="Date / Time (America/New_York)",
        yaxis_title="Pickup Volume",
        template="plotly_white",
        height=380,
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        hovermode="x unified",
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # Diagnostic 2: Grouped Bar Chart Comparing Day-of-Week (Train vs Test)
    col1, col2 = st.columns([1, 1])

    with col1:
        # Zone-specific day-of-week
        train_slice = zone_counts[(zone_counts["hour_ny"] >= train_start) & (zone_counts["hour_ny"] < train_end)].copy()
        test_slice = zone_counts[(zone_counts["hour_ny"] >= test_start) & (zone_counts["hour_ny"] < test_end)].copy()

        train_slice["day_of_week"] = train_slice["hour_ny"].dt.day_name()
        test_slice["day_of_week"] = test_slice["hour_ny"].dt.day_name()

        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        train_means = train_slice.groupby("day_of_week")["pickups"].mean().reindex(dow_order).fillna(0)
        test_means = test_slice.groupby("day_of_week")["pickups"].mean().reindex(dow_order).fillna(0)

        fig_dow = go.Figure()
        fig_dow.add_trace(
            go.Bar(
                x=dow_order,
                y=train_means,
                name="Train Period (Jan 08 - Feb 03)",
                marker_color=KNICKS_BLUE,
                hovertemplate="%{x}<br>Train Mean: %{y:.1f} pickups/hr<extra></extra>",
            )
        )
        fig_dow.add_trace(
            go.Bar(
                x=dow_order,
                y=test_means,
                name="Test Period (Feb 17 - Mar 01)",
                marker_color=KNICKS_ORANGE,
                hovertemplate="%{x}<br>Test Mean: %{y:.1f} pickups/hr<extra></extra>",
            )
        )

        fig_dow.update_layout(
            title={
                "text": f"Day-of-Week Diurnal Shift: {chosen_meta.get('name', f'Zone {chosen_zone_id}')}",
                "font": {"size": 14, "color": KNICKS_BLUE},
            },
            barmode="group",
            xaxis_title="Day of Week (Local NY Time)",
            yaxis_title="Average Pickups / Hour",
            template="plotly_white",
            height=340,
            margin={"l": 40, "r": 20, "t": 45, "b": 40},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col2:
        # Citywide aggregate day-of-week
        city_counts = zone_hour_df.copy()
        city_counts["hour_ny"] = city_counts["hour_utc"].dt.tz_convert("America/New_York")
        city_hourly = city_counts.groupby("hour_ny")["pickups"].sum().reset_index()

        city_train = city_hourly[(city_hourly["hour_ny"] >= train_start) & (city_hourly["hour_ny"] < train_end)].copy()
        city_test = city_hourly[(city_hourly["hour_ny"] >= test_start) & (city_hourly["hour_ny"] < test_end)].copy()

        city_train["day_of_week"] = city_train["hour_ny"].dt.day_name()
        city_test["day_of_week"] = city_test["hour_ny"].dt.day_name()

        city_train_means = city_train.groupby("day_of_week")["pickups"].mean().reindex(dow_order).fillna(0)
        city_test_means = city_test.groupby("day_of_week")["pickups"].mean().reindex(dow_order).fillna(0)

        fig_city = go.Figure()
        fig_city.add_trace(
            go.Bar(
                x=dow_order,
                y=city_train_means,
                name="Train Period",
                marker_color=KNICKS_BLUE,
                hovertemplate="%{x}<br>Citywide Train Mean: %{y:,.0f} pickups/hr<extra></extra>",
            )
        )
        fig_city.add_trace(
            go.Bar(
                x=dow_order,
                y=city_test_means,
                name="Test Period",
                marker_color=KNICKS_ORANGE,
                hovertemplate="%{x}<br>Citywide Test Mean: %{y:,.0f} pickups/hr<extra></extra>",
            )
        )

        fig_city.update_layout(
            title={
                "text": "Citywide Aggregate Day-of-Week Diurnal Shift (All 20 Zones)",
                "font": {"size": 14, "color": KNICKS_BLUE},
            },
            barmode="group",
            xaxis_title="Day of Week (Local NY Time)",
            yaxis_title="Citywide Avg Pickups / Hour",
            template="plotly_white",
            height=340,
            margin={"l": 40, "r": 20, "t": 45, "b": 40},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        st.plotly_chart(fig_city, use_container_width=True)


if __name__ == "__main__":
    main()
