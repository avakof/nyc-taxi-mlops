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
    load_forecast_vs_actual,
    load_params,
    load_selected_zones,
    load_weeks_payload,
    load_zone_hour_counts,
    modeled_zone_counts,
    weekday_means,
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


def show_missing_weekdays(train: pd.Series, test: pd.Series) -> None:
    for label, means in (("Train", train), ("Test", test)):
        missing = means.index[means.isna()]
        if len(missing):
            st.caption(f"{label} · No data for {', '.join(missing)}.")


@st.cache_data
def get_scope_aggregated_series(
    modeled_zone_ids: tuple[int, ...],
    chosen_zone_ids: tuple[int, ...],
    resample_freq: str,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """NaN-aware: an hour missing/incomplete for any zone in scope stays unknown,
    rather than silently counting as 0 (ADR: modeled-zone totals must be observed)."""
    zone_hour_df = load_zone_hour_counts()
    forecast_df = load_forecast_vs_actual()
    universe = list(chosen_zone_ids) if chosen_zone_ids else list(modeled_zone_ids)

    counts_by_zone = modeled_zone_counts(zone_hour_df, universe).rename_axis("hour_ny")
    scope_hourly = counts_by_zone.sum(axis=1, min_count=len(universe)).rename("pickups")

    agg_series = scope_hourly.resample(resample_freq).sum().reset_index()

    forecast_scope_df = forecast_df[forecast_df["zone_id"].isin(universe)]
    forecast_hourly = (
        forecast_scope_df.groupby("hour_ny")["forecasted_pickups"].sum().reset_index()
    )
    forecast_agg_series = (
        forecast_hourly.set_index("hour_ny")
        .resample(resample_freq)["forecasted_pickups"]
        .sum()
        .reset_index()
    )
    return scope_hourly, agg_series, forecast_agg_series


@st.cache_data
def get_citywide_dow_figure(
    modeled_zone_ids: tuple[int, ...],
    train_start_iso: str,
    train_end_iso: str,
    test_start_iso: str,
    test_end_iso: str,
) -> tuple[go.Figure, pd.Series, pd.Series]:
    """Fixed reference scope (always every modeled zone) -- a modeled-area total
    needs an observed count from every selected zone, else the hour is unknown."""
    zone_hour_df = load_zone_hour_counts()
    counts_by_zone = modeled_zone_counts(zone_hour_df, list(modeled_zone_ids)).rename_axis(
        "hour_ny"
    )
    city_hourly = counts_by_zone.sum(axis=1, min_count=len(modeled_zone_ids))

    train_start = pd.Timestamp(train_start_iso)
    train_end = pd.Timestamp(train_end_iso)
    test_start = pd.Timestamp(test_start_iso)
    test_end = pd.Timestamp(test_end_iso)

    city_train_means = weekday_means(city_hourly, train_start, train_end)
    city_test_means = weekday_means(city_hourly, test_start, test_end)

    fig_city = go.Figure()
    fig_city.add_trace(
        go.Bar(
            x=city_train_means.index,
            y=city_train_means,
            name="Train Period",
            marker_color=KNICKS_BLUE,
            hovertemplate="%{x}<br>Citywide Train Mean: %{y:,.0f} pickups/hr<extra></extra>",
        )
    )
    fig_city.add_trace(
        go.Bar(
            x=city_test_means.index,
            y=city_test_means,
            name="Test Period",
            marker_color=KNICKS_ORANGE,
            hovertemplate="%{x}<br>Citywide Test Mean: %{y:,.0f} pickups/hr<extra></extra>",
        )
    )

    fig_city.update_layout(
        title={
            "text": f"Average hourly pickups by weekday ({len(modeled_zone_ids)} modeled zones)",
            "font": {"size": 14, "color": KNICKS_BLUE},
        },
        barmode="group",
        xaxis_title="Day of Week (Local NY Time)",
        yaxis_title="Citywide Avg Pickups / Hour",
        template="plotly_white",
        height=340,
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig_city, city_train_means, city_test_means


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
    weeks_payload = load_weeks_payload()

    # Zone mapping dictionaries
    zones_by_id = {z["zone_id"]: z for z in selected_zones_list}
    # Default zone: highest pickups_train
    default_zone = max(selected_zones_list, key=lambda z: z.get("pickups_train", 0))
    default_zone_id = default_zone["zone_id"]

    if not weeks_payload:
        st.error(
            "No weekly evaluation windows found in reports/drift_report_by_zone.parquet."
        )
        return

    available_weeks_utc = [
        pd.Timestamp(w["window_end_utc"]) for w in weeks_payload
    ]

    # 2. Mount Custom React Component
    ctrl_state = drift_dashboard_controls(
        weeks_data=weeks_payload,
        initial_week_index=0,
        initial_zone_id=default_zone_id,
        key="drift_controls_component",
    )

    active_week_idx = int(ctrl_state.get("weekIndex", 0))
    if active_week_idx < 0 or active_week_idx >= len(available_weeks_utc):
        active_week_idx = 0
    active_week_utc = available_weeks_utc[active_week_idx]
    active_week_ny = active_week_utc.tz_convert("America/New_York")

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

    # Zone filter: multiselect, default empty -- "overall" (all 20 zones, summed).
    zone_options = [z["zone_id"] for z in selected_zones_list]
    zone_labels = {
        z[
            "zone_id"
        ]: f"{z['name']} ({z['borough']}) — Train: {z.get('pickups_train', 0):,} pickups"
        for z in selected_zones_list
    }
    chosen_zone_ids = st.multiselect(
        "Filter Zones (default: overall, all 20 zones summed):",
        options=zone_options,
        default=[],
        format_func=lambda zid: zone_labels.get(zid, f"Zone {zid}"),
    )
    if chosen_zone_ids:
        scope_label = (
            ", ".join(
                zones_by_id.get(z, {}).get("name", f"Zone {z}") for z in chosen_zone_ids
            )
            if len(chosen_zone_ids) <= 3
            else f"{len(chosen_zone_ids)} Selected Zones"
        )
    else:
        scope_label = "All 20 Zones (Overall)"

    # Periods from params.yaml
    periods = params.get("periods", {})

    def _period(
        name: str, default_start: str, default_end: str
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        bounds = periods.get(name, {})
        start = pd.Timestamp(bounds.get("start", default_start), tz="America/New_York")
        end = pd.Timestamp(bounds.get("end", default_end), tz="America/New_York")
        return start, end

    train_start, train_end = _period("train", "2020-01-08", "2020-02-03")
    calib_start, calib_end = _period("calibration", "2020-02-03", "2020-02-17")
    test_start, test_end = _period("test", "2020-02-17", "2020-03-01")
    replay_start, replay_end = _period("replay", "2020-03-01", "2020-05-01")

    # Diagnostic 1: Raw Demand Time Series -- Daily by default, Weekly optional.
    agg_choice = st.radio(
        "Aggregate demand by:",
        options=["Daily", "Weekly"],
        index=0,
        horizontal=True,
    )
    resample_freq = "D" if agg_choice == "Daily" else "W"
    scope_hourly, agg_series, forecast_agg_series = get_scope_aggregated_series(
        tuple(zone_options),
        tuple(chosen_zone_ids),
        resample_freq,
    )

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=agg_series["hour_ny"],
            y=agg_series["pickups"],
            mode="lines",
            name="Live Pickups",
            line={"color": KNICKS_BLUE, "width": 1.5 if agg_choice == "Daily" else 2.5},
            hovertemplate=f"{agg_choice} starting: "
            + "%{x|%Y-%m-%d}<br>Live Pickups: %{y:,}<extra></extra>",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=forecast_agg_series["hour_ny"],
            y=forecast_agg_series["forecasted_pickups"],
            mode="lines",
            name="Forecasted Pickups (HGB candidate v3)",
            line={
                "color": KNICKS_ORANGE,
                "width": 1.5 if agg_choice == "Daily" else 2.5,
                "dash": "dot",
            },
            hovertemplate=f"{agg_choice} starting: "
            + "%{x|%Y-%m-%d}<br>Forecasted Pickups: %{y:,.0f}<extra></extra>",
        )
    )

    # Shaded periods -- Replay (Mar 1 - May 1) is the real natural-drift window now that
    # aggregate_and_split.py covers it, not just the healthy warmup/train/calibration/test.
    period_shading = [
        ("Warmup", "2020-01-01", train_start, "rgba(229, 231, 235, 0.4)"),
        ("Train Period", train_start, train_end, "rgba(0, 107, 182, 0.10)"),
        ("Calibration Period", calib_start, calib_end, "rgba(190, 192, 194, 0.20)"),
        ("Test Period", test_start, test_end, "rgba(245, 132, 38, 0.12)"),
        (
            "Replay Period (Natural Drift)",
            replay_start,
            replay_end,
            "rgba(220, 38, 38, 0.08)",
        ),
    ]

    for label, start_dt, end_dt, color in period_shading:
        start_ts = (
            pd.Timestamp(start_dt, tz="America/New_York")
            if isinstance(start_dt, str)
            else start_dt
        )
        end_ts = (
            pd.Timestamp(end_dt, tz="America/New_York")
            if isinstance(end_dt, str)
            else end_dt
        )
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

    fig_line.update_layout(
        title={
            "text": f"Forecasted versus Live Pickups ({agg_choice}): {scope_label}",
            "font": {"size": 15, "color": KNICKS_BLUE},
        },
        xaxis_title="Date (America/New_York)",
        yaxis_title=f"Pickup Volume ({agg_choice} Total)",
        template="plotly_white",
        height=380,
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        hovermode="x unified",
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # Diagnostic 2: Grouped Bar Chart Comparing Day-of-Week (Train vs Test)
    col1, col2 = st.columns([1, 1])

    with col1:
        # Respects the zone filter above -- defaults to the same "All 20 Zones" scope as
        # the line chart, so this and the fixed citywide reference in col2 start identical.
        train_means = weekday_means(scope_hourly, train_start, train_end)
        test_means = weekday_means(scope_hourly, test_start, test_end)

        fig_dow = go.Figure()
        fig_dow.add_trace(
            go.Bar(
                x=train_means.index,
                y=train_means,
                name="Train Period (Jan 08 - Feb 03)",
                marker_color=KNICKS_BLUE,
                hovertemplate="%{x}<br>Train Mean: %{y:.1f} pickups/hr<extra></extra>",
            )
        )
        fig_dow.add_trace(
            go.Bar(
                x=test_means.index,
                y=test_means,
                name="Test Period (Feb 17 - Mar 01)",
                marker_color=KNICKS_ORANGE,
                hovertemplate="%{x}<br>Test Mean: %{y:.1f} pickups/hr<extra></extra>",
            )
        )

        fig_dow.update_layout(
            title={
                "text": f"Day-of-Week Diurnal Shift: {scope_label}",
                "font": {"size": 14, "color": KNICKS_BLUE},
            },
            barmode="group",
            xaxis_title="Day of Week (Local NY Time)",
            yaxis_title="Average Pickups / Hour",
            template="plotly_white",
            height=340,
            margin={"l": 40, "r": 20, "t": 45, "b": 40},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        show_missing_weekdays(train_means, test_means)

    with col2:
        fig_city, city_train_means, city_test_means = get_citywide_dow_figure(
            tuple(zone_options),
            train_start.isoformat(),
            train_end.isoformat(),
            test_start.isoformat(),
            test_end.isoformat(),
        )
        st.plotly_chart(fig_city, use_container_width=True)
        show_missing_weekdays(city_train_means, city_test_means)


if __name__ == "__main__":
    main()
