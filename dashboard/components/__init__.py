from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "dist")
_drift_controls = components.declare_component("drift_controls", path=build_dir)


def drift_dashboard_controls(
    weeks_data: list[dict[str, Any]],
    initial_week_index: int = 0,
    initial_zone_id: int | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Render the React drift dashboard controls.

    Includes:
      - Overall Citywide KPI cards & 4 dimension sub-cards
      - Weekly simulation slider & Play auto-advance button
      - Zone-level table/bar chart with exploratory what-if threshold slider
    """
    default_val = {
        "weekIndex": initial_week_index,
        "whatIfThreshold": 20,
        "selectedZoneId": initial_zone_id,
    }
    result = _drift_controls(
        weeks=weeks_data,
        initialWeekIndex=initial_week_index,
        initialZoneId=initial_zone_id,
        key=key,
        default=default_val,
    )
    if result is None:
        return default_val
    return result
