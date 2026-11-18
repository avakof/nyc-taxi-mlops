"""Batch drift report (ADR-0009). Builds the reference profile from the training period,
then scores every hour of the requested evaluation periods against it: PSI/Cohen's-d on
the deseasonalized pickups residual, PSI/Cohen's-w on zone and hour-of-day mix,
PSI/Cohen's-h on weekend mix, combined into a 2x2 state per dimension plus the
persistence-gated alert flag.

`--periods` defaults to calibration + test because the later natural-drift replay window
(Mar 1 - May 1 2020) isn't aggregated yet -- scripts/aggregate_and_split.py currently
stops at the test period's end. Re-run with `--periods calibration test replay` once
that's extended.

Usage:
    uv run python scripts/compute_drift.py
    uv run python scripts/compute_drift.py --periods calibration test replay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from taxi_mlops.config import load_params
from taxi_mlops.features import split_by_period
from taxi_mlops.monitoring import (
    build_reference_profile,
    compute_drift_report,
    compute_zone_drift_report,
    weekly_evaluation_hours,
)

ROOT = Path(__file__).resolve().parents[1]


def load_selected_zone_ids() -> list[int]:
    zones = json.loads((ROOT / "config" / "selected_zones.json").read_text())
    return [z["zone_id"] for z in zones]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--periods",
        nargs="+",
        default=["calibration", "test"],
        help="params.yaml period names to evaluate hourly.",
    )
    parser.add_argument("--out", default="reports/drift_report.parquet")
    parser.add_argument("--zone-out", default="reports/drift_report_by_zone.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params()
    drift_cfg = params["drift"]
    zone_ids = load_selected_zone_ids()

    zone_hour = pd.read_parquet(
        ROOT / "data" / "processed" / "zone_hour_counts.parquet"
    )
    zone_hour = zone_hour[zone_hour.zone_id.isin(zone_ids)].reset_index(drop=True)

    train = split_by_period(zone_hour, params, "train")
    reference = build_reference_profile(
        train, weekend_floor=drift_cfg["weights"]["weekend_floor"]
    )
    print(
        "reference weights (ADR-0009, eta-squared on the training period): "
        + ", ".join(
            f"{name}={weight:.3f}" for name, weight in reference.weights.items()
        )
    )

    evaluation_period = pd.concat(
        [split_by_period(zone_hour, params, name) for name in args.periods],
        ignore_index=True,
    )
    evaluation_hours = pd.DatetimeIndex(sorted(evaluation_period.hour_utc.unique()))

    # `zone_hour` (not just the evaluation period) is passed as history: each evaluation
    # hour's trailing window may reach back into the training period, which is causal
    # (data from before that hour), never leakage.
    report = compute_drift_report(
        zone_hour, reference, drift_cfg, evaluation_hours=evaluation_hours
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(out_path, index=False)

    n_alerts = (
        int(report["persistent_alert"].sum()) if "persistent_alert" in report else 0
    )
    n_insufficient = int(report["insufficient_data"].sum())
    print(
        f"wrote {out_path.relative_to(ROOT)}: {len(report)} evaluation hours "
        f"({n_insufficient} marked insufficient_data, never silently healthy), "
        f"{n_alerts} persistent alerts"
    )
    if n_alerts:
        first_alert = report.loc[report["persistent_alert"], "window_end"].min()
        print(
            f"first persistent alert at {first_alert} UTC -- use "
            "taxi_mlops.monitoring.zone_breakdown on that window's history slice to see "
            "which zone is driving it"
        )

    # Weekly, per-zone: the question this answers is "how many of 20 zones crossed the
    # retrain threshold this week," for the dashboard's zone view -- not the citywide
    # escalation decision above.
    weekly_hours = weekly_evaluation_hours(
        evaluation_period["hour_utc"], drift_cfg["window_hours"]
    )
    zone_report = compute_zone_drift_report(
        zone_hour,
        reference,
        drift_cfg,
        zone_ids=zone_ids,
        evaluation_hours=weekly_hours,
    )
    zone_out_path = ROOT / args.zone_out
    zone_out_path.parent.mkdir(parents=True, exist_ok=True)
    zone_report.to_parquet(zone_out_path, index=False)

    valid = zone_report[~zone_report["insufficient_data"]]
    if not valid.empty:
        weekly_share = valid.groupby("window_end")["state_pickups_residual"].apply(
            lambda states: float((states >= 2).mean())
        )
        worst_week = weekly_share.idxmax()
        print(
            f"wrote {zone_out_path.relative_to(ROOT)}: {len(zone_ids)} zones x "
            f"{zone_report['window_end'].nunique()} weeks; worst week {worst_week} UTC had "
            f"{weekly_share.loc[worst_week] * 100:.0f}% of zones at warning (state >= 2) or above"
        )


if __name__ == "__main__":
    main()
