"""Data-layer drift detection (ADR-0009): PSI, Cohen's d/h/w, a deseasonalized rolling
z-score, and an ADF confirmatory check, combined into a per-dimension 2x2 escalation
state. Pure functions over DataFrames/arrays -- no I/O -- so a batch report script and a
future live worker share one implementation, same shape as features.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

NY = ZoneInfo("America/New_York")

# ADR-0009: pickups_residual replaces the 8 correlated lag/rolling features; zone/hour/
# weekend mix cover the remaining calendar/zone structure.
DIMENSIONS = ("pickups_residual", "zone_mix", "hour_of_day_mix", "weekend_mix")


# --- Calendar + reference profile --------------------------------------------------


def _with_calendar(zone_hour: pd.DataFrame) -> pd.DataFrame:
    df = zone_hour.copy()
    local = df["hour_utc"].dt.tz_convert(NY)
    df["hour_of_day"] = local.dt.hour
    df["is_weekend"] = (local.dt.day_of_week >= 5).astype(int)
    return df


def _cell_residuals(df: pd.DataFrame, cell_mean: pd.Series) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([df.zone_id, df.hour_of_day, df.is_weekend])
    means = cell_mean.reindex(keys).to_numpy()
    return df["pickups"].to_numpy(dtype=float) - means


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Share of `values`' variance explained by group means (one-way ANOVA effect size)."""
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    agg = (
        pd.DataFrame({"value": values.to_numpy(), "group": groups.to_numpy()})
        .groupby("group")["value"]
        .agg(["mean", "count"])
    )
    ss_between = float((agg["count"] * (agg["mean"] - grand_mean) ** 2).sum())
    return ss_between / ss_total


def dimension_weights(
    reference: pd.DataFrame, *, weekend_floor: float = 0.05
) -> dict[str, float]:
    """ADR-0009: weights are the reference-period variance share each grouping explains in
    `pickups`, with pickups_residual taking the unexplained remainder -- not the model's
    SHAP importances, which measure predictive reliance, not monitoring relevance."""
    df = _with_calendar(reference)
    eta_zone = eta_squared(df["pickups"], df["zone_id"])
    eta_hour = eta_squared(df["pickups"], df["hour_of_day"])
    eta_weekend = eta_squared(df["pickups"], df["is_weekend"])
    explained = min(eta_zone + eta_hour + eta_weekend, 1.0)
    weights = {
        "pickups_residual": 1.0 - explained,
        "zone_mix": eta_zone,
        "hour_of_day_mix": eta_hour,
        "weekend_mix": eta_weekend,
    }
    if weights["weekend_mix"] < weekend_floor:
        deficit = weekend_floor - weights["weekend_mix"]
        others = ("pickups_residual", "zone_mix", "hour_of_day_mix")
        total_others = sum(weights[key] for key in others)
        for key in others:
            share = weights[key] / total_others if total_others > 0 else 1 / len(others)
            weights[key] = max(weights[key] - deficit * share, 0.0)
        weights["weekend_mix"] = weekend_floor
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        # Degenerate reference (near-constant residual): widen artificially so PSI stays
        # computable instead of raising on a single-bin histogram.
        spread = max(abs(float(np.median(finite))), 1.0)
        edges = np.array(
            [finite.min() - spread, finite.min(), finite.max(), finite.max() + spread]
        )
    edges = edges.astype(float)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


@dataclass
class ReferenceProfile:
    cell_mean: pd.Series
    residual: np.ndarray
    residual_std: float
    residual_bin_edges: np.ndarray
    pickups: np.ndarray
    zone_share: pd.Series
    hour_share: pd.Series
    weekend_share: float
    weights: dict[str, float]
    residual_by_zone: dict[int, np.ndarray]
    pickups_by_zone: dict[int, np.ndarray]
    residual_std_by_zone: dict[int, float]
    residual_bin_edges_by_zone: dict[int, np.ndarray]


def build_reference_profile(
    reference: pd.DataFrame, *, n_psi_bins: int = 10, weekend_floor: float = 0.05
) -> ReferenceProfile:
    """reference: zone_id, hour_utc (UTC, tz-aware), pickups -- the training-period
    zone-hour table. Everything a later window is compared against is computed here
    once, so a drift run never refits its own yardstick mid-report."""
    df = _with_calendar(reference)
    cell_mean = df.groupby(["zone_id", "hour_of_day", "is_weekend"])["pickups"].mean()
    residual = _cell_residuals(df, cell_mean)
    pickups = df["pickups"].to_numpy(dtype=float)

    zone_share = df.groupby("zone_id")["pickups"].sum()
    zone_share = zone_share / zone_share.sum()
    hour_share = df.groupby("hour_of_day")["pickups"].sum()
    hour_share = hour_share / hour_share.sum()
    weekend_share = float(
        df.loc[df.is_weekend == 1, "pickups"].sum() / df["pickups"].sum()
    )

    # Per-zone reference samples: zones have very different baseline volumes (training
    # pickups range from ~110k to ~255k across the 20 selected zones), so a per-zone
    # comparison must never fall back to the pooled, all-zone reference -- that would
    # read every zone's own characteristic level as "drift" against the cross-zone mix.
    zone_values = df["zone_id"].to_numpy()
    residual_by_zone: dict[int, np.ndarray] = {}
    pickups_by_zone: dict[int, np.ndarray] = {}
    residual_std_by_zone: dict[int, float] = {}
    residual_bin_edges_by_zone: dict[int, np.ndarray] = {}
    for zone_id in pd.unique(zone_values):
        mask = zone_values == zone_id
        zone_residual = residual[mask]
        residual_by_zone[zone_id] = zone_residual
        pickups_by_zone[zone_id] = pickups[mask]
        residual_std_by_zone[zone_id] = float(np.nanstd(zone_residual, ddof=1))
        residual_bin_edges_by_zone[zone_id] = quantile_bin_edges(
            zone_residual, n_psi_bins
        )

    return ReferenceProfile(
        cell_mean=cell_mean,
        residual=residual,
        residual_std=float(np.nanstd(residual, ddof=1)),
        residual_bin_edges=quantile_bin_edges(residual, n_psi_bins),
        pickups=pickups,
        zone_share=zone_share,
        hour_share=hour_share,
        weekend_share=weekend_share,
        weights=dimension_weights(reference, weekend_floor=weekend_floor),
        residual_by_zone=residual_by_zone,
        pickups_by_zone=pickups_by_zone,
        residual_std_by_zone=residual_std_by_zone,
        residual_bin_edges_by_zone=residual_bin_edges_by_zone,
    )


# --- Detection statistics ------------------------------------------------------------


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    bin_edges: np.ndarray,
    epsilon: float = 1e-4,
) -> float | None:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return None
    # Open-ended outer bins: a large shift must still land somewhere, never fall outside
    # every bin and leave a count vector that sums to zero.
    edges = bin_edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(
    reference_share: pd.Series, current_share: pd.Series, epsilon: float = 1e-4
) -> float:
    index = reference_share.index
    ref = reference_share.reindex(index).fillna(0.0).clip(lower=epsilon)
    ref = ref / ref.sum()
    cur = current_share.reindex(index).fillna(0.0).clip(lower=epsilon)
    cur = cur / cur.sum()
    return float(((cur - ref) * np.log(cur / ref)).sum())


def deseasonalized_zscore(
    residuals: np.ndarray, reference_residual_std: float
) -> float | None:
    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)
    if n == 0 or reference_residual_std == 0:
        return None
    return float(residuals.mean() / (reference_residual_std / np.sqrt(n)))


def confirm_persistence(
    residuals: np.ndarray, *, alpha: float = 0.05, min_observations: int = 30
) -> str:
    """'persistent' (ADF fails to reject the unit-root null -- looks like a lasting level
    shift), 'reverting' (rejects it -- looks mean-reverting), or 'inconclusive' (too little
    data, or a degenerate series ADF can't handle). ADR-0009: confirmatory only, never a
    standalone gate -- low power on short windows."""
    values = residuals[np.isfinite(residuals)]
    if len(values) < min_observations or np.allclose(values, values[0]):
        return "inconclusive"
    try:
        _, pvalue, *_ = adfuller(values, autolag="AIC", result_object=False)
    except (ValueError, np.linalg.LinAlgError):
        return "inconclusive"
    return "reverting" if pvalue < alpha else "persistent"


# --- Impact (effect size) statistics --------------------------------------------------


def cohens_d_log1p(reference: np.ndarray, current: np.ndarray) -> float | None:
    """Mean shift of log1p(pickups) in pooled-SD units. log1p, not raw counts, because
    pickup counts are skewed and zero-heavy -- raw-scale d would be dominated by a few
    high-volume zone-hours."""
    ref = np.log1p(reference[np.isfinite(reference)])
    cur = np.log1p(current[np.isfinite(current)])
    n1, n2 = len(ref), len(cur)
    if n1 < 2 or n2 < 2:
        return None
    pooled_var = ((n1 - 1) * ref.var(ddof=1) + (n2 - 1) * cur.var(ddof=1)) / (
        n1 + n2 - 2
    )
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return 0.0
    return float((cur.mean() - ref.mean()) / pooled_sd)


def cohens_h(reference_share: float, current_share: float) -> float:
    """Effect size for a two-proportion comparison (e.g. weekend vs. weekday share)."""
    return float(
        2 * np.arcsin(np.sqrt(current_share)) - 2 * np.arcsin(np.sqrt(reference_share))
    )


def cohens_w(reference_share: pd.Series, current_share: pd.Series) -> float:
    """Effect size for a multi-category chi-square comparison (e.g. zone or hour-of-day
    share vectors)."""
    index = reference_share.index
    ref = reference_share.reindex(index).fillna(0.0).clip(lower=1e-9)
    ref = ref / ref.sum()
    cur = current_share.reindex(index).fillna(0.0)
    total = cur.sum()
    cur = cur / total if total > 0 else cur
    return float(np.sqrt(np.sum((cur - ref) ** 2 / ref)))


# --- Escalation: 2x2 state, persistence, aggregation ----------------------------------


def classify_state(
    detection: float | None,
    impact: float | None,
    *,
    detection_warning: float,
    detection_retrain: float,
    impact_warning: float,
    impact_retrain: float,
) -> int | None:
    """ADR-0009's explicit 5-state table -- never a linear sum of the two statistics,
    since "detected but not impactful" and "impactful but barely detected" are not the
    same situation.

    0 ok, 1 watch (one metric warns), 2 warning (both warn), 3 elevated (either metric
    alone reaches retrain), 4 retrain-investigate (both reach retrain).
    """
    if detection is None or impact is None:
        return None
    detection, impact = abs(detection), abs(impact)
    detection_retrain_hit = detection >= detection_retrain
    impact_retrain_hit = impact >= impact_retrain
    if detection_retrain_hit and impact_retrain_hit:
        return 4
    if detection_retrain_hit or impact_retrain_hit:
        return 3
    detection_warning_hit = detection >= detection_warning
    impact_warning_hit = impact >= impact_warning
    if detection_warning_hit and impact_warning_hit:
        return 2
    if detection_warning_hit or impact_warning_hit:
        return 1
    return 0


def aggregate_score(states: dict[str, int | None], weights: dict[str, float]) -> dict:
    """Weighted mean (for trending) alongside the unweighted max (so one severely
    drifted dimension isn't diluted into invisibility by three calm ones)."""
    known = {key: value for key, value in states.items() if value is not None}
    if not known:
        return {"weighted_score": None, "max_state": None}
    weight_total = sum(weights[key] for key in known)
    weighted = sum(states[key] * weights[key] for key in known) / weight_total
    return {"weighted_score": weighted, "max_state": max(known.values())}


def persistent_alert(
    max_state: pd.Series, *, min_state: int, required_consecutive: int
) -> pd.Series:
    """True once `max_state` has held at or above `min_state` for `required_consecutive`
    straight evaluations -- the rule that keeps one noisy/event-driven window from
    reading as a confirmed alert (ADR-0009)."""
    flagged = max_state.fillna(min_state - 1) >= min_state
    return flagged.rolling(required_consecutive).sum() >= required_consecutive


# --- Per-window evaluation and the batch report ---------------------------------------


def _threshold(cfg: dict, metric: str) -> tuple[float, float]:
    bounds = cfg["thresholds"][metric]
    return bounds["warning"], bounds["retrain"]


def evaluate_window(
    window: pd.DataFrame, reference: ReferenceProfile, cfg: dict
) -> dict:
    """One trailing window vs. the reference profile, scored on all four dimensions."""
    df = _with_calendar(window)
    residual = _cell_residuals(df, reference.cell_mean)
    pickups = df["pickups"].to_numpy(dtype=float)

    psi_warn, psi_retrain = _threshold(cfg, "psi")
    d_warn, d_retrain = _threshold(cfg, "cohens_d")
    w_warn, w_retrain = _threshold(cfg, "cohens_w")
    h_warn, h_retrain = _threshold(cfg, "cohens_h")

    psi_residual = psi(reference.residual, residual, reference.residual_bin_edges)
    z_residual = deseasonalized_zscore(residual, reference.residual_std)
    d_residual = cohens_d_log1p(reference.pickups, pickups)
    adf_label = confirm_persistence(
        residual,
        alpha=cfg["adf"]["alpha"],
        min_observations=cfg["adf"]["min_observations"],
    )
    state_residual = classify_state(
        psi_residual,
        d_residual,
        detection_warning=psi_warn,
        detection_retrain=psi_retrain,
        impact_warning=d_warn,
        impact_retrain=d_retrain,
    )

    zone_share = df.groupby("zone_id")["pickups"].sum()
    psi_zone = categorical_psi(reference.zone_share, zone_share)
    w_zone = cohens_w(reference.zone_share, zone_share)
    state_zone = classify_state(
        psi_zone,
        w_zone,
        detection_warning=psi_warn,
        detection_retrain=psi_retrain,
        impact_warning=w_warn,
        impact_retrain=w_retrain,
    )

    hour_share = df.groupby("hour_of_day")["pickups"].sum()
    psi_hour = categorical_psi(reference.hour_share, hour_share)
    w_hour = cohens_w(reference.hour_share, hour_share)
    state_hour = classify_state(
        psi_hour,
        w_hour,
        detection_warning=psi_warn,
        detection_retrain=psi_retrain,
        impact_warning=w_warn,
        impact_retrain=w_retrain,
    )

    total_pickups = float(pickups.sum())
    weekend_share = (
        float(df.loc[df.is_weekend == 1, "pickups"].sum() / total_pickups)
        if total_pickups > 0
        else None
    )
    if weekend_share is None:
        psi_weekend, h_weekend, state_weekend = None, None, None
    else:
        weekend_shares = pd.Series({0: 1 - weekend_share, 1: weekend_share})
        reference_weekend_shares = pd.Series(
            {0: 1 - reference.weekend_share, 1: reference.weekend_share}
        )
        psi_weekend = categorical_psi(reference_weekend_shares, weekend_shares)
        h_weekend = cohens_h(reference.weekend_share, weekend_share)
        state_weekend = classify_state(
            psi_weekend,
            h_weekend,
            detection_warning=psi_warn,
            detection_retrain=psi_retrain,
            impact_warning=h_warn,
            impact_retrain=h_retrain,
        )

    states = {
        "pickups_residual": state_residual,
        "zone_mix": state_zone,
        "hour_of_day_mix": state_hour,
        "weekend_mix": state_weekend,
    }
    aggregate = aggregate_score(states, reference.weights)

    return {
        "n_obs": int(np.isfinite(residual).sum()),
        "psi_pickups_residual": psi_residual,
        "z_pickups_residual": z_residual,
        "cohens_d_pickups_residual": d_residual,
        "adf_label": adf_label,
        "state_pickups_residual": state_residual,
        "psi_zone_mix": psi_zone,
        "cohens_w_zone_mix": w_zone,
        "state_zone_mix": state_zone,
        "psi_hour_of_day_mix": psi_hour,
        "cohens_w_hour_of_day_mix": w_hour,
        "state_hour_of_day_mix": state_hour,
        "psi_weekend_mix": psi_weekend,
        "cohens_h_weekend_mix": h_weekend,
        "state_weekend_mix": state_weekend,
        **aggregate,
    }


def compute_drift_report(
    history: pd.DataFrame,
    reference: ReferenceProfile,
    cfg: dict,
    *,
    evaluation_hours: pd.DatetimeIndex,
) -> pd.DataFrame:
    """One row per evaluation hour. Each window is strictly causal: `[eval_hour -
    window_hours, eval_hour)`, the same availability rule forecast_features uses
    (docs/forecast_contract.md) -- drift must never be scored on data the service itself
    could not yet have seen at that hour."""
    ordered = history.sort_values("hour_utc").reset_index(drop=True)
    hour_index = pd.DatetimeIndex(ordered["hour_utc"])
    window_hours = cfg["window_hours"]
    min_window_hours = cfg["min_window_hours"]

    rows = []
    for eval_hour in evaluation_hours:
        window_start = eval_hour - pd.Timedelta(hours=window_hours)
        start_idx = hour_index.searchsorted(window_start, side="left")
        end_idx = hour_index.searchsorted(eval_hour, side="left")
        window = ordered.iloc[start_idx:end_idx]
        # zone_hour_counts.parquet is a dense zero-filled zone x hour grid (no gaps), so a
        # distinct-hour count is a valid proxy for "is this window complete enough."
        available_hours = window["hour_utc"].nunique()
        if available_hours < min_window_hours:
            rows.append({"window_end": eval_hour, "insufficient_data": True})
            continue
        rows.append(
            {
                "window_end": eval_hour,
                "insufficient_data": False,
                **evaluate_window(window, reference, cfg),
            }
        )

    report = pd.DataFrame(rows)
    if "max_state" in report.columns:
        report["persistent_alert"] = persistent_alert(
            report["max_state"],
            min_state=cfg["persistence"]["min_state"],
            required_consecutive=cfg["persistence"]["required_consecutive_windows"],
        )
    return report


def weekly_evaluation_hours(hour_utc: pd.Series, window_hours: int) -> pd.DatetimeIndex:
    """Non-overlapping windows, one per `window_hours`-sized step, spanning `hour_utc`'s
    range -- the cadence for a weekly simulation view, as opposed to compute_drift_report's
    hourly-cadence/trailing-window default."""
    start = hour_utc.min()
    end = hour_utc.max() + pd.Timedelta(hours=1)
    return pd.date_range(start=start, end=end, freq=f"{window_hours}h")


def compute_zone_drift_report(
    history: pd.DataFrame,
    reference: ReferenceProfile,
    cfg: dict,
    *,
    zone_ids: list[int],
    evaluation_hours: pd.DatetimeIndex,
) -> pd.DataFrame:
    """One row per (zone_id, window_end), scored on `pickups_residual` only -- the one
    dimension that is meaningfully a per-zone question. `zone_mix`/`hour_of_day_mix`/
    `weekend_mix` are inherently cross-zone or cross-time aggregates and stay citywide-
    only in `compute_drift_report`. This table answers "how many zones crossed the
    retrain threshold this window," not the citywide escalation decision itself
    (ADR-0009) -- a single construction-project zone shouldn't retrain the whole model,
    which is exactly the question this is for."""
    ordered = history.sort_values("hour_utc").reset_index(drop=True)
    hour_index = pd.DatetimeIndex(ordered["hour_utc"])
    window_hours = cfg["window_hours"]
    min_window_hours = cfg["min_window_hours"]
    psi_warn, psi_retrain = _threshold(cfg, "psi")
    d_warn, d_retrain = _threshold(cfg, "cohens_d")

    rows = []
    for eval_hour in evaluation_hours:
        window_start = eval_hour - pd.Timedelta(hours=window_hours)
        start_idx = hour_index.searchsorted(window_start, side="left")
        end_idx = hour_index.searchsorted(eval_hour, side="left")
        window = _with_calendar(ordered.iloc[start_idx:end_idx])
        window["residual"] = _cell_residuals(window, reference.cell_mean)

        for zone_id in zone_ids:
            zone_rows = window.loc[window.zone_id == zone_id]
            if zone_rows["hour_utc"].nunique() < min_window_hours:
                rows.append(
                    {
                        "window_end": eval_hour,
                        "zone_id": zone_id,
                        "insufficient_data": True,
                    }
                )
                continue
            # Compared against THIS zone's own training reference, never the pooled
            # all-zone one -- zones have very different baseline volumes, and the pooled
            # reference would read a zone's own characteristic level as drift.
            psi_value = psi(
                reference.residual_by_zone[zone_id],
                zone_rows["residual"].to_numpy(),
                reference.residual_bin_edges_by_zone[zone_id],
            )
            d_value = cohens_d_log1p(
                reference.pickups_by_zone[zone_id], zone_rows["pickups"].to_numpy()
            )
            rows.append(
                {
                    "window_end": eval_hour,
                    "zone_id": zone_id,
                    "insufficient_data": False,
                    "psi_pickups_residual": psi_value,
                    "z_pickups_residual": deseasonalized_zscore(
                        zone_rows["residual"].to_numpy(),
                        reference.residual_std_by_zone[zone_id],
                    ),
                    "cohens_d_pickups_residual": d_value,
                    "state_pickups_residual": classify_state(
                        psi_value,
                        d_value,
                        detection_warning=psi_warn,
                        detection_retrain=psi_retrain,
                        impact_warning=d_warn,
                        impact_retrain=d_retrain,
                    ),
                }
            )

    report = pd.DataFrame(rows)
    if "state_pickups_residual" in report.columns:
        report = report.sort_values(["zone_id", "window_end"])
        report["persistent_alert"] = report.groupby("zone_id")[
            "state_pickups_residual"
        ].transform(
            lambda states: persistent_alert(
                states,
                min_state=cfg["persistence"]["min_state"],
                required_consecutive=cfg["persistence"]["required_consecutive_windows"],
            )
        )
        report = report.sort_values(["window_end", "zone_id"]).reset_index(drop=True)
    return report


def zone_breakdown(window: pd.DataFrame, reference: ReferenceProfile) -> pd.DataFrame:
    """Per-zone residual z-score and Cohen's d for one window -- drill-down detail for
    "which zone is driving this," not part of the escalation state itself (ADR-0009)."""
    df = _with_calendar(window)
    df["residual"] = _cell_residuals(df, reference.cell_mean)
    rows = []
    for zone_id, group in df.groupby("zone_id", observed=True):
        rows.append(
            {
                "zone_id": zone_id,
                "n_obs": int(group["residual"].notna().sum()),
                "mean_residual": float(group["residual"].mean()),
                "z_residual": deseasonalized_zscore(
                    group["residual"].to_numpy(),
                    reference.residual_std_by_zone[zone_id],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("z_residual", key=abs, ascending=False)
