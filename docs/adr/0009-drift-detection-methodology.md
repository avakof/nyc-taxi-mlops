# 0009. Data-drift detection methodology

Status: Accepted

*In one sentence: detect whether drift is present with PSI, how impactful it is with
Cohen's d/h/w, confirm with ADF whether a flagged shift looks like a lasting level
change rather than a one-off event, and require that signal to persist across
consecutive trailing-window evaluations before treating it as a real alert.*

## Context

Milestone 8 (ADR-0001) needs a concrete drift signal: windowed PSI on
lag-count distributions plus a demand-volume deviation signal, calibrated
on healthy data, with persistence required before anything is treated as
real. That's a design intent, not a computable method. We need to decide
exactly which variables get monitored, which statistics detect "did it
move" vs. "does it matter," how those combine into one escalation
decision, and at what granularity -- before writing any code.

This ADR covers the **data layer only** (ADR-0001 Sec. 7's four-layer
split: service / data / predictions / outcomes). Prediction- and
outcome-level drift (forecast distribution shift, MAE/WAPE degradation
vs. baseline) are a separate concern with a 1-hour label latency that
makes direct outcome monitoring cheap here -- that gets its own ADR when
we build the retraining trigger (ADR-0001 Sec. 8), rather than being
folded into this one.

## Decision

**Monitor four dimensions, not twelve features.** The model's 12 inputs
(`lag_1/2/3/24/168`, `roll_3/24/168`, `hour_of_day`, `day_of_week`,
`is_weekend`, `zone_id`) are not 12 independent signals: 8 of them are
shifted/averaged copies of the same `pickups` series, so a real demand
shift moves all 8 together and a SHAP- or feature-level weighted sum
would count that one underlying event roughly eight times. We monitor:

1. **`pickups_residual`** -- `pickups` minus the reference mean for its
   own `(zone_id, hour_of_day, is_weekend)` cell. This is the
   deseasonalized signal that `lag_1`/`lag_24`/`lag_168`/`roll_*` are all
   noisy restatements of; monitoring it once avoids the redundancy above
   while still capturing the thing that matters (demand level changed).
2. **`zone_mix`** -- each zone's share of total pickups.
3. **`hour_of_day_mix`** -- each local hour's share of pickups (24
   buckets).
4. **`weekend_mix`** -- weekend vs. weekday share.

Each dimension gets one **detection** statistic (did the distribution
move) and one **impact** statistic (does the move matter), matched to
its actual data type instead of using one metric everywhere:

| Dimension | Detection | Impact |
|---|---|---|
| `pickups_residual` | PSI on the residual distribution (quantile-binned from reference); rolling z-score of the window's mean residual as a faster secondary read | Cohen's d, computed on `log1p(pickups)` (count data is skewed and zero-heavy; raw-scale `d` would be dominated by high-volume zones and distorted by the zero mass) |
| `zone_mix` | PSI / chi-square over the zone-share vector | Cohen's w (effect size for a multi-category chi-square) |
| `hour_of_day_mix` | PSI / chi-square over the 24-bucket share vector | Cohen's w |
| `weekend_mix` | PSI / two-proportion chi-square | Cohen's h (effect size for a two-proportion comparison) |

**Confirmatory check, not a gate:** an Augmented Dickey-Fuller test on the
windowed, deseasonalized residual series, used only to label a flagged
`pickups_residual` window as `"persistent"` (fails to reject the
unit-root null -- looks like a lasting level shift) vs. `"reverting"`
(rejects it -- looks mean-reverting, i.e. an event spike). This feeds the
persistence rule below; it does not independently escalate anything.
ADF has low power on short windows and needs the same deseasonalized
input as everything else here, so it is explicitly a secondary,
confidence-adjusting signal, not a primary detector.

**Window, not single-hour, comparisons.** Each dimension's statistics are
computed over a trailing window ending at the evaluation hour (default
168h / 7 days, configurable), not on that hour's ~20 data points alone.
Evaluation still runs every hour (`drift.evaluation_cadence_hours: 1` in
`params.yaml`); only the statistic's sample size is decoupled from the
cadence.

**2x2 escalation per dimension, not a linear sum.** Each dimension gets a
5-state score from its own detection/impact pair, via an explicit table
rather than adding two numbers (a PSI-high/impact-low window and a
PSI-low/impact-high window mean different things and shouldn't collapse
to the same score):

| State | Meaning | Rule |
|---|---|---|
| 0 | ok | neither statistic reaches warning |
| 1 | watch | exactly one statistic reaches warning |
| 2 | warning | both reach warning, neither reaches retrain |
| 3 | elevated | one statistic alone reaches the retrain threshold |
| 4 | retrain-investigate | both reach the retrain threshold |

**Persistence before anything counts as actionable:** a dimension's
state must stay at or above `warning` (2) for
`drift.persistence.required_consecutive_windows` consecutive hourly
evaluations (default 3, each over its own trailing 168h window) before
it's surfaced as a real alert rather than a single noisy read. A
`pickups_residual` window the ADF check calls `"reverting"` needs the
full persistence count to still hold; one called `"persistent"` can
satisfy it early. This is how a big one-off event (concert, storm)
gets distinguished from a lasting regime change, without a separate
ad hoc rule bolted on afterward.

**Weights come from the reference data, not from the model's SHAP
importances.** `monitoring.dimension_weights` computes each weight as
the share of `pickups` variance explained by that dimension on the
reference period -- a one-way eta-squared for `zone_id`, `hour_of_day`,
and `is_weekend`, with `pickups_residual`'s weight set to the
*unexplained* remainder (which is, by construction, exactly the variance
the residual signal represents). `is_weekend` gets a configurable floor
(`drift.weights.weekend_floor`, default 0.05) taken proportionally from
the other three and then the four weights are renormalized to sum to 1,
because this project's target incident (a demand collapse) is exactly a
weekday/weekend-pattern change, and eta-squared alone can floor it near
zero in a way that's about the model's current reliance on the feature,
not about whether that regime shift is worth seeing. The aggregate score
is the weighted sum of the four dimension states (for dashboard
trending) alongside the unweighted max dimension state (so one severely
drifted dimension can't be diluted into invisibility by three calm
ones).

**Thresholds are placeholders pending calibration**, not measured
values: PSI 0.10/0.25, Cohen's d 0.2/0.5, Cohen's w 0.1/0.3, Cohen's h
0.2/0.5 (warning/retrain), drawn from standard PSI and Cohen's-effect-size
conventions rather than this project's own healthy data. ADR-0001 already
flags that two healthy calibration weeks give limited evidence for a
false-positive rate; recalibrating these against Feb 3-17 2020 is tracked
as follow-up work, not claimed as done by this ADR.

## Alternatives considered

**Monitor all 12 engineered features with SHAP-weighted PSI+Cohen's-d**
(the originally discussed version): rejected -- 8 of the 12 are
correlated transforms of one series, so the weighted sum mostly just
re-derives "has `pickups` drifted" while looking like it's combining
independent evidence, and the `is_weekend`/`day_of_week` split getting
exactly `0.0` SHAP importance (see ADR-0004 and `reports/shap_summary.json`)
would silently zero out the one signal most relevant to this project's
target incident.

**Reuse the model's SHAP-importance ratios directly as monitoring
weights**: rejected for the same `is_weekend = 0.0` reason, plus a
conceptual mismatch -- SHAP measures what the *model* currently leans on
to predict, not how much variance a grouping actually explains in the
quantity being monitored. A model that happened to ignore a feature
doesn't mean that feature's regime shift is uninteresting to a human
investigating drift.

**Cohen's d for every variable, including `zone_id` and `is_weekend`**:
rejected -- `d` is a continuous-mean-shift effect size; applying it to a
20-level category or a binary share either requires an awkward
0/1 encoding that loses the multi-category structure or misrepresents a
proportion shift as a mean shift. Cohen's w and h exist precisely for
these cases.

**Compute PSI/Cohen's-d on each single hour's ~20 data points**:
rejected as too fine-grained for a stable statistic, per our own
discussion -- decoupling window size (168h default) from evaluation
cadence (hourly) keeps the per-hour responsiveness without a 20-point
sample.

**ADF (or a structural-break test) as the primary detector**: rejected
as a first-line gate -- low power on short windows, and it answers "is
this one series stationary," not "how far did it move" or "does it
matter," so it can't stand in for PSI/Cohen's d. Kept as the confirmatory
signal for the persistence rule instead.

**A single additive 0-4 score (PSI contribution + Cohen's contribution)**:
rejected -- addition treats "detected but not impactful" and "impactful
but barely detected" as interchangeable, which defeats the stated reason
for pairing the two metrics in the first place. The explicit state table
keeps those cases distinguishable.

**Folding outcome-based (MAE/WAPE) drift into this same scheme**:
rejected for now -- outcomes have a 1-hour label latency here, which is
fast enough to monitor directly and simply, and conflating "inputs moved"
with "the model got worse" is exactly the mistake ADR-0001's adversarial
audit calls out ("a histogram change is treated as proof of model
failure"). That gets a separate ADR when the retraining trigger is
built.

## Consequences

`src/taxi_mlops/monitoring.py` implements PSI, categorical PSI, Cohen's
d/h/w, the deseasonalized rolling z-score, the ADF wrapper, eta-squared
dimension weighting, and the state/persistence logic as pure functions
over DataFrames -- no I/O, same shape as `features.py` -- so they're
testable with small synthetic frames and reusable from both a batch
report script and, later, a live monitoring worker.

`scripts/compute_drift.py` builds the reference profile from the
training period and produces a per-hour drift report over a requested
replay range; it is the input to the Streamlit dashboard discussed next,
and to the eventual retraining trigger.

This design monitors citywide-pooled dimensions by default; per-zone
`pickups_residual` breakdown is computed as a secondary table for
drill-down (which zones are driving a flagged window) rather than 20
parallel top-level drift timelines -- revisit if a specific zone needs
its own persistence/escalation state rather than appearing only as
dashboard detail.

Threshold recalibration against the Feb 3-17 2020 healthy window, and
the separate outcome-drift ADR, are both open follow-up work, not
delivered here.
