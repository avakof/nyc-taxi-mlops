### Outline

- Problem introduction - forecasting model works in a notebook but we cannot trust it in production. we need a few things for it to be trusted
  - what data is used
  - whether the data has drifted or not
  - how to rollback to a previous version
- An overview of the Solution (we will then get into the specifics of why we chose this methodology)
  - using a script to run the model
  - implemented mlflow for tracking and versioning
  - api services for putting into production (with various tests)
  - drift monitoring
    - not only can we view that it is drifting, but we can look through where it is coming from, and how the current data differs from the past
    - there currently is no retraining trigger - the trigger is to create an incident report, and we can manually go in and retrain, since this is new, we want to understand the data and how the model acts in production before automation. However, we would build [xyz] if we wanted to automate it.
- modeling - how we chose the split, creating a baseline, feature engineering & choosing the metrics (this can be important in the drift)
- putting it into production (the important bit that solves our problem) 
  - mlflow - tracking the model and choosing a champion model
  - FastAPI - serving
  - Docker - packaging
- monitoring - drift monitoring
  - which variables / calculations we do and what they actually measure
  - scope (by zone and overall)
  - thresholds and how we decide a warning vs danger and if it is a short term spike or real signal
  - dashboard
- recap on how this system solves the problems we introduced
- final note: ADRs and ensuring we could hand off monitoring to anyone

- what we would do next 
  - implementing auto triggering policy



Julia: intro + model

Ali: putting this model into production

Julia: drift monitoring

Ali: documentation policy and what we would do next

---

## Expanded script — one idea per slide

~10 minutes, ~13 slides, roughly 45-55s each. Each slide below is `Speaker — Slide title`, a one-line key idea, then the supporting detail/numbers to say or show. This is a handoff doc for the team, not just presenter notes — assume the reader hasn't seen the repo.

### Slide 1 (Julia) — The problem

**Key idea:** a notebook model isn't a production system — we don't know what it was trained on, whether it still works, or how to undo it.

- Frame it exactly as the brief does: "the forecasting model works beautifully in a notebook, nobody trusts it in production." The gap isn't model quality, it's operability.
- Three concrete questions we had to be able to answer for any deployed model: *what data trained it, is that data drifting, how do we undo a bad version.*
- This becomes the spine of the whole talk: every later section is "here's how we answer one of these three questions."

### Slide 2 (Julia) — The system, on one diagram

**Key idea:** one picture of the pipeline — data → MLflow/DVC → FastAPI/Docker → monitoring → (manual) retrain/rollback decision.

- Walk left to right: TLC monthly Parquet → DVC-tracked raw/processed data → feature pipeline → MLflow (tracking + Registry, `champion`/`candidate` aliases) → exported release → FastAPI in Docker → drift monitoring reading forecast history → a human decision (retrain or roll back), not an automatic one.
- Say explicitly: *script-driven, not a 24/7 service* — each stage is a `uv run python scripts/...` command, not a scheduler. That's a deliberate scope choice for a 2-person team on a course timeline, not an oversight.
- This is the diagram the grading rubric asks for ("the problem and your system on one diagram") — reuse it, don't redraw it later.

### Slide 3 (Julia) — Data and the chronological split

**Key idea:** we forecast next-hour pickups per zone, split strictly by time, never at random.

- Source: NYC TLC yellow-taxi trip records, Jan–Apr 2020, aggregated from raw trips to one row per (zone, hour). 20 zones selected by training-period pickup volume and frozen in `config/selected_zones.json` — mostly Manhattan (Upper East Side South/North, Midtown Center/East, Times Sq) plus a mid/low-volume tail, so results generalize across demand levels, not just the busiest corridor.
- Periods (`params.yaml`): warmup Jan 1–8 (fills the 168h lag), train Jan 8–Feb 3 (26 days), calibration Feb 3–17 (14 days), frozen test Feb 17–Mar 1 (13 days), then Mar 1–May 1 held out as the drift/replay period — this is the exact window the course brief suggests (demand collapse).
- Name the one real bug this caught: `HistGradientBoostingRegressor`'s default `early_stopping="auto"` silently activates above 10k rows and validates on an internally **shuffled** split — a chronological-leak bug hiding inside a chronological pipeline. Caught in PR review, fixed with `early_stopping=False`. Worth one sentence live: it's a good answer if someone on the jury asks "give me an example of a design choice you got wrong and fixed."

### Slide 4 (Julia) — Baseline first, then a model

**Key idea:** the baseline isn't a formality — it's the thing that tells you whether the fancier model is worth deploying at all.

- Two seasonal baselines, zero parameters: previous-day and previous-week-same-hour (`lag_24`, `lag_168`). Test MAE: prev-day 63.46, prev-week 47.87 — prev-week wins, so *that's* the one registered as the initial MLflow Registry `champion`, not a placeholder.
- Candidate: one `HistGradientBoostingRegressor` (scikit-learn), trained on 12 features — lags at 1/2/3/24/168h, rolling means over 3/24/168h, `zone_id`, `hour_of_day`, `day_of_week`, `is_weekend`. Fit only on training data; tuned externally against calibration, not via internal random early stopping (see Slide 3's bug).
- Result: test MAE 27.44 vs. the 47.87 baseline — a 42.7% improvement, WAPE 0.100 vs. 0.175, no zone regressing more than ~20% (promotion-gate passed on all checks). Say it plainly: *"a simple, well-operated model beats a brilliant one nobody can reproduce" — the course's own framing — so we spent the saved time on the pipeline around the model, not more tuning.*

### Slide 5 (Julia) — Feature importance, and a lesson in trusting your tools

**Key idea:** we checked which features the model actually uses — and found our explainability library was lying to us.

- First SHAP run (`TreeExplainer`) said `zone_id` had exactly 0.0 importance. That's suspicious on its face — zone obviously matters for a demand forecast.
- Cross-checked against permutation importance: the two disagreed sharply (different top features entirely, and permutation showed `zone_id` as a real mid-tier contributor). Root-caused to `TreeExplainer` being broken for this estimator's native categorical splits; switched to a model-agnostic `shap.explainers.Permutation`, which then matched permutation importance feature-for-feature.
- Why this matters beyond modeling: this result is *why* drift monitoring (Slide 9) doesn't use SHAP weights — a model that (wrongly, as it turned out) looks like it ignores `is_weekend`/`zone_id` would have silently zeroed out the exact signal this project's target incident (a weekday/weekend demand collapse) depends on. Final ranked features: `lag_1` dominant, then `lag_168`, `lag_24`, `hour_of_day`.

### Slide 6 (Ali) — Tracking and versioning: MLflow + DVC

**Key idea:** every served prediction has to trace back to exact code, data bytes, and a model version — not "the file that happened to be on disk."

- DVC versions raw/processed data; the download script records source URL, hash, size per month and freezes it in a manifest — a re-download that doesn't match the frozen hash stops instead of silently replacing data.
- MLflow (explicit SQLite backend + persistent artifact store, not the default file store) logs every run's params, seed, split boundaries, MAE/RMSE/WAPE, Git commit + dirty-state, and the data hash — so any served model can be traced back to the exact commit and data that produced it.
- Registry name: `nyc-taxi-demand-forecaster` — deliberately model-agnostic (one name holds both the baseline wrapper and trained HGB versions over time). We use aliases (`champion`/`candidate`), not MLflow's deprecated numbered stages. `champion` = prev-week baseline v1; `candidate` = HGB v3. Registration resolves to one **immutable numeric version** and runs a smoke prediction before declaring success — never "whatever `champion` currently points to," because an alias can move.

### Slide 7 (Ali) — Serving: FastAPI

**Key idea:** the service is boring on purpose — validate the request, build features from controlled history, return a number, fail loudly when it can't.

- Three endpoints: `POST /forecast` (zone + forecast-origin timestamp in → expected next-hour pickups out, plus model version and release ID), `GET /health` (200 only when model + history are usable), `GET /model-info` (which exact model version, source run, release is loaded).
- The model is resolved once at startup to one immutable version — a Registry alias moving later does **not** change what's being served. That's deliberate: redeploying is a separate, explicit action, never implicit.
- Correctness contract: missing/incomplete history is never silently filled with a zero and forecast — it's a 503. Invalid zone or timestamp is a 422. An inference failure is a 500. This distinction (unavailable vs. invalid vs. broken) is covered by integration tests that spin up a real isolated MLflow backend and a real registered model — not mocks.

### Slide 8 (Ali) — Packaging and rollback

**Key idea:** Docker makes the release reproducible; retaining the *previous* release makes it reversible — and we measured the reversal, not assumed it.

- Docker + Docker Compose, uv-locked environment, one API worker — same environment locally, in CI, and at "deploy" time. A smoke-test script hits the running container and checks a known forecast after every build.
- Rollback: retain at least two tested release images/bundles. `rollback_demo.py` runs the whole drill end-to-end in an isolated demo: verifies a good deployment, swaps in a deliberately faulty image (forecasts return 500 while `/health` still reports 200 — a reminder that liveness isn't correctness), restores the original image, re-verifies a known forecast.
- The number to say out loud: **21.47 wall-clock seconds** measured recovery on the recorded drill. Caveat it honestly — one measured run, not an SLA — and note it's written up as a full postmortem (`docs/postmortems/0001-serving-rollback.md`), not just a console log.

### Slide 9 (Julia) — Drift monitoring: what we actually measure

**Key idea:** we don't monitor 12 correlated features, we monitor 4 independent *dimensions* of change, each with a detection statistic and an impact statistic.

- The 12 model inputs aren't 12 independent signals — 8 of them are lagged/rolling copies of the same `pickups` series, so summing "drift" across all of them would just count one real event eight times.
- Four dimensions instead: `pickups_residual` (demand minus its seasonal reference for that zone/hour/weekend cell — the deseasonalized signal), `zone_mix`, `hour_of_day_mix`, `weekend_mix`.
- Each gets a statistic matched to its data type: PSI (or chi-square) to *detect* a shift, paired with an effect size to say whether it *matters* — Cohen's d for the continuous residual, Cohen's w for the two categorical mixes, Cohen's h for the weekend proportion. This is the answer to "a histogram changed, so what" — detection alone was explicitly rejected as insufficient during design (ADR-0009).

### Slide 10 (Julia) — Scope, thresholds, and not crying wolf

**Key idea:** drift has to persist before it's an alert, and we can tell a one-off spike from a real regime change.

- Scope: computed citywide-pooled by default (one escalation decision), with a per-zone weekly breakdown for drill-down — "which of the 20 zones is driving this week's alert" — shown live on the dashboard, not 20 parallel top-level alerts.
- Each dimension gets a 5-state score (ok → watch → warning → elevated → retrain-investigate) from its detection/impact *pair*, not an additive score — a highly-detected-but-low-impact window and a barely-detected-but-high-impact window mean different things and shouldn't collapse to the same number.
- Persistence gate: a dimension has to stay at warning-or-above for 3 consecutive trailing-168-hour evaluations before it counts as a real alert — this is how a one-off event (a storm, a concert) gets told apart from a lasting shift, backed by an Augmented Dickey-Fuller test as a confirmatory (not primary) signal on whether the residual series looks like a lasting level change.
- Be honest about the caveat: thresholds (PSI 0.10/0.25, Cohen's d/w/h 0.2/0.5) are *placeholders* from standard statistical convention, not yet calibrated against our own two-week healthy window — that recalibration is explicitly tracked as open follow-up, not hidden.

### Slide 11 (Julia) — Dashboard demo

**Key idea:** show, don't just claim — the drift report is a browsable artifact, not a log line.

- Live or recorded: Streamlit dashboard reading the precomputed `reports/drift_report.parquet` / `drift_report_by_zone.parquet`. Walk through one healthy week and one week where a dimension state escalates, showing PSI/Cohen's values and the "which zone is driving it" per-zone table side by side with the raw ground-truth pickup series for that zone.
- Make the one-sentence point the course brief wants here: this is read-only over a *static historical replay snapshot* — we're not claiming a live feed, and we say so on the dashboard itself.

### Slide 12 (Julia) — Why there's no auto-retrain trigger yet

**Key idea:** we chose not to automate the response, and that was a decision, not a gap.

- Current state: drift detection produces evidence and an alert; the response is a human-written incident report, then a manual retrain if warranted. No code automatically promotes a new model.
- Why: this is new, unvalidated machinery on two weeks of calibration data — we want to watch how the model and the detector actually behave in production before letting a threshold crossing trigger a retrain+promote with nobody looking. Automating the *decision* before trusting the *signal* would be backwards.
- What the automated version would need (said, not built): a durable trigger job with cooldown/dedup state so a flapping alert can't spawn duplicate retrains, minimum-history/coverage checks before a candidate is even attempted, and the same promotion gates we already apply manually (≥5% MAE improvement over champion, no zone regressing >10%, passing smoke checks) — sets up Slide 13's "what's next."

### Slide 13 (Ali) — Documentation as the handoff mechanism

**Key idea:** the point of the ADRs is that this team can leave and the next one can still operate the system.

- Nine numbered ADRs (`docs/adr/`), one per decision, each with Context / Decision / Alternatives considered / Consequences — including the ones we got wrong and changed (the early-stopping leak, the broken SHAP explainer) and the ones still explicitly open (drift threshold calibration, outcome-based drift). This is deliberate: an ADR that only records decisions that worked isn't useful to someone debugging a decision that didn't.
- Runbooks and postmortems are separate from ADRs: a runbook is "do this," a postmortem is "this happened, here's what we measured." The rollback runbook has the literal restore command for a real deployment, not just the demo script.
- The handoff test we used: could someone with zero context `git clone`, follow the README, and reach a working, inspectable system? That's load-bearing for grading too — "a fresh clone plus your README must get a working system."

### Slide 14 (Ali) — What we'd do with another month

**Key idea:** the next investment is automation and calibration, not new features.

- **Automate the retraining trigger** (Slide 12): wire the durable-job design into `compute_drift.py`'s output so a persistent alert creates a tracked candidate-training request instead of a printed recommendation — promotion still stays a separate, explicit, gated step.
- **Calibrate drift thresholds for real**, against the Feb 3–17 healthy window specifically, instead of textbook PSI/Cohen's conventions — and be honest that two weeks is thin evidence for a false-positive rate.
- **Extend replay through the full March–April collapse** (currently `compute_drift.py` only scores calibration+test; the aggregation pipeline stops at the test boundary) to get the actual demand-collapse incident, not just a healthy-period dry run.
- **Close the loop with an outcome-based drift ADR** (MAE/WAPE degradation vs. baseline, which has only a 1-hour label latency so it's cheap to monitor directly) — currently a deliberately separate, not-yet-built concern from the input-drift monitoring shown today.
- Close on the framing from Slide 1: every one of these line items answers one of the three original questions better, not a new fourth thing.

---

## System diagram (for Slide 2)

The professor's suggested structure asks for "the problem and your system on one diagram." This is that diagram — reflects what's actually built (script-driven batch pipeline), not the full aspirational design in `docs/PROJECT_PLAN.md`.

```mermaid
flowchart TD
    subgraph DATA["Data & versioning"]
        A["TLC monthly Parquet\n+ zone lookup"] --> B["download_data.py\nSHA-256 manifest"]
        B --> C["DVC: data/raw, data/processed\n(.dvc pointers in Git)"]
        C --> D["aggregate_and_split.py\nzone-hour counts, chronological split"]
    end

    subgraph MODEL["Tracking & registry (Julia)"]
        D --> E["baselines.py\nprev-day / prev-week"]
        D --> F["tune_hgb.py / train_hgb.py\nHistGradientBoostingRegressor"]
        E --> G["MLflow tracking\nparams, metrics, git commit, data hash"]
        F --> G
        G --> H["MLflow Registry\nnyc-taxi-demand-forecaster\nchampion (baseline) / candidate (HGB)"]
    end

    subgraph SERVE["Serving & rollback (Ali)"]
        H --> I["export_release.py\nmodels/final-serving (frozen version)"]
        I --> J["Docker image\nFastAPI + Uvicorn"]
        J --> K["docker compose\nGET /health, GET /model-info\nPOST /forecast"]
        K -. smoke_api.py .-> K
        L["Previous retained release"] --> M["rollback_demo.py / restore_release.py"]
        M --> K
    end

    subgraph MONITOR["Drift monitoring (Julia)"]
        D --> N["compute_drift.py\nreference profile from train period"]
        K -- "forecast history" --> N
        N --> O["drift_report.parquet\n+ by-zone breakdown"]
        O --> P["Streamlit dashboard\nPSI / Cohen's d,w,h / persistence state"]
        P --> Q{"Persistent alert?"}
    end

    subgraph RESPONSE["Manual response (no auto-trigger)"]
        Q -- "yes" --> R["Incident report\n(human reviews evidence)"]
        R --> S{"Retrain or roll back?"}
        S -- retrain --> F
        S -- "roll back" --> M
        Q -- "no" --> T["Healthy — no action"]
    end
```

**How to narrate it in 30-40 seconds:** data flows down the left (TLC → DVC → features), splits into tracking/registry in the middle (Julia's half), becomes a served release on the right (Ali's half) with a retained-previous-release path for rollback, and the served forecast history feeds back up into drift monitoring at the bottom, which ends in a human decision — retrain (loops back into training) or roll back (loops back into serving) — not an automatic one. That loop is the answer to "how do we undo a bad version" from Slide 1.