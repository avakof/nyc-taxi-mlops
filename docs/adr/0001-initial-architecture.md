# 0001. Initial architecture

Status: Accepted

Baseline design decisions made together before implementation. Later decisions (drift thresholds, retraining-gate tuning, incident responses) get their own numbered ADR, not an edit to this one.

## Task and contract

**Context:** need a scoped, unambiguous forecasting target before any code.
**Decision:** predict next-hour pickups per zone, aggregated to one row per zone-hour, from recent-hour + calendar features only. Full contract in [../forecast_contract.md](../forecast_contract.md).
**Consequences:** any feature using information not available at forecast issuance time is a bug, not a design choice.

## Data and versioning

**Context:** TLC publishes monthly Parquet; raw files can't go in Git per the project rules.
**Decision:** Jan–Apr 2020 yellow-taxi data, 20 zones selected by training-period pickup volume and frozen. Raw files fetched by a download script, versioned with DVC, never committed to Git.
**Alternatives considered:** committing a pre-aggregated CSV directly — rejected, loses provenance and reproducibility.
**Consequences:** need a working DVC remote before this is actually reproducible from a clean clone, not just locally.

## Split / leakage prevention

**Context:** the course labs split randomly; random splits leak future information into training for time series.
**Decision:** chronological split only — train → calibration → test, then Mar–Apr 2020 as the drift/replay period. See adversarial-audit corrections in [../COURSE_REVIEW.md](../COURSE_REVIEW.md).
**Consequences:** every evaluation must use timestamp cutoffs, never row position, even when stacking zones.

**Correction (PR #2 review, Ali):** `HistGradientBoostingRegressor`'s `early_stopping` was left at its default (`"auto"`), which silently activates above 10k training samples (ours: 12,480 rows) and validates internally on a **randomly shuffled** split — violating this exact chronological-only rule, undetected until external review. Fixed: `early_stopping=False` in both `scripts/tune_hgb.py` and `scripts/train_hgb.py`; redundant anyway since `max_iter` is already tuned externally against the calibration period. Retrained: test MAE 27.44 (was 27.71), promotion gates still pass (42.7% improvement over champion). The outer train/calibration/test split itself was never wrong — only this one model's internal training mechanics were.

## Modeling

**Context:** project is graded on operation, not model quality ("a simple model that's well operated beats a brilliant model nobody can reproduce").
**Decision:** seasonal (prev-day/prev-week) baselines first, then one scikit-learn regressor. Metrics: MAE (primary), RMSE, WAPE. The baseline exists specifically to answer whether the trained model is worth deploying at all, not as a formality — if HGB doesn't beat the naive prev-day-per-zone forecast, that's a real finding, not a bug.
**Consequences:** no time spent on model tuning beyond a few recorded configurations; effort goes to the pipeline around it.

Feature-set questions raised after this baseline (e.g. lag/rolling redundancy found in the EDA) are tracked in [ADR-0004](0004-defer-feature-pruning-to-shap.md), not as edits here.

## Experiment tracking

**Context:** need to trace any served model back to the data/code/config that produced it.
**Decision:** MLflow with an explicit SQLite backend store (not the default file store) and a persistent artifact directory. Registry aliases (`candidate`/`champion`), not deprecated stages.
**Consequences:** every training run logs params, seed, split boundaries, data hash, git commit.

**Local dev note:** started with (verified working against our locked MLflow version; matches the pattern in `course_content/Session 1/01-mlflow`):

```bash
uv run mlflow ui --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts
```

`127.0.0.1`, not `0.0.0.0` — loopback-only per the packaging decision below. `mlflow.db` and `mlartifacts/` are both gitignored; this is local runtime state, not a tracked artifact.

## Serving

**Context:** need a contract the FastAPI service and its tests can be built against.
**Decision:** `POST /v1/forecast`, `GET /health/live`, `GET /health/ready`, `GET /model-info`, `GET /metrics`. Resolve a Registry alias once at startup to an immutable version; don't re-resolve per request.
**Consequences:** a Registry alias change alone does not redeploy the running service — deployment is a separate explicit step.

## Packaging

**Context:** need the same environment locally, in CI, and at "deploy" time.
**Decision:** Docker + Docker Compose, uv-locked Python environment, single API worker + single serialized training/monitoring worker for now.
**Consequences:** no HA claim; this is a local-project choice. A Windows-specific local dev environment constraint (WDAC blocking compiled extensions) is tracked separately in [ADR-0003](0003-wsl-local-environment.md).

## Monitoring (intent only — detail deferred)

**Context:** full four-layer monitoring (service/data/prediction/outcome) is a stretch goal given the timeline.
**Decision:** scope to whatever's achievable; Prometheus/Grafana are not required for the core deliverables.
**Consequences:** drift detection, retraining trigger, and rollback mechanics each get their own ADR when actually built — this section is intentionally not the final design.
