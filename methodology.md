# Methodology

**Due: 2026-11-19** (confirmed by the professor; the project PDF prints Nov 5, which is stale).
Team of 2. Scope below is compressed to fit that window — see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the original, fuller design if time allows revisiting it.

## Key points

- **Task:** predict next-hour pickups per zone from recent-hour + calendar features, aggregated to one row per zone-hour. Productionize it.
- **Data:** TLC yellow taxi, Jan–Apr 2020, 20 zones selected by training-period volume. Raw files never committed; downloaded via script + DVC.
- **Split:** chronological only. Train → calibration → test, then Mar–Apr 2020 as the drift/replay period (natural demand collapse; synthetic injection optional, skip if time-constrained).
- **Metrics:** MAE (primary), RMSE, WAPE.
- **Stack:** MLflow (tracking + registry), DVC (data), FastAPI (serving), Docker Compose (packaging). Full Prometheus/Grafana monitoring is a stretch goal, not required for the core 7 deliverables.
- **Drift:** needs its own rigorous treatment (reference window, test statistic, threshold calibration) — detailed in its own ADR when we get there, not hand-waved.
- **Documentation:** baseline design decisions are in one combined [ADR-0001](docs/adr/0001-initial-architecture.md), one section per area. Decisions made after the baseline (drift thresholds, retraining-gate tuning, an actual rollback) get their own numbered ADR. Incidents get written up as [post-mortems](docs/postmortems/), linked back from the ADR that set the relevant gate/threshold. The ADR set is the live-demo backbone for the presentation; the final ADR's Consequences carries "what we'd do with more time."

## Tasks

| Task                                                                               | Owner     | Status      |
| ---------------------------------------------------------------------------------- | --------- | ----------- |
| Data download + DVC snapshot (raw parquet, zone lookup, hashes)                    | Julia     | Done |
| Feature pipeline (zone-hour aggregation, causal lags/rolling, chronological split) | Julia     | Done |
| Baseline + model training, MLflow tracking                                         | Julia     | Done |
| MLflow Registry (candidate/champion alias, version export)                         | Julia     | Not started |
| FastAPI service (`/forecast`, `/health`, `/model-info`) + integration tests        | Ali       | Not started |
| Docker packaging (Dockerfile, Compose, smoke test)                                 | Ali       | Not started |
| Drift detection (reference vs. replay comparison) + retraining trigger             | Julia     | Not started |
| Rollback procedure (retain previous release, restore + verify)                     | Ali       | Not started |
| ADR-0001 baseline + ADR-0002 (tracking conventions), ADR-0003 (WSL env), ADR-0004 (feature pruning) done; new ADRs per decision after that | Ali/Julia | In progress |
| README bootstrap verification (fresh clone → working demo)                         | Ali/Julia | Not started |

Fill in Owner before starting; update Status as work lands.
