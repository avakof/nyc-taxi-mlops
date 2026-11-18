# Methodology

**Due: 2026-11-19** (confirmed by the professor; the project PDF prints Nov 5, which is stale).
Team of 2. Scope below is compressed to fit that window — see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the original, fuller design if time allows revisiting it.

## Key points

- **Task:** predict next-hour pickups per zone from recent-hour + calendar features, aggregated to one row per zone-hour. Productionize it.
- **Data:** TLC yellow taxi, Jan–Apr 2020, 20 zones selected by training-period volume (frozen in `config/selected_zones.json`). Raw files never committed; downloaded via `scripts/download_data.py` + DVC.
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
| MLflow Registry (candidate/champion alias, version export)                         | Julia     | Done |
| FastAPI service (`/forecast`, `/health`, `/model-info`) + integration tests        | Ali       | Done on milestone/serving; PR pending |
| Docker packaging (Dockerfile, Compose, smoke test)                                 | Ali       | Done on milestone/serving; PR pending |
| Drift detection (reference vs. replay comparison) + retraining trigger             | Julia     | Not started |
| Rollback procedure (retain previous release, restore + verify)                     | Ali       | Done on milestone/serving; PR pending |
| ADRs for each decision made (see [docs/adr/](docs/adr/) for the current list)      | Ali/Julia | In progress |
| README bootstrap verification (fresh clone → working demo)                         | Ali/Julia | Done on milestone/serving; PR pending |

## Serving milestone evidence

- Complete API and self-contained hashed releases; 66 serving-era tests plus downloader regression coverage.
- Docker/Compose smoke verified baseline and native HGB loading.
- User rollback drill `20261118T095307Z-3c11e0`: forecast 200 → 500 → 200, recovery 21.47 wall-clock seconds.
- [Postmortem](docs/postmortems/0001-serving-rollback.md), [bootstrap decision](docs/adr/0008-reproducible-local-bootstrap.md), and README describe reproduction and local-only artifact storage.
- Drift detection/retraining and a joined demand-drift incident remain separate team work. The deployment-failure drill is not evidence of completed drift detection.
