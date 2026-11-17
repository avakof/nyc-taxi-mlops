# 0002. MLflow logging conventions

Status: Accepted

## Context

ADR-0001 commits to logging params/metrics/tags/git-commit/data-hash on every run, but doesn't fix exact key names. Without a single source of truth, scripts drift (`"mae"` vs `"MAE"` vs `"test_mae"`), splits collide in the metrics table, and 20 zones × several metrics would explode the run view if logged flat.

## Decision

Centralize all MLflow key names and provenance lookups in `src/taxi_mlops/tracking.py`. Every script that logs to MLflow (training, registration, future monitoring) imports from it rather than hardcoding strings.

- **Params** (flat, snake_case): `model_type`, `seed`, `feature_lags_hours`, `feature_rolling_windows_hours`, `n_zones`, `data_months`, plus `split_<period>_start`/`_end` for whichever periods the run used.
- **Tags**: `git_commit`, `git_dirty`, `data_manifest_sha256` (hash of `data/manifest.json`, covering every tracked raw file in one fingerprint), `scenario` (`healthy` / `drift_natural` / `drift_injected` / `rollback_drill`). Run name: `<model_type>_<split_role>`, e.g. `hgbr_test`.
- **Metrics**, namespaced by split so train/calibration/test/replay never collide: `{split}/mae`, `{split}/rmse`, `{split}/wape`, `{split}/mae_vs_baseline_pct`, `{split}/worst_zone_mae`.
- **Per-zone and time-slice detail** is logged as an artifact report (`reports/metrics_by_zone.json`), not as individual metrics — only the worst-zone summary (`{split}/worst_zone_mae` metric + `{split}_worst_zone_id` tag) surfaces directly in the run view.

## Alternatives considered

Free-form key names decided per script, documented only in prose — rejected, invites drift between what the plan says and what code actually logs, with no way to catch it until someone reads the MLflow UI and finds inconsistent names across runs.

Logging every zone's metrics flat (`mae_zone_4`, `mae_zone_7`, ...) — rejected, 20 zones × 4 metrics × 5 splits is 400 metric keys per run; buries the overall signal and isn't how MLflow's UI is meant to be read. Full detail belongs in an artifact.

## Consequences

Any new script that logs to MLflow must import `taxi_mlops.tracking` instead of hardcoding key strings. Changing the naming scheme later means editing one file, not hunting across scripts. `src/taxi_mlops/` now exists as a package (`[build-system]`/`uv_build` added to `pyproject.toml`), matching the layout already started on Ali's unmerged `milestone/data-foundation` branch — intentional, to keep the eventual merge simple.
