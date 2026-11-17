# NYC taxi demand: the MLOps layer

Team project for operating an hourly NYC yellow-taxi demand forecasting system. The objective is reproducible training, traceable deployment, monitoring, controlled retraining, and verified rollback.

Repository: [avakof/nyc-taxi-mlops](https://github.com/avakof/nyc-taxi-mlops).

**Current status:** data pipeline (download, DVC, zone-hour aggregation, chronological split), baselines, a tuned HGB model, and MLflow Registry (`candidate`/`champion`) are done — see [methodology.md](methodology.md)'s task table for exactly what's done vs. outstanding. Serving, drift detection, retraining, and rollback are not yet built.

## Join the project

The repository is private. Accept the GitHub collaborator invitation before cloning. Each teammate uses their own GitHub account and local environment.

Prerequisites: Git and [uv](https://docs.astral.sh/uv/). The project selects Python 3.12.14; uv can install it.

**Intel Mac only:** `cryptography` (pulled in transitively by DVC's `scmrepo`) has no prebuilt wheel for macOS x86_64 above version 42 — `uv sync` will build it from source, which needs OpenSSL dev headers and `pkg-config`:

```bash
brew install openssl@3 pkg-config
export PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig"
```

Apple Silicon Macs, Linux, and Windows are unaffected (wheels exist).

```bash
git clone https://github.com/avakof/nyc-taxi-mlops.git
cd nyc-taxi-mlops
uv python install 3.12.14
uv sync --locked
cp .env.example .env
```

Check the setup:

```bash
uv lock --check
uv run --locked python --version
uv run --locked pytest --version
uv run --locked ruff --version
```

Expected Python version: `3.12.14`. Package versions come from the committed `uv.lock`.

### Windows: use WSL, not native PowerShell/Git Bash

If your machine enforces Windows Defender Application Control (WDAC) — common on managed/school devices — `dvc`, `pyarrow`, and `mlflow` will fail to import with a DLL-load error, because WDAC blocks unsigned compiled extensions outside native Python/stdlib. This isn't fixable without admin/IT access to the policy.

Run everything data/ML-related from a WSL (Ubuntu) terminal instead. The repo stays on the Windows filesystem; WSL just reaches it via `/mnt/c/...`:

```bash
cd /mnt/c/path/to/nyc-taxi-mlops
export UV_PROJECT_ENVIRONMENT=.venv-wsl   # keep a separate venv from the Windows one
uv sync --locked
```

macOS and Linux don't have this restriction — run the setup commands above normally. See [docs/adr/0001-initial-architecture.md](docs/adr/0001-initial-architecture.md) ("Local environment: WSL, not native Windows") for why.

`.env` contains local settings and may later contain credentials. `.env.example` documents safe shared defaults, including `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME`/`MLFLOW_MODEL_NAME`. Copying `.env.example` does not start an MLflow server — run it yourself (see ADR-0001, "Experiment tracking") before running any script under `scripts/`.

## Reproduce the pipeline from a fresh clone

Run in order (same terminal — WSL on Windows, native on macOS/Linux):

```bash
# 1. Start MLflow (leave running in this terminal, or background it)
mkdir -p mlartifacts
uv run mlflow ui --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

# 2. In a second terminal: data, features, model, registry
uv run python scripts/download_data.py
uv run python scripts/aggregate_and_split.py
uv run python scripts/select_zones.py
uv run python scripts/baselines.py
uv run python scripts/tune_hgb.py
uv run python scripts/train_hgb.py          # note the run_id it prints at the end
uv run python scripts/register_baseline_champion.py
uv run python scripts/register_model.py <run_id_from_train_hgb_output>
uv run python scripts/evaluate_promotion.py # compares candidate vs. champion, records the decision

# 3. Tests
uv run pytest tests/ -v
```

This reproduces the data pipeline, baselines, tuned HGB model, and MLflow Registry state (`candidate`/`champion`) from nothing but the repo and the public TLC source. See `methodology.md`'s task table for what this does *not* yet cover (serving, drift, retraining, rollback).

## Work as a team

Use a short-lived branch for each task, push the branch, open a pull request, and ask a teammate to review it before merging into `main`. The detailed workflow is in [CONTRIBUTING.md](docs/CONTRIBUTING.md). Track work through GitHub Issues and reference the corresponding project milestone.

This is the team's contribution policy; no CI is configured yet.

## Forecasting contract

Predict expected pickups for each supported zone during `[T, T + 1 hour)`, using complete observations available by forecast issuance `T`. Read the [forecast contract](docs/forecast_contract.md) before changing aggregation, features, evaluation, or serving behavior.

## Data and artifact sharing

Source: [NYC Taxi and Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). January–April 2020 yellow-taxi Parquet records and the taxi-zone lookup, downloaded via `scripts/download_data.py` with a hashed manifest (`data/manifest.json`).

GitHub stores source code, configuration, tests, documentation, and DVC metadata (`data/*.dvc`). Raw trip files, local environments, credentials, runtime databases, and model binaries stay outside Git. DVC and MLflow Registry both currently use **local-only** storage (a local DVC cache, a local SQLite-backed MLflow server) — neither is yet accessible independently of this laptop. A shared DVC remote and a team-accessible MLflow server are still required before final submission; see ADR-0001's "Data and versioning" consequences.

The operational demonstration uses historical replay with explicit simulated hourly availability. TLC's monthly publication is not an hourly live feed.

## Project documents

- [Methodology and task assignment](methodology.md)
- [Architecture decision records](docs/adr/) — start at [0001](docs/adr/0001-initial-architecture.md)
- [Forecast contract](docs/forecast_contract.md)
- [Original project plan (superseded by methodology.md)](docs/PROJECT_PLAN.md)
- [Course review and adaptation notes](docs/COURSE_REVIEW.md)
- [Contribution workflow](docs/CONTRIBUTING.md)

The original `course/` reference assets are local, ignored materials. Teammates can use the project documents without those files; access the original course materials through the course's distribution channel.
