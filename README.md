# NYC taxi demand: the MLOps layer

Team project for operating an hourly NYC yellow-taxi demand forecasting system. The objective is reproducible training, traceable deployment, monitoring, controlled retraining, and verified rollback.

Repository: [avakof/nyc-taxi-mlops](https://github.com/avakof/nyc-taxi-mlops).

**Current milestone:** forecast contract, Git history, locked Python environment, configuration template, and team collaboration setup. Training, serving, data versioning, monitoring, and recovery are subsequent milestones in [methodology.md](methodology.md).

## Join the project

The repository is private. Accept the GitHub collaborator invitation before cloning. Each teammate uses their own GitHub account and local environment.

Prerequisites: Git and [uv](https://docs.astral.sh/uv/). The project selects Python 3.12.14; uv can install it.

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

Expected Python version: `3.12.14`. Package versions come from the committed `uv.lock`. The current checks verify environment setup; model/API tests arrive with their implementation.

### Windows: use WSL, not native PowerShell/Git Bash

If your machine enforces Windows Defender Application Control (WDAC) — common on managed/school devices — `dvc`, `pyarrow`, and `mlflow` will fail to import with a DLL-load error, because WDAC blocks unsigned compiled extensions outside native Python/stdlib. This isn't fixable without admin/IT access to the policy.

Run everything data/ML-related from a WSL (Ubuntu) terminal instead. The repo stays on the Windows filesystem; WSL just reaches it via `/mnt/c/...`:

```bash
cd /mnt/c/path/to/nyc-taxi-mlops
export UV_PROJECT_ENVIRONMENT=.venv-wsl   # keep a separate venv from the Windows one
uv sync --locked
```

macOS and Linux don't have this restriction — run the setup commands above normally. See [docs/adr/0001-initial-architecture.md](docs/adr/0001-initial-architecture.md) ("Local environment: WSL, not native Windows") for why.

`.env` contains local settings and may later contain credentials. `.env.example` documents safe shared defaults. Configuration loading will be implemented with the application; copying this file does not start an MLflow server.

## Work as a team

Use a short-lived branch for each task, push the branch, open a pull request, and ask a teammate to review it before merging into `main`. The detailed workflow is in [CONTRIBUTING.md](docs/CONTRIBUTING.md). Track work through GitHub Issues and reference the corresponding project milestone.

This is the team's contribution policy. Enforced branch rules and automated CI checks will be added and documented when configured.

## Forecasting contract

Predict expected pickups for each supported zone during `[T, T + 1 hour)`, using complete observations available by forecast issuance `T`. Read the [forecast contract](docs/forecast_contract.md) before changing aggregation, features, evaluation, or serving behavior.

## Data and artifact sharing

Source: [NYC Taxi and Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Planned initial source months: January–April 2020 yellow-taxi Parquet records and the taxi-zone lookup.

GitHub stores source code, configuration, tests, documentation, and future DVC metadata. Raw trip files, local environments, credentials, runtime databases, and model binaries stay outside Git. The approved milestones will introduce a team-accessible DVC data remote and MLflow artifact/Registry access. Those storage services must be accessible independently of an individual developer's laptop.

The operational demonstration uses historical replay with explicit simulated hourly availability. TLC's monthly publication is not an hourly live feed.

## Project documents

- [Methodology and task assignment](methodology.md)
- [Forecast contract](docs/forecast_contract.md)
- [Original project plan (superseded by methodology.md)](docs/PROJECT_PLAN.md)
- [Course review and adaptation notes](docs/COURSE_REVIEW.md)
- [Contribution workflow](docs/CONTRIBUTING.md)

The original `course/` reference assets are local, ignored materials. Teammates can use the project documents without those files; access the original course materials through the course's distribution channel.
