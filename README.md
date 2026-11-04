# NYC taxi demand: the MLOps layer

Team project for operating an hourly NYC yellow-taxi demand forecasting system. The objective is reproducible training, traceable deployment, monitoring, controlled retraining, and verified rollback.

Repository: [avakof/nyc-taxi-mlops](https://github.com/avakof/nyc-taxi-mlops).

**Current milestone:** forecast contract, Git history, locked Python environment, configuration template, and team collaboration setup. Training, serving, data versioning, monitoring, and recovery are subsequent milestones in the [approved project plan](PROJECT_PLAN.md).

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

`.env` contains local settings and may later contain credentials. `.env.example` documents safe shared defaults. Configuration loading will be implemented with the application; copying this file does not start an MLflow server.

## Work as a team

Group several related lessons and tasks on one milestone branch. Commit small completed changes and push progress regularly. Open one pull request and request teammate approval when the milestone's deliverables work together and its checks pass. Keep using the same branch between lessons. The review batches and detailed workflow are in [CONTRIBUTING.md](CONTRIBUTING.md).

This is the team's contribution policy. Enforced branch rules and automated CI checks will be added and documented when configured.

## Forecasting contract

Predict expected pickups for each supported zone during `[T, T + 1 hour)`, using complete observations available by forecast issuance `T`. Read the [forecast contract](docs/forecast_contract.md) before changing aggregation, features, evaluation, or serving behavior.

## Data and artifact sharing

Source: [NYC Taxi and Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Planned initial source months: January–April 2020 yellow-taxi Parquet records and the taxi-zone lookup.

GitHub stores source code, configuration, tests, documentation, and future DVC metadata. Raw trip files, local environments, credentials, runtime databases, and model binaries stay outside Git. The approved milestones will introduce a team-accessible DVC data remote and MLflow artifact/Registry access. Those storage services must be accessible independently of an individual developer's laptop.

The operational demonstration uses historical replay with explicit simulated hourly availability. TLC's monthly publication is not an hourly live feed.

## Project documents

- [Approved plan and acceptance criteria](PROJECT_PLAN.md)
- [Forecast contract](docs/forecast_contract.md)
- [Course review and adaptation notes](COURSE_REVIEW.md)
- [Contribution workflow](CONTRIBUTING.md)

The original `course/` reference assets are local, ignored materials. Teammates can use the project documents without those files; access the original course materials through the course's distribution channel.
