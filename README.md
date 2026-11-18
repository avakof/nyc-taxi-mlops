# NYC taxi demand: the MLOps layer

A reproducible historical-replay service forecasting next-hour yellow-taxi pickups for 20 frozen NYC zones. The serving milestone provides exact model/release identity, causal historical features, FastAPI, integration tests, locked Docker packaging, and measured rollback.

**Current serving model:** the parameter-free 168-elapsed-hour seasonal baseline. The existing training scripts also reproduce a tuned HGB candidate. Model quality is secondary to reproducibility and operations.

**Remaining team work:** drift detection, retraining-trigger integration, and the joined demand-drift incident are Julia's workflow. The bad-deployment rollback drill below is separate evidence. Track status in [methodology.md](methodology.md).

## Prerequisites

- Git, [uv](https://docs.astral.sh/uv/), Docker Desktop/Engine with Docker Compose.
- Python **3.12.14**, selected by `.python-version`; uv can install it.
- Approximately 235 MB of public TLC downloads, plus Python/image storage. Build and download before presenting.
- This repository is private; accept the collaborator invitation before cloning. Before merge, review this work with `git clone --branch milestone/serving <repository-url>`.

On Intel Macs, locked `cryptography==50.0.2` needs a source build. Install its prerequisites before syncing:

```bash
brew install openssl@3 pkg-config
export PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig"
```

On Windows, use WSL with Docker Desktop's WSL integration. Native managed Windows can block compiled Python dependencies through WDAC. The course/team decision is [ADR-0003](docs/adr/0003-wsl-local-environment.md).

## Fresh clone to working service

Run from the repository root:

```bash
git clone https://github.com/avakof/nyc-taxi-mlops.git
cd nyc-taxi-mlops
uv python install 3.12.14
uv sync --locked
cp .env.example .env
uv run --locked pytest -q
uv run --locked python scripts/bootstrap.py --accept-complete-snapshot
docker compose up --build --detach --wait --wait-timeout 120
uv run --locked python scripts/smoke_api.py --base-url http://127.0.0.1:8001 --expected-release models/final-serving/release.json
```

Expected: all tests pass, Docker reports a healthy container, and the smoke result is `PASS` with **586.0** pickups for zone 237 at `2020-02-20T15:00:00Z`.

The bootstrap requires a clean Git working tree. It downloads and checks source hashes, rebuilds hourly counts, checks that outputs match the committed DVC snapshots, populates the local DVC cache, and creates a baseline champion only if this local Registry has none. It starts its own temporary-port MLflow process, preserves the database/artifacts under `.runtime/mlflow/`, exports `models/final-serving`, and stops only its own process. Run/version IDs are local and may differ between clones; equivalent source bytes, settings and predictions are the reproduction criteria.

`--accept-complete-snapshot` explicitly accepts the documented replay assumption: accepted monthly snapshots are complete and hourly observations become available at hour closure. Existing completeness and availability metadata are preserved. Missing files, missing hours, incomplete observations, or late observations are not filled with zero.

Generated data, models, runtime state and secrets are ignored by Git. An existing release is retained; choose a new `--output` directory when the code changes. Re-running bootstrap against an unchanged, validated output reports `ALREADY_PREPARED`.

## API contract

The container is available at `http://127.0.0.1:8001`; interactive documentation is `/docs`. The container itself listens on port 8000.

| Route | Behavior |
| --- | --- |
| `GET /health` | 200 when the model, bundle and history source are usable; otherwise 503 |
| `GET /model-info` | Loaded model name, numeric Registry version, source run, release ID and code identity |
| `POST /forecast` | Validate zone/origin, construct eligible historical features, predict `[T, T+1 hour)` |

```bash
curl -sS http://127.0.0.1:8001/forecast \
  -H 'Content-Type: application/json' \
  -d '{"zone_id":237,"forecast_origin":"2020-02-20T15:00:00Z"}'
```

Use timezone-aware, UTC-hour-aligned origins and a supported zone. Invalid requests return 422; unavailable required history/model returns 503; unexpected inference errors or invalid outputs return 500. Expected counts may be fractional. Read [the forecast contract](docs/forecast_contract.md) and [ADR-0006](docs/adr/0006-serving-api-contract.md).

For the January–February snapshot, history begins `2020-01-01T05:00:00Z`; 168-hour features first become available at `2020-01-08T05:00:00Z`. Forecasts after the stored history ends lack recent observations and return 503. UTC defines elapsed lags; New York time defines calendar features. Request-level history checks apply even when global health is 200.

The current public bootstrap is the parameter-free baseline. Native HGB loading is integration-tested; later trained-model replay must use models trained before the forecast origin and the corresponding validated feature contract. The serving layer validates observation timing; orchestration of model-training cutoffs belongs to the candidate/replay integration.

## Inspect MLflow and reproduce the trained candidate

The bootstrap Registry and artifacts persist locally. To inspect them in a separate terminal:

```bash
uv run --locked mlflow server --host 127.0.0.1 --port 5050 \
  --backend-store-uri sqlite:///.runtime/mlflow/mlflow.db \
  --artifacts-destination .runtime/mlflow/artifacts
```

Open `http://127.0.0.1:5050`. Baseline source runs record parameters, test MAE/RMSE/WAPE, Git identity and the data-manifest hash.

The optional full modeling pipeline uses the same dataset and frozen zones. With the UI/server running, use a second terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_REGISTRY_URI=http://127.0.0.1:5050
uv run --locked python scripts/baselines.py
uv run --locked python scripts/tune_hgb.py
uv run --locked python scripts/train_hgb.py
# Use the printed training run ID:
uv run --locked python scripts/register_model.py <run_id>
uv run --locked python scripts/evaluate_promotion.py
```

The candidate evaluation records its decision; it does not automatically replace the running container. `register_baseline_champion.py` is initial setup only and directly assigns `champion`; do not use it to overwrite an established champion. Bootstrap avoids that reset. Export selected versions to a new output directory; retain the prior image and bundle before deployment.

## Rollback and incident evidence

```bash
uv run --locked python scripts/rollback_demo.py
```

The isolated project uses port 8002. It verifies a good deployment, introduces a distinct faulty image whose forecasts return 500 while health stays 200, restores the original image and bundle, and verifies the known prediction. It retains two images, bundles, deployment records and JSON evidence, then removes its demo containers. The normal service on port 8001 remains available.

The recorded user-run drill recovered in **21.47 wall-clock seconds**. This is one measured run, not an availability guarantee. See the [completed postmortem](docs/postmortems/0001-serving-rollback.md), [rollback runbook](docs/runbooks/rollback.md), and [ADR-0007](docs/adr/0007-release-rollback.md). The runbook includes the explicit `restore_release.py` command for an actual retained deployment.

Do not rewind live observations when restoring a model. This demo has identical frozen history in its good/faulty bundles. Demand collapse is a separate drift scenario requiring measured evaluation and an appropriate retraining/fallback decision.

## Verification and collaboration

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv lock --check
```

Tests create their own synthetic data and temporary real MLflow server. They need no TLC download or existing local service. Pytest discovers only `tests/`. GitHub Actions runs the locked checks and a Docker build on pushes/PRs; the first remote result must be checked before merge.

Use one milestone PR after the batch is complete, request Julia's review, and merge after review/checks. See [contribution instructions](docs/CONTRIBUTING.md). Raw Parquet, `.env`, local registries, generated model packages and image archives never belong in Git.

DVC and MLflow storage remain local to each clone. Public-source download plus hash validation and DVC cache creation make the demo independently reproducible. Shared artifact storage/Registry would be required for a central team deployment; it is a documented limitation of this local demonstration.

## Demonstration backup

Rehearse before presenting. Keep the retained image/bundle, rollback JSON, and completed postmortem available. Show the API identity and known forecast, then the isolated failure/recovery. If the live demonstration fails, use the saved evidence and identify it as a recorded run. See [presentation guide](docs/PRESENTATION.md). An optional image archive uses `docker image save`/`docker image load`; archives stay outside Git.

Source: [NYC Taxi and Limousine Commission trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), yellow taxi January–April 2020 and taxi-zone lookup. TLC monthly publication is not an hourly live feed. March–April files are downloaded for the later drift workflow; the current serving snapshot covers warmup through the healthy test period.

The original `course/` assets are ignored reference materials; obtain them from the course distribution. Teammates need only this repository and its documented prerequisites to run the system.
