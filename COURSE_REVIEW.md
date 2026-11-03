# Course review and coverage

Reviewed 2026-11-03. The main agent and three delegated reviewers covered every file under `course/` before finalizing the project plan. This report distinguishes reading/parsing from executing working software.

**Coverage**

- 92 files, 60 distinct byte contents, verified with SHA-256.
- Four actual PDFs: `MLOps_EN.pdf` (58 pages), `MLFlow_EN.pdf` (44), `Docker_EN.pdf` (30), `APIs.pdf` (41): 173 total pages. All extracted text was read; every page was also inspected in rendered contact sheets, with enlarged inspection where needed.
- `02-github-actions.pdf` is HTML, titled “GitHub Actions: CI/CD for ML”. Its full lecture text was parsed and read; its bytes, including embedded resources, were inventoried. Embedded JavaScript was not executed.
- All seven lab directories were covered: two tracking copies, two container copies, two different serving labs, and one monitoring lab.
- Large CSV and lockfiles were parsed completely rather than sampling only their beginning. Both model archives were inspected without deserializing/executing an estimator.
- No course code was run. No course files were edited. Reading completed does not mean the exercises pass or that declared dependency versions have been tested together.

**Duplicate and variant findings**

- `course/01-mlflow` and `course/mlops-labs-main/01-mlflow` are identical: 10 file pairs.
- `course/lab-session-1` and `course/mlops-labs-main/02-docker-ci` are identical: 18 file pairs.
- `course/03-serving-1` and `course/mlops-labs-main/03-serving` are different. The first serves an MLflow alias; the second uses a local joblib artifact and contains intentional TODOs.
- Four transaction CSV copies contain the same 30,000 rows, with MD5 matching their supplied DVC pointers. The monitoring CSV contains 20,000 rows and differs from those copies.
- Two `.skops` archives each contain 143 members: recursive JSON model schema and 142 NumPy arrays, inspected without pickle loading.

**Course-to-project mapping**

| Material | Concepts retained | Adaptation required |
|---|---|---|
| MLOps lecture, especially pp.8, 14–16, 24–28, 33–40 | ML systems include data, software, operation, automation, and recovery; baseline-first delivery | Small local implementation; the tool landscape is not a required shopping list |
| MLflow lecture, pp.12–21, 29–38, 40–43 | Tracking, artifacts, executable environments, model documentation, DVC | Current aliases, explicit provenance, database backend; old screenshots are conceptual |
| MLflow labs | Pure training function, parameters/metrics/Git/data identity, Registry and reproduction | Time splits, actual-byte hashes, clean release code, finish missing checks |
| Docker lecture/lab | Images versus containers, build cache, Compose networking, persistence, smoke test | Pin compatible environment, keep model/data binaries out of Git, retain exact release artifacts |
| GitHub Actions lecture | PR checks, artifacts, immutable references, metric gates, controlled delivery | Use current supported actions and exact pins; full downloads/training separate from fast CI |
| API lecture/labs | Contract, lifespan loading, validation, model identity, real integration and latency tests | Forecast origin/history contract; readiness; exact version resolution; no copied random splits |
| Monitoring lab | Service, data, prediction, and outcome layers; delayed label joins | Event-time windows; separate reference/live sets; complete metrics; candidate trigger and recovery |

**Specific source findings**

1. `course/01-mlflow/train.py:37` uses random stratified splitting. Taxi forecasting requires chronological splits and causal features.
2. `course/01-mlflow/train.py:60` reads the dataset hash from the DVC pointer; it does not itself verify the bytes consumed. A release should verify both.
3. `course/01-mlflow/train.py:73` already implements tracking TODOs described as incomplete in its README. Actual source and exercise text differ.
4. `course/01-mlflow/register.py:13` leaves alias assignment as a TODO. `repro_check.py:22` contains an unresolved placeholder.
5. `course/01-mlflow/gen_data.py:20` changes the label-generating relationship while using the same feature-generation distributions. Distinguish concept change from input drift.
6. `course/03-serving-1/app.py:34` loads an alias before separately fetching the version. Resolve the immutable version once to avoid mislabeling a concurrently changed alias.
7. `course/mlops-labs-main/03-serving/app.py:55`, `:132`, and `test_app.py:43` contain teaching TODOs. These are not passing implementation evidence.
8. `course/lab-session-1/export_model.py:15` removes the old exported model before the replacement download succeeds. Stage and validate replacements before switching.
9. `course/lab-session-1/README.md:135` suggests committing `model/`. The project instead keeps model binaries in artifact storage and data in DVC.
10. `course/mlops-labs-main/04-monitoring/train_model.py:19` trains from the full CSV with a random split, but `app.py:70` calls its first half the training reference and `traffic.py:23` replays the second half. Training/live separation is therefore not clean.
11. The monitoring app leaves PSI, flagged counting, latency, and prediction-log fields incomplete; `prometheus/alerts.yml:19` has placeholder threshold 99. Outcome monitoring is a text panel/script, not a complete live layer.
12. `course/mlops-labs-main/04-monitoring/join_labels.py:16` uses an inner join without explicit coverage reporting. Missing labels can be hidden.
13. Monitoring and serving labs use a short data hash as model identity. It cannot distinguish code/configuration/environment changes on the same data.
14. The monitoring lab has no completed retraining trigger, promotion controller, or rollback runbook. These are additional project work.
15. MLflow lecture pp.22/31 show an MLflow 2.2.2/Python 3.9.13/scikit-learn 1.0.2 example from 2023; the tracking lock records MLflow 3.17.1 and Python >=3.12. Do not treat screenshots as version specifications.
16. MLflow lecture pp.35–36 uses older stage terminology; the newer lab uses aliases. Current official documentation confirms stages are deprecated.
17. API lecture p.7 mixes an image request and fraud response and is not a reusable contract. Its latency plots are illustrations, not measured project benchmarks.
18. API lecture pp.35–37 supports measuring client/server/model latency separately. Independently calculated P95 values cannot generally be added or subtracted to recover a component P95.
19. Docker lecture simplifications need qualification: containers share the runtime host kernel; on macOS the Linux runtime runs in a VM. Not every Dockerfile instruction adds a filesystem layer.
20. The course's local DVC remote and permissive Compose examples require adaptation for cross-machine reproducibility, secret handling, durable state, and local-only exposure.

**Limits of this review**

Dependency versions and archive contents are observations from the files, not certification that they are current, safe, or compatible. Large machine-generated files were validated structurally; embedded binary/script bytes were not manually interpreted instruction by instruction. No claim is made that the provided labs run successfully. Actual TLC data has not been downloaded or profiled during planning.

**File inventory**

The shortened SHA-256 below is an inventory fingerprint. Project dataset verification will use full hashes.

| File | Bytes | SHA-256 prefix | Review method |
|---|---:|---|---|
| `course/01-mlflow/.gitignore` | 48 | `30cf742327dc95a2` | Full text/structured-content read |
| `course/01-mlflow/README.md` | 5,779 | `43dda6f56cc7bbe5` | Full text/structured-content read |
| `course/01-mlflow/gen_data.py` | 1,683 | `d07eef72e3b6d75c` | Full text/structured-content read |
| `course/01-mlflow/params.yaml` | 70 | `6f53226000c3d2b6` | Full text/structured-content read |
| `course/01-mlflow/pyproject.toml` | 203 | `7c15554f281bb444` | Full text/structured-content read |
| `course/01-mlflow/register.py` | 583 | `d8e18c79bb176ecb` | Full text/structured-content read |
| `course/01-mlflow/repro_check.py` | 1,216 | `ec74bf9d968933d7` | Full text/structured-content read |
| `course/01-mlflow/score.py` | 925 | `fbdc4d00b5b9bc6e` | Full text/structured-content read |
| `course/01-mlflow/train.py` | 3,392 | `3fc032f05c203d8b` | Full text/structured-content read |
| `course/01-mlflow/uv.lock` | 711,073 | `99e552b98bab7da4` | Full TOML parse; dependencies and artifact metadata inspected |
| `course/02-github-actions.pdf` | 5,621,502 | `38ca931229c60a6b` | Complete HTML lecture parsed; embedded assets inspected as file content |
| `course/03-serving-1/README.md` | 5,935 | `30cddcbc62733cbb` | Full text/structured-content read |
| `course/03-serving-1/app.py` | 3,469 | `c14a15e5dc63be33` | Full text/structured-content read |
| `course/03-serving-1/data/transactions.csv` | 1,043,632 | `23adb5d988ef3464` | All records parsed; schema/statistics checked |
| `course/03-serving-1/data/transactions.csv.dvc` | 99 | `75282fa7734cc891` | Full text/structured-content read |
| `course/03-serving-1/locustfile.py` | 938 | `ca4e6fcce7af3293` | Full text/structured-content read |
| `course/03-serving-1/pyproject.toml` | 301 | `a02700c20c45e4ce` | Full text/structured-content read |
| `course/03-serving-1/test_app.py` | 1,192 | `2a9701806d7f4f24` | Full text/structured-content read |
| `course/03-serving-1/uv.lock` | 614,540 | `7c01c768d36bc965` | Full TOML parse; dependencies and artifact metadata inspected |
| `course/APIs.pdf` | 1,369,658 | `92c30e6a806d57ec` | Full extracted text + all rendered pages |
| `course/Docker_EN.pdf` | 1,366,779 | `30020ea2eadc4df8` | Full extracted text + all rendered pages |
| `course/MLFlow_EN.pdf` | 2,440,128 | `ac1137d3603e5276` | Full extracted text + all rendered pages |
| `course/MLOps_EN.pdf` | 9,415,573 | `c01ef6e47f6c7327` | Full extracted text + all rendered pages |
| `course/lab-session-1/.dockerignore` | 63 | `7df8c7620410912f` | Full text/structured-content read |
| `course/lab-session-1/Dockerfile` | 417 | `b1cbc71bcb2e96dd` | Full text/structured-content read |
| `course/lab-session-1/README.md` | 7,287 | `9c500bb42543ab7e` | Full text/structured-content read |
| `course/lab-session-1/data/transactions.csv` | 1,043,632 | `23adb5d988ef3464` | All records parsed; schema/statistics checked |
| `course/lab-session-1/data/transactions.csv.dvc` | 99 | `75282fa7734cc891` | Full text/structured-content read |
| `course/lab-session-1/export_model.py` | 633 | `ad2dbeb0ab3c2839` | Full text/structured-content read |
| `course/lab-session-1/model/MLmodel` | 1,460 | `9f4150c3e51dd3ba` | Full text/structured-content read |
| `course/lab-session-1/model/conda.yaml` | 3,558 | `bdac2650d5364edb` | Full text/structured-content read |
| `course/lab-session-1/model/input_example.json` | 195 | `744dd78dfbc91173` | Full text/structured-content read |
| `course/lab-session-1/model/model.skops` | 397,121 | `54370825e8461e7a` | All archive members/schema/NumPy arrays inspected; no estimator execution |
| `course/lab-session-1/model/python_env.yaml` | 106 | `c5d26348f93b918f` | Full text/structured-content read |
| `course/lab-session-1/model/registered_model_meta` | 46 | `83366eeb5c97da71` | Full text/structured-content read |
| `course/lab-session-1/model/requirements.txt` | 2,853 | `bdeb55137b5d0378` | Full text/structured-content read |
| `course/lab-session-1/model/serving_input_example.json` | 466 | `1179d2069373832e` | Full text/structured-content read |
| `course/lab-session-1/predict.py` | 755 | `7eebee2e98a980e0` | Full text/structured-content read |
| `course/lab-session-1/pyproject.toml` | 342 | `f86e6f2bc2ce53dd` | Full text/structured-content read |
| `course/lab-session-1/sample.csv` | 3,524 | `00d651ab4861177d` | All records parsed; schema/statistics checked |
| `course/lab-session-1/uv.lock` | 242,219 | `7cd243ae0afc3066` | Full TOML parse; dependencies and artifact metadata inspected |
| `course/mlops-labs-main/01-mlflow/.gitignore` | 48 | `30cf742327dc95a2` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/README.md` | 5,779 | `43dda6f56cc7bbe5` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/gen_data.py` | 1,683 | `d07eef72e3b6d75c` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/params.yaml` | 70 | `6f53226000c3d2b6` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/pyproject.toml` | 203 | `7c15554f281bb444` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/register.py` | 583 | `d8e18c79bb176ecb` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/repro_check.py` | 1,216 | `ec74bf9d968933d7` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/score.py` | 925 | `fbdc4d00b5b9bc6e` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/train.py` | 3,392 | `3fc032f05c203d8b` | Full text/structured-content read |
| `course/mlops-labs-main/01-mlflow/uv.lock` | 711,073 | `99e552b98bab7da4` | Full TOML parse; dependencies and artifact metadata inspected |
| `course/mlops-labs-main/02-docker-ci/.dockerignore` | 63 | `7df8c7620410912f` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/Dockerfile` | 417 | `b1cbc71bcb2e96dd` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/README.md` | 7,287 | `9c500bb42543ab7e` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/data/transactions.csv` | 1,043,632 | `23adb5d988ef3464` | All records parsed; schema/statistics checked |
| `course/mlops-labs-main/02-docker-ci/data/transactions.csv.dvc` | 99 | `75282fa7734cc891` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/export_model.py` | 633 | `ad2dbeb0ab3c2839` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/MLmodel` | 1,460 | `9f4150c3e51dd3ba` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/conda.yaml` | 3,558 | `bdac2650d5364edb` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/input_example.json` | 195 | `744dd78dfbc91173` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/model.skops` | 397,121 | `54370825e8461e7a` | All archive members/schema/NumPy arrays inspected; no estimator execution |
| `course/mlops-labs-main/02-docker-ci/model/python_env.yaml` | 106 | `c5d26348f93b918f` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/registered_model_meta` | 46 | `83366eeb5c97da71` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/requirements.txt` | 2,853 | `bdeb55137b5d0378` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/model/serving_input_example.json` | 466 | `1179d2069373832e` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/predict.py` | 755 | `7eebee2e98a980e0` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/pyproject.toml` | 342 | `f86e6f2bc2ce53dd` | Full text/structured-content read |
| `course/mlops-labs-main/02-docker-ci/sample.csv` | 3,524 | `00d651ab4861177d` | All records parsed; schema/statistics checked |
| `course/mlops-labs-main/02-docker-ci/uv.lock` | 242,219 | `7cd243ae0afc3066` | Full TOML parse; dependencies and artifact metadata inspected |
| `course/mlops-labs-main/03-serving/README.md` | 5,068 | `c197c434488b234c` | Full text/structured-content read |
| `course/mlops-labs-main/03-serving/app.py` | 5,678 | `438803e243002230` | Full text/structured-content read |
| `course/mlops-labs-main/03-serving/data/transactions.csv` | 1,043,632 | `23adb5d988ef3464` | All records parsed; schema/statistics checked |
| `course/mlops-labs-main/03-serving/locustfile.py` | 984 | `b847f92141a4b195` | Full text/structured-content read |
| `course/mlops-labs-main/03-serving/pyproject.toml` | 518 | `66170a65187f2cef` | Full text/structured-content read |
| `course/mlops-labs-main/03-serving/test_app.py` | 1,390 | `c8d8eb82325b5719` | Full text/structured-content read |
| `course/mlops-labs-main/03-serving/train_model.py` | 1,318 | `b35058cad7ab115a` | Full text/structured-content read |
| `course/mlops-labs-main/03-serving/uv.lock` | 466,254 | `157330e90beb63b6` | Full TOML parse; dependencies and artifact metadata inspected |
| `course/mlops-labs-main/04-monitoring/README.md` | 4,506 | `713d94cacd8a5cbe` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/alertmanager/alertmanager.yml` | 263 | `6683f0af6d42d3c8` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/app.py` | 6,510 | `0428519b431cf125` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/data/transactions.csv` | 695,691 | `fbe4a9a9ba7e6ab0` | All records parsed; schema/statistics checked |
| `course/mlops-labs-main/04-monitoring/docker-compose.yml` | 959 | `7af2b26b698afb25` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/grafana/dashboards/fraud.json` | 4,748 | `57a87741a10355d0` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/grafana/provisioning/dashboards/dashboards.yml` | 109 | `434b2bf56e6eb846` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/grafana/provisioning/datasources/prometheus.yml` | 159 | `19d98cca02290926` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/join_labels.py` | 1,251 | `e9573e7bb6b65b6b` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/prometheus/alerts.yml` | 1,301 | `8d3b6ebb5d6b9521` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/prometheus/prometheus.yml` | 444 | `7a2e6b267dab9ac9` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/pyproject.toml` | 355 | `715dd75ce82adb5f` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/traffic.py` | 4,099 | `77a544b2079b45c6` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/train_model.py` | 1,318 | `b35058cad7ab115a` | Full text/structured-content read |
| `course/mlops-labs-main/04-monitoring/uv.lock` | 223,698 | `77b6f124c0f36876` | Full TOML parse; dependencies and artifact metadata inspected |
