# Presentation and backup

Prepare downloads, dependencies and images before the presentation. Never make a live demonstration depend on downloading months of TLC data or completing tuning.

1. Show `docker compose ps`, `/health`, `/model-info`, and a known forecast. Explain model version versus release identity.
2. Show the MLflow source run: params, MAE/RMSE/WAPE, Git commit and data hash. Show the DVC sidecars and clean `dvc status`.
3. Explain observation availability and complete zero versus missing data; demonstrate an unavailable-history 503.
4. Run `rollback_demo.py`: health remains 200 during the injected forecast failure, so a forecast probe is required. Show the actual image/release identity restored and the measured recovery duration.
5. Julia presents the separate drift/retraining evidence and candidate decision when that integration is available.

Backup: retain `reports/rollback/<run>/result.json`, the completed postmortem, the verified package and named image tags. A screen recording or captures can support the JSON evidence. State clearly when showing recorded evidence. Keep source files and the frozen data identity available offline; do not commit binaries or secrets.

The report distinguishes simulated forecast origin (2020) from UTC wall-clock execution timestamps (2026). A single probe latency is not a P95 measurement. Functional recovery of one known prediction does not establish model quality after demand drift.
