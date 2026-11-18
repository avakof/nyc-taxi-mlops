# Restore a retained deployment

Related decision: [ADR-0007](../adr/0007-release-rollback.md).

## Retain before changing a deployment

Retain an immutable local image ID with a named tag, a checksum-validated model/features/zones/history bundle, a deployment record, and the result of a known forecast. A Registry alias is selection metadata; changing it does not replace the running container.

The current service is historical replay. The good and faulty bundles in the drill contain identical history bytes. This procedure does not implement database rollback or rewind observations. A future live history database must remain outside release rollback and maintain schema compatibility.

## Run the isolated drill

```bash
uv run --locked python scripts/rollback_demo.py
```

The command uses port 8002 and a unique Compose project. It retains the good image/bundle, builds a deliberately faulty wrapper image, verifies a healthy baseline, observes forecast HTTP 500 while health remains 200, restores the pinned good deployment, and checks image/platform/release identity and the known forecast. It writes `reports/rollback/<run>/result.json` and `known-good-deployment.json`. The demo containers are removed afterward; images, package copies, and evidence are retained. The normal service on port 8001 is not recreated.

## Restore an actual service

Select the previously verified `known-good-deployment.json`. Explicitly choose the target project and host port. For this repository's normal Compose project the project is normally `mlops`; verify with `docker compose ls` instead of assuming it on a differently named checkout.

```bash
uv run --locked python scripts/restore_release.py \
  --record reports/rollback/<run>/known-good-deployment.json \
  --project mlops \
  --port 8001
```

This recreates the named service and can briefly interrupt traffic. It uses `--no-build --pull never --force-recreate`, waits for health, verifies the served release identity, verifies the actual platform image, and compares a known forecast with the retained result. Recovery reports include duration and the provisional 120-second target. If a check fails, inspect the command error and container logs; do not mark recovery successful or change the Registry alias. Do not rebuild or retrain as a substitute for restoring the retained artifact.

## Retention and demonstration backup

Keep the named retained/fault images and `models/rollback-*` packages until the project is assessed. Generated images, packages, and reports stay outside Git. The drill is repeatable from the tracked scripts. An optional offline image archive can be created with `docker image save --output <archive.tar> <retained-tag>` and restored with `docker image load --input <archive.tar>`; do not commit the archive.

## Scope of the incident

This drill is an injected software deployment failure. It is separate from the March–April demand-drift scenario. A working older model is not assumed to fix a genuine change in taxi demand; Julia's drift/retraining evaluation must provide evidence for that decision.
