# 0007. Retain and restore complete serving deployments

Status: Accepted

## Context

Registry aliases and image tags can move. A service may pass its resource health check while its forecast route fails. Recovery must restore actual executable code and compatible model/feature/history resources.

## Decision

Retain local image content IDs with named tags and checksum-validated release copies. Capture the actual platform manifest, served release identity, and a known forecast before deployment. Restore with an explicit image ID and release path through `compose.deploy.yaml`; recreate the container without building or pulling. Verify readiness, model-info, the actual platform image, and a known forecast after restoration. Record wall-clock recovery duration separately from the historical forecast origin.

The isolated demonstration introduces a wrapper that returns HTTP 500 for forecasts while leaving resource health at 200. Its separate scenario-labelled bundle has the same history bytes. Recovery restores both the original image and bundle identity.

## Consequences

A single replica can have brief downtime. The provisional recovery target is 120 wall-clock seconds and is measured, not promised. The drill retains evidence and artifacts but cleans up its own containers. It does not modify the normal service or Registry aliases. Historical replay data is frozen; live observation storage must not be rewound during a future model rollback. A genuine demand change is evaluated through the drift/retraining workflow separately.

See [rollback runbook](../runbooks/rollback.md).

Recorded evidence: [injected serving rollback postmortem](../postmortems/0001-serving-rollback.md).
