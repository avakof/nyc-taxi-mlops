# 0008. Reproducible local bootstrap and clean final releases

Status: Proposed

## Context

Git does not contain raw data, model packages or a running Registry. Re-downloading identical source bytes previously changed the manifest timestamp and data fingerprint. A fresh clone needs an executable path to a working service.

## Decision

Keep the first accepted snapshot timestamp when source metadata and hashes are unchanged. Download from TLC, verify the frozen SHA-256 values, rebuild counts, reject mismatches against committed DVC snapshots, then populate the local DVC cache. Require a clean source checkout for final releases.

Bootstrap uses a temporary-port MLflow server with persistent local state under `.runtime/mlflow`, creates a baseline champion only when none exists, and exports an immutable checked package to `models/final-serving`. It stops only its owned server. Baseline runs log params, test metrics, code identity and data hash. The API and Compose defaults point to the final package. A smoke check can assert the intended release ID.

## Consequences

Each clone creates its own run/version IDs; equivalence is defined by source/configuration identity and predictions rather than equal local IDs. Shared Registry/DVC hosting remains a team-deployment extension. The current demonstration is historical replay with an explicitly accepted completeness/availability policy, not a live TLC feed. A baseline keeps bootstrap fast; HGB training remains an optional separately evaluated pipeline. Future candidate integration must enforce training/release event-time eligibility and configuration compatibility.
