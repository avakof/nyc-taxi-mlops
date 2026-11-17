# 0005. MLflow Registry: naming and alias mechanics

Status: Accepted

## Context

ADR-0001 commits to Registry aliases (`candidate`/`champion`), not MLflow's deprecated stage system, but didn't settle the registered model's name or the concrete registration/verification mechanics.

## Decision

Registered model name: `nyc-taxi-demand-forecaster`, set via `MLFLOW_MODEL_NAME` (`.env`/`.env.example`), same pattern as `MLFLOW_EXPERIMENT_NAME`. Deliberately model-agnostic, not `nyc-taxi-hgb` — a single name needs to hold both a baseline wrapper and trained models as different versions over time; an HGB-specific name would have made that impossible.

`scripts/register_model.py <run_id>`:
1. `mlflow.register_model(f"runs:/{run_id}/model", model_name)` — creates a new immutable version.
2. `MlflowClient().set_registered_model_alias(model_name, "candidate", version)` — tags it `candidate`.
3. Loads that **exact numeric version** (`models:/{model_name}/{version}`, not by re-resolving the alias) and runs a smoke prediction, verifying it's nonnegative and finite, before declaring registration successful. This satisfies the plan's "resolve an alias once to an immutable version, export and validate that exact package" requirement at registration time, not deferred to serving.

This script never sets or moves `champion`. Promotion is intentionally a separate, deliberate action — the project's own adversarial-audit correction is explicit that automatic retraining must not be confused with automatic deployment.

## Alternatives considered

Bundling promotion to `champion` into the same script, since the HGB candidate already clearly beats both baselines on test (MAE 27.71 vs. 47.87) — rejected. Beating a baseline is necessary but the plan's candidate gates (ADR-0001 §8: ≥5% MAE improvement over champion specifically, no zone regressing >10%, passing smoke/readiness checks) are about replacing an existing champion, which doesn't exist yet — collapsing "register" and "promote" into one action would make the next real promotion decision (replacing *this* champion with a better candidate later) inconsistent with how the first one was done.

## Consequences

**Champion bootstrap, resolved:** `scripts/register_baseline_champion.py` wraps the prev-week (168h lag) baseline as an `mlflow.pyfunc.PythonModel` (`predict()` just returns the `lag_168` column — nothing fit, nothing pickled beyond the wrapper itself) and registers it directly as `champion` v1. This was the better-evidenced baseline (test MAE 47.87 vs. prev-day's 63.46), consistent with the project's "a well-operated simple model beats an unreproducible one" principle, and it means the HGB candidate's promotion gate ("≥5% lower MAE than champion") has a real, defined target from day one instead of a special-cased exception. The wrapper exists specifically so the Registry's alias mechanics (resolve-once, load-by-version) work identically regardless of whether `champion` happens to be a baseline or a trained model — serving code never needs to know which.

Order of operations: `register_baseline_champion.py` runs first (bootstraps `champion`), then `register_model.py <run_id>` registers the HGB run as `candidate` under the same registered model name. A later, separate promotion decision (not yet made) would move `champion` from the baseline to the HGB version once the gates in ADR-0001 §8 are actually checked against it — that comparison still needs to happen for real, this ADR only resolves how the *first* champion came to exist.
