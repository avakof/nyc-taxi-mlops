# Team contribution workflow

## Start a milestone

Group several related lessons and implementation tasks into one milestone branch. Make small commits as work progresses and push regularly. Open one pull request and request teammate review when the milestone's deliverables work together and its checks pass. Individual lessons do not require separate pull requests or contributor approval.

Track tasks through GitHub Issues and coordinate changes to the shared forecast contract and dependency files.

Start from current `main`, with no unrelated uncommitted changes:

```bash
git switch main
git pull --ff-only
git switch -c milestone/data-foundation
```

Use a descriptive milestone name. Once the branch exists, keep working on it across lessons; use `git switch milestone/data-foundation` to return to it. Start a new branch after completing and integrating the milestone.

`git pull --ff-only` updates the branch without creating an unexpected merge commit. If it reports divergent history, inspect the differences and coordinate before changing history.

## Implement and verify

Use the project's locked environment:

```bash
uv sync --locked
```

When intentionally adding a dependency, use `uv add` (or `uv add --dev` for development tooling) and commit both `pyproject.toml` and `uv.lock`. Commit `.python-version` when intentionally changing the Python selection.

Review your changes before staging:

```bash
git diff
git status --short
```

Stage the intended files by their actual paths. Run checks relevant to the behavior changed. Describe those checks in the pull request. When tests exist, use `uv run --locked pytest`; before that milestone, environment verification does not establish model or service correctness.

Keep raw datasets, model binaries, `.env`, `.venv/`, local databases, and credentials outside commits. Share safe configuration names/defaults through `.env.example`.

## Publish progress and request milestone review

After a coherent step, commit the intended files and publish the current branch:

```bash
git push -u origin HEAD
```

Continue implementing related tasks on the same branch. Pushing progress provides a shared backup and does not require opening a pull request.

When the milestone is ready, open the repository's **Pull requests** tab and select **New pull request**, targeting `main`. Explain the resulting behavior, completed deliverables, and verification. Link relevant issues. Request a teammate's review using GitHub's reviewer selector.

Address review feedback by adding commits to the same branch. Merge after milestone review and applicable checks pass. Delete the merged remote branch, then update your local `main` before starting the next milestone.

The review requirement is a team policy until repository rules explicitly enforce it. Do not assume an unconfigured CI workflow has verified a change.

## Review batches

| Milestone branch | Deliverables reviewed together |
|---|---|
| `milestone/data-foundation` | Configuration loader, tests, CI, verified TLC acquisition, data validation/preparation, and DVC reproduction |
| `milestone/training-tracking` | Baselines, chronological evaluation, MLflow provenance, model Registry, and training reproduction |
| `milestone/serving` | FastAPI, integration tests, model identity, Docker packaging, and release verification |
| `milestone/monitoring` | Causal replay, four monitoring layers, drift detection, dashboards, and alerts |
| `milestone/recovery` | Retraining trigger, candidate gates, promotion, rollback, incident report, and final clone/demo rehearsal |

These batches group the detailed checkpoints in the project plan. Learning questions and local verification continue during each lesson; peer review happens at milestone integration.

## Reproducibility responsibilities

Git identifies versioned code/configuration. DVC will identify and retrieve dataset snapshots. MLflow will record experiments and model versions. A teammate's local path or local-only cache is not a shared artifact store.

The project's final acceptance check includes reproducing the documented system from a clean clone, with independently accessible data and model artifacts.
