# Team contribution workflow

## Start a task

Choose one project milestone or GitHub Issue. Keep the change small enough for a teammate to review. Agree in the issue who owns the task; coordinate changes to the shared forecast contract and dependency files.

Start from current `main`, with no unrelated uncommitted changes:

```bash
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Replace `feature/short-description` with a descriptive branch name such as `feature/hourly-aggregation` or `docs/update-setup`.

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

## Publish a branch and request review

After committing the task, publish the current branch:

```bash
git push -u origin HEAD
```

Open the repository's **Pull requests** tab and select **New pull request**, targeting `main`. Explain the problem, resulting behavior, and verification. Link the relevant issue. Request a teammate's review using GitHub's reviewer selector.

Address review feedback by adding commits to the same branch. Merge after review and applicable checks pass. Delete the merged remote branch, then update your local `main` before starting another task.

The review requirement is a team policy until repository rules explicitly enforce it. Do not assume an unconfigured CI workflow has verified a change.

## Reproducibility responsibilities

Git identifies versioned code/configuration. DVC will identify and retrieve dataset snapshots. MLflow will record experiments and model versions. A teammate's local path or local-only cache is not a shared artifact store.

The project's final acceptance check includes reproducing the documented system from a clean clone, with independently accessible data and model artifacts.
