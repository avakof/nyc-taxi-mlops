# 0003. Local environment: WSL, not native Windows

Status: Accepted

## Context

The dev machine runs Windows with WDAC (Windows Defender Application Control) enforced, which is centrally managed — not something a local, non-admin account can change (confirmed: not a local admin, and `CodeIntegrityPolicyEnforcementStatus: 2` = Enforced). WDAC blocks loading unsigned/untrusted compiled DLLs, which breaks `dvc` (via `pygit2`), `pyarrow`, and by extension `mlflow` (which pulls in pandas/numpy) when run from a native Windows Python environment.

## Decision

Run all data/ML tooling (`dvc`, `pyarrow`, `mlflow`, training, feature code) from WSL (Ubuntu), using a separate uv-managed virtualenv (`UV_PROJECT_ENVIRONMENT=.venv-wsl`) so it doesn't collide with the native Windows `.venv`. The repo lives on the Windows filesystem and is accessed from WSL via `/mnt/c/...`; both environments share the same `pyproject.toml`/`uv.lock`, with platform markers already correctly excluding Windows-only packages (e.g. `pywin32`) from the WSL resolve.

## Alternatives considered

Asking IT to allowlist the binaries — ruled out as too slow for the timeline, and not guaranteed to be grantable from a non-admin account on a centrally-managed policy. Running natively on Windows with DVC/MLflow dropped from scope — rejected, both are required project tools (per the project PDF's "what you build" list).

## Consequences

Every teammate on a similarly locked-down Windows machine needs the same WSL workaround; `README.md`'s setup instructions say so explicitly, scoped to Windows only (macOS/Linux teammates are unaffected and should ignore this). Not every native-Windows run actually fails this way in practice — `numpy`/`pandas`/`scikit-learn`/`mlflow` have run cleanly on Windows in this session without hitting WDAC, while `pygit2` and one `pyarrow` call did — so this is worth periodically re-checking rather than assumed universal, but WSL remains the default path for anything DVC/MLflow-related.
