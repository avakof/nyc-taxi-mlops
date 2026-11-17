# 0004. Defer feature pruning to SHAP, not the correlation matrix

Status: Accepted

## Context

The EDA (`notebooks/01_eda.ipynb`) found strong correlation among the lag/rolling features (e.g. `roll_3` ≈ mean of `lag_1`/`lag_2`/`lag_3`), expected given pickup demand's high hour-to-hour autocorrelation (0.975 at lag 1h, 0.963 at 24h, 0.924 at 168h on the top training zone). The question: does this multicollinearity mean some features should be dropped before training?

## Decision

Keep the full feature set from `docs/forecast_contract.md` as-is going into training; do not pre-prune based on the correlation matrix. `HistGradientBoostingRegressor` (ADR-0001, Modeling) splits on one feature at a time, so correlated features don't destabilize it the way they would a linear model — correlation-based pre-filtering is the standard move for linear/regularized models, not GBMs. Defer any trimming to **feature importance / SHAP values computed from the actual trained model**.

`scripts/train_hgb.py` logs SHAP values as an MLflow artifact on every training run (`shap/shap_values.npz`, `shap/shap_summary.json`, naming centralized in `taxi_mlops.tracking`), so this evidence exists without needing a retrain later. (Initially via `shap.TreeExplainer` — see the fix further down; this line described the state before that was found broken.)

## Alternatives considered

Trimming `roll_3` (the most obviously redundant feature) immediately, based on the correlation matrix alone — rejected as premature: correlation doesn't capture whether a tree model actually uses a feature differently across subpopulations (zones, time-of-day), and the project's own stated priority is evidence-based decisions, not statistical folklore.

## Consequences

First real SHAP run (via `scripts/train_hgb.py`) found `lag_2`, `roll_3`, and `lag_168` as the dominant features, with `zone_id` at exactly `0.0` mean |SHAP|. That exact-zero result was flagged as worth independently verifying rather than accepted outright.

**Verification result (`notebooks/02_feature_importance.ipynb`):** cross-checked against `sklearn.inspection.permutation_importance` on the same model and test split. The two methods disagree sharply — SHAP's top two (`lag_2`, `roll_3`) rank near the bottom in permutation importance, and permutation importance's clear #1 (`lag_1`) ranks near the bottom in SHAP. They agree only on `lag_168` (3rd by both). Permutation importance shows `zone_id` as a real, mid-tier contributor (4.85 MAE increase), directly contradicting SHAP's `0.0`.

This is too large a disagreement to be ordinary multicollinearity noise — it points to a real problem with the SHAP computation, most likely `TreeExplainer`'s interaction with `categorical_features="from_dtype"` (native categorical bitset splits). **Decision: reject the SHAP-based `zone_id` finding; make no feature-set change from this evidence.**

**Root cause, confirmed (`notebooks/02_feature_importance.ipynb`):** tried `TreeExplainer` with `feature_perturbation="interventional"` instead of the default `tree_path_dependent` — this did not resolve the disagreement (it got worse: `lag_1` and `hour_of_day` both dropped to exactly `0.0` too). Cross-checked against a fully model-agnostic explainer (`shap.explainers.Permutation`, which only calls `model.predict` and never inspects tree structure) — this matches permutation importance almost exactly, feature-for-feature (`lag_1` top at 86.5 vs. permutation's 82.6; `zone_id` mid-tier at 7.07 vs. 4.85). **Conclusion: `TreeExplainer` is broken for this model in both modes**, specifically for `HistGradientBoostingRegressor` with `categorical_features="from_dtype"` — not a perturbation-assumption nuance. The model-agnostic explainer and permutation importance are both trustworthy; `TreeExplainer`'s output for this model is not.

**Fix applied:** `scripts/train_hgb.py` now computes SHAP via `shap.explainers.Permutation` (model-agnostic) instead of `shap.TreeExplainer`. Any future feature-set change may use this SHAP output or permutation importance as evidence — not output from `TreeExplainer` on this model family.

**Sample size, measured not guessed:** the model-agnostic explainer is far more expensive than `TreeExplainer` (confirmed firsthand — a 5-point convergence sweep up to n=800 used 7-8 cores for a substantial run), so the explain-sample size needed justification, not a round number. Measured worst-case relative standard error (std/√n, max across all 12 features) at increasing sample sizes: n=50 → 17.0%, n=100 → 11.2%, n=200 → 7.7%, n=400 → 5.3%, n=800 → 3.8%. **Decision: n=200** for both the background and explain-sample (already what `scripts/train_hgb.py` uses) — comfortably under the ~15-20% sufficiency heuristic, with diminishing returns beyond it not worth the cost for a secondary diagnostic.
