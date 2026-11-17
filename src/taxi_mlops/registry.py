"""Shared verification for Registry smoke checks. `pred >= 0` alone is not a finiteness
check -- it's true for +inf too. Used by every registration script so this can't regress
in one without regressing in the other.
"""

from __future__ import annotations

import numpy as np


def assert_valid_predictions(pred, expected_length: int) -> None:
    pred = np.asarray(pred)
    if len(pred) != expected_length:
        raise SystemExit(
            f"smoke check failed: expected {expected_length} predictions, got {len(pred)}"
        )
    if not np.isfinite(pred).all():
        raise SystemExit(
            f"smoke check failed: non-finite prediction(s): {pred.tolist()}"
        )
    if not (pred >= 0).all():
        raise SystemExit(f"smoke check failed: negative prediction(s): {pred.tolist()}")
