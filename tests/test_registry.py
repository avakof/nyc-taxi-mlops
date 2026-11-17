"""assert_valid_predictions is the fix for the smoke check accepting infinity: pred >= 0 is
true for +inf too, so finiteness has to be checked explicitly, not implied by non-negativity."""

from __future__ import annotations

import pytest

from taxi_mlops.registry import assert_valid_predictions


def test_valid_predictions_pass() -> None:
    assert_valid_predictions([1.0, 2.0, 3.0], expected_length=3)


def test_infinite_prediction_rejected() -> None:
    with pytest.raises(SystemExit, match="non-finite"):
        assert_valid_predictions([1.0, float("inf"), 3.0], expected_length=3)


def test_nan_prediction_rejected() -> None:
    with pytest.raises(SystemExit, match="non-finite"):
        assert_valid_predictions([1.0, float("nan"), 3.0], expected_length=3)


def test_negative_prediction_rejected() -> None:
    with pytest.raises(SystemExit, match="negative"):
        assert_valid_predictions([1.0, -2.0, 3.0], expected_length=3)


def test_wrong_length_rejected() -> None:
    with pytest.raises(SystemExit, match="expected 3"):
        assert_valid_predictions([1.0, 2.0], expected_length=3)
