"""Tests for the scoring harness — ARCH §6.

The scoring math must be correct: a wrong Brier score or ECE would undermine the
headline claim. These tests prove the math against hand-computed examples.
"""

from __future__ import annotations

import pytest
from evals.scoring import (
    EvalResult,
    accuracy,
    brier_score,
    ece,
    score_claim,
    score_system,
)


def test_score_claim_correct() -> None:
    assert score_claim("true", "true", 0.9) is True
    assert score_claim("false", "false", 0.8) is True
    assert score_claim("mixed", "mixed", 0.5) is True


def test_score_claim_wrong() -> None:
    assert score_claim("true", "false", 0.9) is False
    assert score_claim("true", "mixed", 0.6) is False


def test_accuracy() -> None:
    assert accuracy([True, True, False, True]) == 0.75
    assert accuracy([]) == 0.0


def test_brier_perfect_prediction() -> None:
    # A perfect prediction (correct label, confidence 1.0) has Brier = 0.
    assert brier_score(["true"], ["true"], [1.0]) == pytest.approx(0.0, abs=1e-6)


def test_brier_completely_wrong() -> None:
    # Predicting "true" with confidence 1.0 when the truth is "false" is maximally wrong.
    bs = brier_score(["false"], ["true"], [1.0])
    assert bs > 1.0  # (1-0)^2 + (0-1)^2 + (0-0)^2 = 2


def test_brier_uncertain() -> None:
    # A 50/50 prediction (confidence 0.5) should have a moderate Brier score.
    bs = brier_score(["true"], ["true"], [0.5])
    # (0.5-1)^2 + (0.25-0)^2 + (0.25-0)^2 = 0.25 + 0.0625 + 0.0625 = 0.375
    assert bs == pytest.approx(0.375, abs=1e-6)


def test_brier_empty() -> None:
    assert brier_score([], [], []) == 0.0


def test_ece_perfectly_calibrated() -> None:
    # If every prediction is correct and confidence is 1.0, ECE = 0.
    assert ece([True, True, True], [1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_ece_overconfident() -> None:
    # All wrong but confident: high ECE.
    e = ece([False, False, False], [0.9, 0.9, 0.9])
    assert e > 0.8  # gap between 0% accuracy and 90% confidence


def test_ece_well_calibrated() -> None:
    # 80% correct at 80% confidence → low ECE.
    e = ece(
        [True, True, True, True, False],
        [0.8, 0.8, 0.8, 0.8, 0.8],
    )
    assert e < 0.05  # close to perfect calibration


def test_ece_empty() -> None:
    assert ece([], []) == 0.0


def test_system_metrics_aggregates() -> None:
    results = [
        EvalResult("c1", "a", "true", "true", 0.9, True, 100, 1.0, "fast"),
        EvalResult("c2", "b", "false", "true", 0.8, False, 200, 2.0, "adversarial"),
        EvalResult("c3", "c", "true", "true", 0.95, True, 150, 1.5, "fast"),
    ]
    metrics = score_system(results, "test")
    assert metrics.name == "test"
    assert metrics.n_claims == 3
    assert metrics.correct == 2
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.avg_tokens == pytest.approx(150.0)
    assert metrics.avg_wall_clock == pytest.approx(1.5)
    assert metrics.brier > 0  # one wrong prediction
    assert metrics.ece_value >= 0
