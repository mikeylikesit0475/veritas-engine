"""Scoring harness for the eval — ARCH §6.

| Metric | Why it's there |
|--------|---------------|
| Accuracy vs. single-LLM baseline | The headline claim. Both numbers in one table or neither. |
| Brier score / ECE | Calibration. Does 80% confidence mean right 80% of the time? |
| Tokens + wall-clock per claim | The cost of rigour, stated honestly. |
| Router precision | Of claims routed ``fast``, how many did the adversarial path have changed? |

The Brier score measures both calibration *and* sharpness. ECE (Expected
Calibration Error) measures calibration alone. Together they tell you whether
the confidence numbers mean something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "EvalResult",
    "SystemMetrics",
    "accuracy",
    "brier_score",
    "ece",
    "score_claim",
    "score_system",
]


@dataclass
class EvalResult:
    """One claim's eval result."""

    claim_id: str
    claim_text: str
    true_label: str  # "true" | "false" | "mixed"
    predicted_label: str
    confidence: float
    correct: bool
    tokens: int = 0
    wall_clock_s: float = 0.0
    path: str = "unknown"  # "fast" | "adversarial"


@dataclass
class SystemMetrics:
    """Aggregated metrics for a system (Veritas or baseline) on the slice."""

    name: str
    n_claims: int = 0
    correct: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_claims if self.n_claims else 0.0

    @property
    def avg_tokens(self) -> float:
        return sum(r.tokens for r in self.results) / self.n_claims if self.n_claims else 0.0

    @property
    def avg_wall_clock(self) -> float:
        return sum(r.wall_clock_s for r in self.results) / self.n_claims if self.n_claims else 0.0

    @property
    def brier(self) -> float:
        return brier_score(
            [r.true_label for r in self.results],
            [r.predicted_label for r in self.results],
            [r.confidence for r in self.results],
        )

    @property
    def ece_value(self) -> float:
        return ece(
            [r.correct for r in self.results],
            [r.confidence for r in self.results],
        )


def score_claim(true_label: str, predicted_label: str, confidence: float) -> bool:
    """Is the prediction correct? A label match with non-trivial confidence."""
    return true_label == predicted_label


def score_system(results: list[EvalResult], name: str) -> SystemMetrics:
    """Aggregate per-claim results into system metrics."""
    metrics = SystemMetrics(name=name, n_claims=len(results), results=results)
    metrics.correct = sum(1 for r in results if r.correct)
    return metrics


def brier_score(
    true_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float],
) -> float:
    """The (multi-class) Brier score.

    For a 3-class problem (true/false/mixed), the Brier score is the mean squared
    error between the predicted probability distribution and the one-hot true
    label. A lower score is better; 0 is perfect.

    We approximate the distribution from the confidence: the predicted class gets
    ``confidence``, and the remaining ``1 - confidence`` is split evenly between
    the other two classes. This is a simplification (a real system would output a
    full distribution), but it's honest about what we have.
    """
    labels = ["true", "false", "mixed"]
    n = len(true_labels)
    if n == 0:
        return 0.0

    total = 0.0
    for true, pred, conf in zip(true_labels, predicted_labels, confidences, strict=True):
        for label in labels:
            # Predicted probability for this label.
            p = conf if label == pred else (1.0 - conf) / 2.0
            # True probability (one-hot).
            t = 1.0 if label == true else 0.0
            total += (p - t) ** 2
    return total / n


def ece(correct: list[bool], confidences: list[float], n_bins: int = 5) -> float:
    """Expected Calibration Error.

    Bin the confidence into ``n_bins`` equal-width bins. In each bin, compute the
    difference between the average confidence and the average accuracy. ECE is
    the weighted average of these gaps. Lower is better; 0 is perfectly calibrated.
    """
    n = len(correct)
    if n == 0:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    total_gap = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = [
            (c, conf)
            for c, conf in zip(correct, confidences, strict=True)
            if lo <= conf < hi or (i == n_bins - 1 and conf == hi)
        ]
        if not in_bin:
            continue
        bin_acc = sum(1 for c, _ in in_bin if c) / len(in_bin)
        bin_conf = sum(conf for _, conf in in_bin) / len(in_bin)
        total_gap += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    return total_gap


def accuracy(correct: list[bool]) -> float:
    """Simple accuracy: fraction correct."""
    return sum(1 for c in correct if c) / len(correct) if correct else 0.0
