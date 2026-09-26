"""Tests for protocol-critical evaluation helpers."""
import numpy as np

from src.evaluate import (
    bootstrap_metric_intervals,
    classification_metrics,
    find_best_threshold,
    mcnemar_exact,
)


def test_metrics_include_requested_imbalance_measures() -> None:
    """F-beta scores and prediction rate must use the frozen threshold."""
    metrics = classification_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.8, 0.7, 0.9]),
        0.5,
    )
    assert {"f0.5", "f1", "f2", "pr_auc", "positive_prediction_rate"} <= set(metrics)
    assert metrics["positive_prediction_rate"] == 0.75


def test_threshold_tie_break_is_deterministic() -> None:
    """Equal scores should select the candidate closest to 0.5."""
    threshold, _ = find_best_threshold(
        np.asarray([0, 1]),
        np.asarray([0.1, 0.9]),
        [0.2, 0.5, 0.8],
    )
    assert threshold == 0.5


def test_mcnemar_requires_paired_correctness() -> None:
    """Discordant cells are counted on the same examples."""
    result = mcnemar_exact(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 1, 0]),
        np.asarray([1, 0, 1, 1]),
    )
    assert result["discordant"] == 3


def test_bootstrap_intervals_are_reproducible_and_contain_estimate_fields() -> None:
    """Fixed seeds must reproduce fixed-test sampling uncertainty."""
    y = np.asarray([0, 0, 0, 1, 1, 1])
    probability = np.asarray([0.1, 0.2, 0.7, 0.4, 0.8, 0.9])
    first = bootstrap_metric_intervals(y, probability, 0.5, n_bootstrap=50, seed=7)
    second = bootstrap_metric_intervals(y, probability, 0.5, n_bootstrap=50, seed=7)
    assert first == second
    assert {"estimate", "ci_low", "ci_high", "valid_replicates"} <= set(first["f1"])
