"""Evaluation metrics, threshold selection, and statistical comparison."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: Sequence[float],
    metric: str = "f1",
) -> Tuple[float, float]:
    """Choose a threshold using validation data only.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.
        grid: Candidate thresholds.
        metric: Threshold-dependent objective: ``f1``, ``f0.5``, or ``f2``.

    Returns:
        Best threshold and corresponding objective value.
    """
    beta_by_metric = {"f0.5": 0.5, "f1": 1.0, "f2": 2.0}
    if metric not in beta_by_metric:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    best_threshold, best_score = 0.5, -1.0
    for threshold in grid:
        score = fbeta_score(
            y_true,
            probabilities >= threshold,
            beta=beta_by_metric[metric],
            zero_division=0,
        )
        # A deterministic tie-break avoids arbitrary low thresholds.
        if score > best_score or (
            np.isclose(score, best_score) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute sentence-level classification metrics.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.
        threshold: Classification threshold.

    Returns:
        Metrics including Average Precision and prevalence-normalized AP lift.
        The historical ``pr_auc`` key stores sklearn Average Precision, not
        trapezoidal area under the precision-recall curve (Appendix B.5).
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    has_two_classes = len(np.unique(y_true)) > 1
    pr_auc = float(average_precision_score(y_true, probabilities)) if has_two_classes else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f0.5": float(fbeta_score(y_true, predictions, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2.0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_two_classes else float("nan"),
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_lift": pr_auc / prevalence if prevalence > 0 else float("nan"),
        "threshold": float(threshold),
        "positive_prediction_rate": float(predictions.mean()) if len(predictions) else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def build_prediction_frame(source: pd.DataFrame, indices: np.ndarray, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    """Attach predictions to selected sentence rows.

    Args:
        source: Full sentence dataset.
        indices: Selected row indices.
        probabilities: Positive-class probabilities.
        threshold: Classification threshold.

    Returns:
        Prediction DataFrame.
    """
    frame = source.iloc[indices].copy()
    frame["row_index"] = np.asarray(indices, dtype=int)
    frame["probability"] = probabilities
    frame["prediction"] = (probabilities >= threshold).astype(int)
    return frame


def aggregate_seed_metrics(runs: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Summarize independent random-seed runs.

    Args:
        runs: Per-seed metric mappings.

    Returns:
        Mean, sample standard deviation, minimum, maximum, and raw values.
    """
    if not runs:
        return {}
    keys = sorted(set.intersection(*(set(run) for run in runs)))
    summary: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    return summary


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Return percentile bootstrap CIs on one fixed test sample.

    This estimates test-sample uncertainty, not model-training or split
    uncertainty. Replicates containing one class are omitted for ranking metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities) or not len(y_true):
        raise ValueError("Labels and probabilities must have the same non-zero length.")
    metric_names = ("precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc")
    samples: Dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        sampled_probability = probabilities[indices]
        metrics = classification_metrics(sampled_y, sampled_probability, threshold)
        for name in metric_names:
            value = float(metrics[name])
            if np.isfinite(value):
                samples[name].append(value)
    alpha = 1.0 - float(confidence)
    output: Dict[str, Dict[str, float]] = {}
    observed = classification_metrics(y_true, probabilities, threshold)
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        output[name] = {
            "estimate": float(observed[name]),
            "ci_low": float(np.quantile(array, alpha / 2)),
            "ci_high": float(np.quantile(array, 1 - alpha / 2)),
            "valid_replicates": int(len(array)),
        }
    return output


def bootstrap_metric_difference(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    metric: str = "pr_auc",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a paired bootstrap interval for a model-metric difference.

    Resampling is paired because both models must predict the same test rows.
    PR-AUC and ROC-AUC replicates containing only one class are discarded.

    Args:
        y_true: Shared binary test labels.
        probabilities_a: Model A positive probabilities.
        probabilities_b: Model B positive probabilities.
        metric: ``pr_auc`` or ``roc_auc``.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Observed difference, confidence interval, and two-sided bootstrap p-value.
    """
    scorers: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
    }
    if metric not in scorers:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")
    y_true = np.asarray(y_true, dtype=int)
    probabilities_a = np.asarray(probabilities_a, dtype=float)
    probabilities_b = np.asarray(probabilities_b, dtype=float)
    if not (len(y_true) == len(probabilities_a) == len(probabilities_b)):
        raise ValueError("Paired predictions must have identical lengths.")

    scorer = scorers[metric]
    observed = float(scorer(y_true, probabilities_a) - scorer(y_true, probabilities_b))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        differences.append(
            scorer(sampled_y, probabilities_a[indices])
            - scorer(sampled_y, probabilities_b[indices])
        )
    values = np.asarray(differences, dtype=float)
    alpha = 1.0 - confidence
    p_value = 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0)))
    return {
        "difference_a_minus_b": observed,
        "ci_low": float(np.quantile(values, alpha / 2)),
        "ci_high": float(np.quantile(values, 1 - alpha / 2)),
        "p_value": min(1.0, p_value),
        "valid_replicates": int(len(values)),
    }


def mcnemar_exact(
    y_true: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
) -> Dict[str, float]:
    """Run the exact paired McNemar test on two hard classifiers.

    McNemar tests disagreement in correctness, not ranking quality. It is
    appropriate only for predictions on the exact same examples.
    """
    y_true = np.asarray(y_true, dtype=int)
    correct_a = np.asarray(predictions_a, dtype=int) == y_true
    correct_b = np.asarray(predictions_b, dtype=int) == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }
