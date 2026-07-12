"""Evaluation metrics and report generation."""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray, grid: Sequence[float]) -> Tuple[float, float]:
    """Choose the validation threshold with the highest F1.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.
        grid: Candidate thresholds.

    Returns:
        Best threshold and corresponding F1.
    """
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute sentence-level classification metrics.

    Args:
        y_true: Binary labels.
        probabilities: Positive-class probabilities.
        threshold: Classification threshold.

    Returns:
        Metrics including PR-AUC and prevalence-normalized PR-AUC lift.
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
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_two_classes else float("nan"),
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_lift": pr_auc / prevalence if prevalence > 0 else float("nan"),
        "threshold": float(threshold),
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
    frame = source.loc[indices].copy()
    frame["probability"] = probabilities
    frame["prediction"] = (probabilities >= threshold).astype(int)
    return frame
