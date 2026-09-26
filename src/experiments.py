"""Research experiments that vary one declared factor at a time."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.training import train_model
from src.utils import ensure_dir
from src.weak_labels import apply_label_rules, label_column_name


METRIC_COLUMNS = (
    "pr_auc", "pr_auc_lift", "roc_auc", "precision", "recall", "f1",
)


def _model_config(
    base_config: Mapping[str, Any],
    model_key: str,
    sensitivity_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one fixed experiment configuration without mutating YAML state."""
    config = copy.deepcopy(dict(base_config))
    config["training"]["hyperparameter_tuning"] = {"enabled": False}
    if model_key == "tfidf":
        config["model"].update(sensitivity_cfg.get("tfidf", {}))
        config["model"]["name"] = str(
            sensitivity_cfg.get("tfidf", {}).get("name", "tfidf_lr")
        )
    elif model_key == "albert":
        config["model"].update(sensitivity_cfg.get("albert", {}))
        config["model"]["name"] = "albert"
        config["training"]["loss"] = "weighted_cross_entropy"
        config["training"]["imbalance_method"] = "none"
    else:
        raise ValueError(f"Unsupported sensitivity model: {model_key}")
    return config


def _read_metrics(path: Path) -> dict[str, Any]:
    """Read the normalized test-metric payload written by training."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return dict(report.get("test_metrics", report.get("metrics", {})))


def run_label_sensitivity_experiment(
    sentence_df: pd.DataFrame,
    splits: Mapping[str, Sequence[int]],
    config: Mapping[str, Any],
    rules: Mapping[str, Mapping[str, Any]],
    report_dir: Path,
    model_root: Path,
    model_keys: Sequence[str] = ("tfidf", "albert"),
    rule_names: Sequence[str] = ("any_support", "consensus_2"),
    force: bool = False,
) -> pd.DataFrame:
    """Compare label rules while holding data partitions and models fixed.

    Args:
        sentence_df: Prepared sentence data.
        splits: Persisted row indices shared by every run.
        config: Full project configuration.
        rules: Available weak-label definitions.
        report_dir: Concentrated weak-label report directory.
        model_root: Root directory for sensitivity checkpoints.
        model_keys: ``tfidf`` and/or ``albert``.
        rule_names: Label rules to compare.
        force: Retrain runs even when a metrics file already exists.

    Returns:
        One row per label-rule/model combination.
    """
    missing_rules = sorted(set(rule_names) - set(rules))
    if missing_rules:
        raise ValueError(f"Unknown sensitivity label rules: {missing_rules}")
    labelled = apply_label_rules(sentence_df, rules)
    sensitivity_cfg = config.get("weak_labels", {}).get("sensitivity", {})
    seed = int(config["project"]["seed"])
    train_indices = np.asarray(splits["train"], dtype=int)
    test_indices = np.asarray(splits["test"], dtype=int)
    output_rows: list[dict[str, Any]] = []

    for rule_name in rule_names:
        rule_frame = labelled.copy()
        rule_frame["label"] = rule_frame[label_column_name(rule_name)].astype(int)
        train_positive = int(rule_frame.iloc[train_indices]["label"].sum())
        test_positive = int(rule_frame.iloc[test_indices]["label"].sum())
        for model_key in model_keys:
            run_report_dir = ensure_dir(report_dir / "runs" / rule_name / model_key)
            run_model_dir = ensure_dir(model_root / rule_name / model_key)
            metrics_path = run_report_dir / "test_metrics.json"
            if force or not metrics_path.exists():
                run_config = _model_config(config, model_key, sensitivity_cfg)
                train_model(
                    rule_frame,
                    {name: np.asarray(indices, dtype=int) for name, indices in splits.items()},
                    run_config,
                    run_model_dir,
                    run_report_dir,
                )
            metrics = _read_metrics(metrics_path)
            output_rows.append({
                "label_rule": rule_name,
                "model": model_key,
                "seed": seed,
                "samples": int(len(rule_frame)),
                "positive_samples": int(rule_frame["label"].sum()),
                "positive_rate": float(rule_frame["label"].mean()),
                "train_samples": int(len(train_indices)),
                "train_positive_samples": train_positive,
                "train_positive_rate": float(train_positive / len(train_indices)),
                "test_samples": int(len(test_indices)),
                "test_positive_samples": test_positive,
                "test_positive_rate": float(test_positive / len(test_indices)),
                **{column: metrics.get(column) for column in METRIC_COLUMNS},
                "threshold": metrics.get("threshold"),
                "positive_prediction_rate": metrics.get("positive_prediction_rate"),
                "metrics_path": str(metrics_path.resolve()),
                "predictions_path": str((run_report_dir / "test_predictions.csv").resolve()),
            })
    output = pd.DataFrame(output_rows)
    ensure_dir(report_dir)
    output.to_csv(report_dir / "model_by_label_rule.csv", index=False)
    return output
