#!/usr/bin/env python
"""Run weak-label diagnostics for the task declared in a YAML config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_pipeline import load_prepared, processed_task_dir
from src.weak_label_analysis import run_weak_label_analysis
from src.weak_labels import configured_label_rules


def parse_args() -> argparse.Namespace:
    """Parse paths and optional named prediction files."""
    parser = argparse.ArgumentParser(description="Audit sentence weak-label quality.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "thesis_protocol.yaml"),
        help="Experiment YAML configuration.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Optional prediction file; may be repeated.",
    )
    return parser.parse_args()


def _prediction_paths(items: list[str]) -> dict[str, Path]:
    """Parse explicit predictions or discover the thesis TF-IDF/ALBERT runs."""
    if items:
        output: dict[str, Path] = {}
        for item in items:
            if "=" not in item:
                raise ValueError("--prediction must use NAME=CSV.")
            name, path = item.split("=", 1)
            output[name.strip()] = Path(path).resolve()
        return output
    return {}


def main() -> None:
    """Load prepared artifacts and write the complete diagnostic bundle."""
    args = parse_args()
    app_config = load_config(args.config)
    sentences, splits = load_prepared(app_config)
    alignment = pd.read_csv(processed_task_dir(app_config) / "alignment_log.csv")
    values = app_config.values
    report_root = app_config.resolve_path(values["output"]["report_dir"])
    output_dir = report_root / "weak_label" / app_config.task_name
    predictions = _prediction_paths(args.prediction)
    if not predictions:
        weak_dir = report_root / "weak_label" / app_config.task_name
        sensitivity_candidates = {
            "tfidf": weak_dir / "runs" / "any_support" / "tfidf" / "test_predictions.csv",
            "albert": weak_dir / "runs" / "any_support" / "albert" / "test_predictions.csv",
        }
        standard_candidates = {
            "tfidf": report_root / "metrics" / app_config.task_name / "tfidf_lr" / "test_predictions.csv",
            "albert": report_root / "metrics" / app_config.task_name / "albert_weighted_ce" / "test_predictions.csv",
        }
        candidates = (
            sensitivity_candidates
            if any(path.exists() for path in sensitivity_candidates.values())
            else standard_candidates
        )
        predictions = {name: path for name, path in candidates.items() if path.exists()}
    outputs = run_weak_label_analysis(
        sentences,
        splits,
        alignment,
        configured_label_rules(values),
        output_dir,
        predictions,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
