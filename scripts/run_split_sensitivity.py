"""Compare leakage-sensitive split strategies on one fixed label snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.audit import audit_dataset
from src.config import load_config
from src.data_pipeline import load_prepared
from src.evaluate import (
    bootstrap_metric_intervals,
    build_prediction_frame,
    classification_metrics,
    find_best_threshold,
)
from src.input_features import build_model_inputs
from src.splitting import split_train_val_test
from src.utils import ensure_dir, save_json


CONDITIONS = {
    "naive_sentence": {"split_strategy": "sentence"},
    "response_grouped": {"split_strategy": "group", "split_group_column": "response_id"},
    "question_grouped": {"split_strategy": "group", "split_group_column": "questionID"},
    "near_duplicate_question_grouped": {
        "split_strategy": "near_duplicate",
        "split_group_column": "questionID",
        "near_duplicate_base_group_column": "questionID",
        "near_duplicate_threshold": 0.90,
        "near_duplicate_min_characters": 30,
        "near_duplicate_assignment_attempts": 500,
    },
    "responder_grouped": {"split_strategy": "group", "split_group_column": "responder"},
    "topic_grouped": {"split_strategy": "group", "split_group_column": "topic"},
}


def _split_fingerprint(splits: dict[str, np.ndarray]) -> str:
    assignments = [
        f"{name}:{','.join(map(str, np.asarray(splits[name], dtype=int)))}"
        for name in ("train", "val", "test")
    ]
    return hashlib.sha256("|".join(assignments).encode("utf-8")).hexdigest()


def _fit_tfidf(
    sentence_df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    threshold_grid: list[float],
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, dict[str, float]]]:
    texts = np.asarray(build_model_inputs(sentence_df, "sentence"), dtype=object)
    labels = sentence_df["label"].to_numpy(dtype=int)
    train, val, test = (np.asarray(splits[name], dtype=int) for name in ("train", "val", "test"))
    candidates = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=c_value, max_iter=3000, solver="liblinear", random_state=seed
            )),
        ])
        model.fit(texts[train], labels[train])
        val_probability = model.predict_proba(texts[val])[:, 1]
        val_pr_auc = classification_metrics(labels[val], val_probability, 0.5)["pr_auc"]
        candidates.append((val_pr_auc, -abs(np.log10(c_value)), c_value, model, val_probability))
    _, _, best_c, best_model, val_probability = max(candidates, key=lambda item: item[:2])
    threshold, _ = find_best_threshold(labels[val], val_probability, threshold_grid, metric="f1")
    test_probability = best_model.predict_proba(texts[test])[:, 1]
    metrics = classification_metrics(labels[test], test_probability, threshold)
    predictions = build_prediction_frame(sentence_df, test, test_probability, threshold)
    intervals = bootstrap_metric_intervals(
        labels[test], test_probability, threshold, n_bootstrap=300, seed=seed
    )
    return {"best_c": float(best_c), **metrics}, predictions, intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reliability_protocol.yaml")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", default="results/reliability_study/split_sensitivity")
    args = parser.parse_args()
    app_config = load_config(args.config)
    sentence_df, _ = load_prepared(app_config)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output_dir = ensure_dir(app_config.resolve_path(args.output_dir))
    threshold_grid = list(app_config.values["training"]["threshold_grid"])
    rows: list[dict[str, object]] = []

    for condition_name, overrides in CONDITIONS.items():
        for seed in seeds:
            frame = sentence_df.drop(columns=["near_duplicate_group_id"], errors="ignore").copy()
            data_cfg = {**app_config.values["data"], **overrides}
            splits = split_train_val_test(frame, data_cfg, seed)
            run_dir = ensure_dir(output_dir / condition_name / f"seed_{seed}")
            assignments = np.full(len(frame), "", dtype=object)
            for split_name, indices in splits.items():
                assignments[np.asarray(indices, dtype=int)] = split_name
            pd.DataFrame({
                "row_index": np.arange(len(frame), dtype=int),
                "split": assignments,
            }).to_csv(run_dir / "splits.csv", index=False)
            metrics, predictions, intervals = _fit_tfidf(
                frame, splits, threshold_grid, seed
            )
            predictions.to_csv(run_dir / "test_predictions.csv", index=False)
            save_json(intervals, run_dir / "test_bootstrap_ci.json")
            if seed == seeds[0]:
                alignment_path = (
                    app_config.resolve_path(app_config.values["data"]["processed_dir"])
                    / app_config.task_name / "alignment_log.csv"
                )
                alignment = pd.read_csv(alignment_path) if alignment_path.exists() else None
                save_json(audit_dataset(frame, splits, alignment), run_dir / "split_diagnostics.json")
            test = frame.iloc[np.asarray(splits["test"], dtype=int)]
            rows.append({
                "split_strategy": condition_name,
                "split_seed": seed,
                "split_sha256": _split_fingerprint(splits),
                "train_rows": int(len(splits["train"])),
                "validation_rows": int(len(splits["val"])),
                "test_rows": int(len(splits["test"])),
                "test_positive": int(test["label"].sum()),
                "test_prevalence": float(test["label"].mean()),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            })

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "split_sensitivity_results.csv", index=False)
    metric_columns = ["pr_auc", "roc_auc", "precision", "recall", "f1", "macro_f1"]
    summary = results.groupby("split_strategy")[metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output_dir / "split_sensitivity_summary.csv", index=False)
    save_json({
        "label_snapshot_sha256": app_config.values.get("_protocol_manifest", {}).get("dataset_sha256"),
        "seeds_are_split_seeds": True,
        "model": "word (1,2)-gram TF-IDF + LogisticRegression",
        "model_selection": "C selected on validation PR-AUC; threshold selected on validation F1",
        "duplicate_clustering_uses_labels": False,
        "conditions": CONDITIONS,
    }, output_dir / "protocol.json")
    print(summary.to_string())


if __name__ == "__main__":
    main()
