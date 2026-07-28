"""Command-line entry point for preprocessing, training, evaluation, and prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.audit import audit_dataset, compare_prediction_files, export_error_audit, run_baseline_suite
from src.data_loader import load_data
from src.preprocessing import build_sentence_dataset, load_spacy_model, split_train_val_test
from src.predict import predict_response
from src.train import train_model
from src.utils import configure_logging, ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sentence-level CounselBench error detector")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration")
    parser.add_argument(
        "--mode",
        choices=("prepare", "train", "evaluate", "predict", "diagnose", "baselines", "audit", "compare", "export"),
        required=True,
    )
    parser.add_argument("--text", help="Response text for predict mode")
    parser.add_argument("--input-file", help="Text file containing a response for predict mode")
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated seeds for baseline repetitions.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Named prediction file for audit, for example tfidf=path.csv.",
    )
    parser.add_argument("--prediction-a", help="First paired prediction CSV for compare mode.")
    parser.add_argument("--prediction-b", help="Second paired prediction CSV for compare mode.")
    parser.add_argument("--model-name", help="Override model.name for export/evaluation workflows.")
    parser.add_argument("--seed", type=int, help="Override training seed; split assignments remain fixed.")
    parser.add_argument("--run-name", help="Output subdirectory name for an ablation run.")
    parser.add_argument("--embedding-model", help="Override model.sentence_transformer_name.")
    parser.add_argument(
        "--embedding-normalization",
        choices=("raw", "l2", "standardized"),
        help="Override fixed-embedding normalization ablation.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("sentence", "previous_target", "target_next", "window", "response", "question_target", "question_window"),
        help="Override the target/context input ablation.",
    )
    parser.add_argument(
        "--loss",
        choices=("cross_entropy", "weighted_cross_entropy", "focal"),
        help="Override transformer loss ablation.",
    )
    parser.add_argument(
        "--imbalance-method",
        choices=("none", "weighted_sampler"),
        help="Override transformer sampling ablation.",
    )
    parser.add_argument(
        "--freeze-strategy",
        choices=("classifier", "last_groups", "full"),
        help="Override transformer freezing ablation.",
    )
    parser.add_argument(
        "--label-threshold",
        type=int,
        help="For baseline sensitivity only, relabel using positive_annotators >= N.",
    )
    return parser.parse_args()


def prepare_data(app_config) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load source data, construct labels, and save deterministic split assignments."""
    values = app_config.values
    task, data = values["task"], values["data"]
    frame = load_data(app_config.resolve_path(data["raw_path"]), [task["text_column"], task["target_column"]])
    nlp = load_spacy_model(data["spacy_model"])
    sentence_df, alignment_df = build_sentence_dataset(frame, task, data, nlp)
    splits = split_train_val_test(sentence_df, data, int(values["project"]["seed"]))
    processed_dir = ensure_dir(app_config.resolve_path(data["processed_dir"]) / task["name"])
    sentence_df.to_csv(processed_dir / "sentences.csv", index=False)
    alignment_df.to_csv(processed_dir / "alignment_log.csv", index=False)
    split_names = np.full(len(sentence_df), "", dtype=object)
    for name, indices in splits.items():
        split_names[indices] = name
    split_frame = pd.DataFrame({
        "row_index": sentence_df.index,
        "response_id": sentence_df["response_id"],
        "label": sentence_df["label"],
        "split": split_names,
    })
    split_frame.to_csv(processed_dir / "splits.csv", index=False)
    fingerprint_columns = ["response_id", "sentence_id", "sentence", "label"]
    fingerprint = hashlib.sha256(
        sentence_df[fingerprint_columns].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    save_json({
        "task": task["name"],
        "seed": int(values["project"]["seed"]),
        "rows": int(len(sentence_df)),
        "responses": int(sentence_df["response_id"].nunique()),
        "split_group_column": data.get("split_group_column", "response_id"),
        "dataset_sha256": fingerprint,
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "test_is_model_selection_forbidden": True,
    }, processed_dir / "protocol_manifest.json")
    return sentence_df, splits


def load_prepared(app_config) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load previously prepared sentence data and split assignments."""
    processed_dir = app_config.resolve_path(app_config.section("data")["processed_dir"]) / app_config.task_name
    sentence_path, split_path = processed_dir / "sentences.csv", processed_dir / "splits.csv"
    if not sentence_path.exists() or not split_path.exists():
        raise FileNotFoundError("Prepared data not found. Run --mode prepare or --mode train first.")
    sentence_df = pd.read_csv(sentence_path)
    split_df = pd.read_csv(split_path)
    if len(split_df) != len(sentence_df) or set(split_df["row_index"]) != set(range(len(sentence_df))):
        raise ValueError("Prepared split assignments do not cover every sentence exactly once.")
    splits = {name: split_df.loc[split_df["split"] == name, "row_index"].to_numpy(dtype=int) for name in ("train", "val", "test")}
    return sentence_df, splits


def main() -> None:
    """Execute the selected pipeline mode."""
    args = parse_args()
    app_config = load_config(args.config)
    values = app_config.values
    if args.model_name:
        values["model"]["name"] = args.model_name
    if args.embedding_model:
        values["model"]["sentence_transformer_name"] = args.embedding_model
    if args.embedding_normalization:
        values["model"]["embedding_normalization"] = args.embedding_normalization
    if args.input_mode:
        values["model"]["input_mode"] = args.input_mode
    if args.loss:
        values["training"]["loss"] = args.loss
    if args.imbalance_method:
        values["training"]["imbalance_method"] = args.imbalance_method
    if args.freeze_strategy:
        values["training"]["freeze_strategy"] = args.freeze_strategy
    if args.seed is not None:
        values["project"]["seed"] = int(args.seed)
    seed = int(values["project"]["seed"])
    set_seed(seed)
    task_name = app_config.task_name
    output_name = args.run_name or values["model"]["name"]
    report_dir = app_config.resolve_path(values["output"]["report_dir"]) / "metrics" / task_name / output_name
    model_dir = app_config.resolve_path(values["output"]["model_dir"]) / task_name / output_name
    if args.seed is not None:
        report_dir = report_dir / f"seed_{seed}"
        model_dir = model_dir / f"seed_{seed}"
    configure_logging(report_dir / "run.log", values["project"].get("log_level", "INFO"))

    if args.mode == "prepare":
        prepare_data(app_config)
        return

    if args.mode == "train":
        # Load prepared data if available, otherwise prepare it.
        try:
            sentence_df, splits = load_prepared(app_config)
        except FileNotFoundError:
            sentence_df, splits = prepare_data(app_config)

        # train the model and save the threshold for inference
        _, threshold = train_model(sentence_df, splits, values, model_dir, report_dir)
        save_json({"threshold": threshold}, model_dir / "inference_config.json")
        return
    
    if args.mode == "evaluate":
        metrics_path = report_dir / "test_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("No saved evaluation report. Run --mode train first.")
        print(metrics_path.read_text(encoding="utf-8"))
        return

    if args.mode in {"diagnose", "baselines", "audit", "compare", "export"}:
        sentence_df, splits = load_prepared(app_config)
        analysis_dir = ensure_dir(
            app_config.resolve_path(values["output"]["report_dir"])
            / "analysis" / task_name
        )
        if args.mode == "diagnose":
            processed_dir = (
                app_config.resolve_path(values["data"]["processed_dir"]) / task_name
            )
            alignment_path = processed_dir / "alignment_log.csv"
            alignment_df = pd.read_csv(alignment_path) if alignment_path.exists() else None
            report = audit_dataset(sentence_df, splits, alignment_df)
            save_json(report, analysis_dir / "dataset_audit.json")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return
        if args.mode == "baselines":
            seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            baseline_frame = sentence_df.copy()
            baseline_output = analysis_dir / "baselines"
            if args.label_threshold is not None:
                if "positive_annotators" not in baseline_frame:
                    raise ValueError("Prepared data lacks positive_annotators.")
                baseline_frame["label"] = (
                    baseline_frame["positive_annotators"] >= args.label_threshold
                ).astype(int)
                baseline_output = analysis_dir / f"label_threshold_{args.label_threshold}"
            results = run_baseline_suite(
                baseline_frame,
                splits,
                baseline_output,
                seeds,
                values["training"]["threshold_grid"],
                str(values["model"].get("input_mode", "sentence")),
            )
            print(results.to_string(index=False))
            return
        if args.mode == "audit":
            named_paths = {}
            for value in args.prediction:
                if "=" not in value:
                    raise ValueError("--prediction must use name=path.csv.")
                name, path = value.split("=", 1)
                named_paths[name] = Path(path).resolve()
            output = export_error_audit(
                sentence_df,
                named_paths,
                analysis_dir / "manual_error_audit.csv",
            )
            print(f"Exported {len(output)} audit rows.")
            return
        if args.mode == "export":
            from src.predict import predict_dataset

            metrics_path = report_dir / "test_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"No saved metrics at {metrics_path}.")
            saved_report = json.loads(metrics_path.read_text(encoding="utf-8"))
            threshold = float(saved_report.get(
                "threshold",
                saved_report.get("test_metrics", saved_report.get("metrics", {})).get(
                    "threshold", 0.5
                ),
            ))
            predictions = predict_dataset(
                sentence_df, splits["test"], values, model_dir, threshold
            )
            predictions.to_csv(report_dir / "test_predictions.csv", index=False)
            print(f"Exported {len(predictions)} predictions to {report_dir}.")
            return
        if not args.prediction_a or not args.prediction_b:
            raise ValueError("Compare mode requires --prediction-a and --prediction-b.")
        comparison = compare_prediction_files(
            Path(args.prediction_a).resolve(),
            Path(args.prediction_b).resolve(),
            seed,
        )
        name_a = Path(args.prediction_a).resolve().parent.name
        name_b = Path(args.prediction_b).resolve().parent.name
        save_json(comparison, analysis_dir / f"model_comparison_{name_a}_vs_{name_b}.json")
        print(json.dumps(comparison, indent=2))
        return

    if not args.text and not args.input_file:
        raise ValueError("Predict mode requires --text or --input-file.")
    response = args.text or Path(args.input_file).read_text(encoding="utf-8")
    inference_path = model_dir / "inference_config.json"
    if not inference_path.exists():
        raise FileNotFoundError("Saved model metadata not found. Run --mode train first.")
    threshold = float(json.loads(inference_path.read_text(encoding="utf-8"))["threshold"])
    predictions = predict_response(response, values, model_dir, threshold)
    prediction_dir = ensure_dir(app_config.resolve_path(values["output"]["report_dir"]) / "predictions" / task_name)
    output_path = prediction_dir / f"{values['model']['name']}_prediction.csv"
    predictions.to_csv(output_path, index=False)
    print(predictions.to_string(index=False))
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
