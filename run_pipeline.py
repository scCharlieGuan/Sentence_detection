"""Command-line entry point for preprocessing, training, evaluation, and prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data_loader import load_data
from src.preprocessing import build_sentence_dataset, load_spacy_model, split_train_val_test
from src.predict import predict_response
from src.train import train_model
from src.utils import configure_logging, ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sentence-level CounselBench error detector")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration")
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate", "predict"), required=True)
    parser.add_argument("--text", help="Response text for predict mode")
    parser.add_argument("--input-file", help="Text file containing a response for predict mode")
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
    pd.DataFrame({"row_index": sentence_df.index, "split": split_names}).to_csv(processed_dir / "splits.csv", index=False)
    return sentence_df, splits


def load_prepared(app_config) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load previously prepared sentence data and split assignments."""
    processed_dir = app_config.resolve_path(app_config.section("data")["processed_dir"]) / app_config.task_name
    sentence_path, split_path = processed_dir / "sentences.csv", processed_dir / "splits.csv"
    if not sentence_path.exists() or not split_path.exists():
        raise FileNotFoundError("Prepared data not found. Run --mode prepare or --mode train first.")
    sentence_df = pd.read_csv(sentence_path)
    split_df = pd.read_csv(split_path)
    splits = {name: split_df.loc[split_df["split"] == name, "row_index"].to_numpy(dtype=int) for name in ("train", "val", "test")}
    return sentence_df, splits


def main() -> None:
    """Execute the selected pipeline mode."""
    args = parse_args()
    app_config = load_config(args.config)
    values = app_config.values
    seed = int(values["project"]["seed"])
    set_seed(seed)
    task_name = app_config.task_name
    report_dir = app_config.resolve_path(values["output"]["report_dir"]) / "metrics" / task_name / values["model"]["name"]
    model_dir = app_config.resolve_path(values["output"]["model_dir"]) / task_name / values["model"]["name"]
    configure_logging(report_dir / "run.log", values["project"].get("log_level", "INFO"))

    if args.mode == "prepare":
        prepare_data(app_config)
        return
    if args.mode == "train":
        # Load prepared data if available, otherwise prepare it.
        processed_dir = ensure_dir(
            app_config.resolve_path(values["data"]["processed_dir"])
            / values["task"]["name"]
        )
        sentences_path = processed_dir / "sentences.csv"
        splits_path = processed_dir / "splits.csv"
        if sentences_path.is_file() and splits_path.is_file():
            sentence_df = pd.read_csv(sentences_path)
            # recover splits from the saved CSV file
            split_df = pd.read_csv(splits_path)
            splits = {
                split_name: group["row_index"].to_numpy()
                for split_name, group in split_df.groupby("split")
            }
        else:
            sentence_df, splits = prepare_data(app_config)
        
        _, threshold = train_model(sentence_df, splits, values, model_dir, report_dir)
        save_json({"threshold": threshold}, model_dir / "inference_config.json")
        return
    if args.mode == "evaluate":
        metrics_path = report_dir / "test_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError("No saved evaluation report. Run --mode train first.")
        print(metrics_path.read_text(encoding="utf-8"))
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
