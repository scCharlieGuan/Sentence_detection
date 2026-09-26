"""Independent application workflows behind the command-line interface."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.audit import audit_dataset, compare_prediction_files, export_error_audit
from src.baselines import run_baseline_suite
from src.data_pipeline import (
    load_prepared,
    prepare_data,
    processed_task_dir,
)
from src.utils import configure_logging, ensure_dir, save_json, set_seed
from src.weak_label_analysis import run_weak_label_analysis
from src.weak_labels import configured_label_rules


@dataclass(frozen=True)
class WorkflowContext:
    """Validated settings and output locations shared by workflow handlers."""

    args: argparse.Namespace
    app_config: AppConfig
    model_dir: Path
    report_dir: Path
    analysis_dir: Path


def _run_prepare(context: WorkflowContext) -> None:
    """Run the prepare workflow."""
    app_config = context.app_config
    prepare_data(app_config)


def _run_train(context: WorkflowContext) -> None:
    """Run the train workflow."""
    from src.training import train_model

    app_config = context.app_config
    values = context.app_config.values
    model_dir = context.model_dir
    report_dir = context.report_dir
    try:
        sentence_df, splits = load_prepared(app_config)
    except FileNotFoundError:
        sentence_df, splits = prepare_data(app_config)

    # train the model and save the threshold for inference
    _, threshold = train_model(sentence_df, splits, values, model_dir, report_dir)
    save_json({"threshold": threshold}, model_dir / "inference_config.json")
    return


def _run_evaluate(context: WorkflowContext) -> None:
    """Run the evaluate workflow."""
    report_dir = context.report_dir
    metrics_path = report_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("No saved evaluation report. Run --mode train first.")
    print(metrics_path.read_text(encoding="utf-8"))
    return


def _run_weak_labels(context: WorkflowContext) -> None:
    """Run the weak labels workflow."""
    args = context.args
    app_config = context.app_config
    values = context.app_config.values
    task_name = context.app_config.task_name
    sentence_df, splits = load_prepared(app_config)
    weak_dir = ensure_dir(
        app_config.resolve_path(values["output"]["report_dir"])
        / "weak_label" / task_name
    )
    alignment_path = processed_task_dir(app_config) / "alignment_log.csv"
    if not alignment_path.exists():
        raise FileNotFoundError("Alignment log not found. Run --mode prepare first.")
    alignment_df = pd.read_csv(alignment_path)
    rules = configured_label_rules(values)
    sensitivity = values.get("weak_labels", {}).get("sensitivity", {})
    if args.mode == "label-sensitivity":
        from src.experiments import run_label_sensitivity_experiment

        model_keys = (
            [item.strip() for item in args.sensitivity_models.split(",") if item.strip()]
            if args.sensitivity_models else list(sensitivity.get("models", ["tfidf", "albert"]))
        )
        rule_names = (
            [item.strip() for item in args.sensitivity_rules.split(",") if item.strip()]
            if args.sensitivity_rules else list(sensitivity.get("rule_names", ["any_support", "consensus_2"]))
        )
        model_root = (
            app_config.resolve_path(values["output"]["model_dir"])
            / task_name / "weak_label_sensitivity"
        )
        results = run_label_sensitivity_experiment(
            sentence_df, splits, values, rules, weak_dir, model_root,
            model_keys=model_keys, rule_names=rule_names, force=args.force,
        )
        print(results.to_string(index=False))

    named_paths = parse_prediction_paths(args.prediction)
    if not named_paths:
        candidates = {
            model: weak_dir / "runs" / "any_support" / model / "test_predictions.csv"
            for model in ("tfidf", "albert")
        }
        if args.mode != "label-sensitivity" and not any(path.exists() for path in candidates.values()):
            root = app_config.resolve_path(values["output"]["report_dir"])
            candidates = {
                "tfidf": root / "metrics" / task_name / "tfidf_lr" / "test_predictions.csv",
                "albert": root / "metrics" / task_name / "albert_weighted_ce" / "test_predictions.csv",
            }
        named_paths = {name: path for name, path in candidates.items() if path.exists()}
    outputs = run_weak_label_analysis(
        sentence_df, splits, alignment_df, rules, weak_dir, named_paths
    )
    print("Weak-label outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return


def _run_diagnose(context: WorkflowContext) -> None:
    """Run the diagnose workflow."""
    app_config = context.app_config
    values = context.app_config.values
    task_name = context.app_config.task_name
    analysis_dir = ensure_dir(context.analysis_dir)
    sentence_df, splits = load_prepared(app_config)
    processed_dir = (
        app_config.resolve_path(values["data"]["processed_dir"]) / task_name
    )
    alignment_path = processed_dir / "alignment_log.csv"
    alignment_df = pd.read_csv(alignment_path) if alignment_path.exists() else None
    report = audit_dataset(sentence_df, splits, alignment_df)
    save_json(report, analysis_dir / "dataset_audit.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return


def _run_baselines(context: WorkflowContext) -> None:
    """Run the baselines workflow."""
    args = context.args
    app_config = context.app_config
    values = context.app_config.values
    analysis_dir = ensure_dir(context.analysis_dir)
    sentence_df, splits = load_prepared(app_config)
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


def _run_audit(context: WorkflowContext) -> None:
    """Run the audit workflow."""
    args = context.args
    app_config = context.app_config
    analysis_dir = ensure_dir(context.analysis_dir)
    sentence_df, splits = load_prepared(app_config)
    named_paths = parse_prediction_paths(args.prediction)
    output = export_error_audit(
        sentence_df,
        named_paths,
        analysis_dir / "manual_error_audit.csv",
    )
    print(f"Exported {len(output)} audit rows.")
    return


def _run_export(context: WorkflowContext) -> None:
    """Run the export workflow."""
    from src.predict import predict_dataset

    app_config = context.app_config
    values = context.app_config.values
    model_dir = context.model_dir
    report_dir = context.report_dir
    sentence_df, splits = load_prepared(app_config)

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


def _run_compare(context: WorkflowContext) -> None:
    """Run the compare workflow."""
    args = context.args
    seed = int(context.app_config.values["project"]["seed"])
    analysis_dir = ensure_dir(context.analysis_dir)
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


def _run_predict(context: WorkflowContext) -> None:
    """Run the predict workflow."""
    from src.predict import predict_response

    args = context.args
    app_config = context.app_config
    values = context.app_config.values
    task_name = context.app_config.task_name
    model_dir = context.model_dir
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


def parse_prediction_paths(arguments: list[str]) -> dict[str, Path]:
    """Parse unique named prediction files without silently overwriting a model."""
    paths: dict[str, Path] = {}
    for argument in arguments:
        name, separator, path = argument.partition("=")
        name, path = name.strip(), path.strip()
        if not separator or not name or not path:
            raise ValueError("--prediction must use a non-empty name=path.csv.")
        if name in paths:
            raise ValueError(f"Duplicate prediction name: {name}")
        paths[name] = Path(path).expanduser().resolve()
    return paths


HANDLERS = {
    "prepare": _run_prepare,
    "train": _run_train,
    "evaluate": _run_evaluate,
    "predict": _run_predict,
    "diagnose": _run_diagnose,
    "baselines": _run_baselines,
    "audit": _run_audit,
    "compare": _run_compare,
    "export": _run_export,
    "weak-label-audit": _run_weak_labels,
    "label-sensitivity": _run_weak_labels,
}


def run_workflow(args: argparse.Namespace, app_config: AppConfig) -> None:
    """Create shared runtime state and dispatch exactly one workflow."""
    values = app_config.values
    seed = int(values["project"]["seed"])
    set_seed(seed)
    task_name = app_config.task_name
    output_name = args.run_name or values["model"]["name"]
    report_root = app_config.resolve_path(values["output"]["report_dir"])
    report_dir = report_root / "metrics" / task_name / output_name
    model_dir = app_config.resolve_path(values["output"]["model_dir"]) / task_name / output_name
    if args.seed is not None:
        report_dir = report_dir / f"seed_{seed}"
        model_dir = model_dir / f"seed_{seed}"
    if args.mode in {"weak-label-audit", "label-sensitivity"}:
        log_path = report_root / "weak_label" / task_name / "workflow.log"
    elif args.mode in {"prepare", "diagnose", "baselines", "audit", "compare"}:
        log_path = report_root / "analysis" / task_name / f"{args.mode}.log"
    else:
        log_path = report_dir / "run.log"
    configure_logging(log_path, values["project"].get("log_level", "INFO"))
    context = WorkflowContext(
        args, app_config, model_dir, report_dir,
        report_root / "analysis" / task_name,
    )
    HANDLERS[args.mode](context)
