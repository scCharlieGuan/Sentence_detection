"""Command-line parsing; importing this module does not load ML backends."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, with_overrides

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/reliability_protocol.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sentence-level CounselBench error detector")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML configuration")
    parser.add_argument(
        "--mode",
        choices=(
            "prepare", "train", "evaluate", "predict", "diagnose", "baselines",
            "audit", "compare", "export", "weak-label-audit", "label-sensitivity",
        ),
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
    parser.add_argument(
        "--sensitivity-models",
        default=None,
        help="Comma-separated sensitivity models: tfidf,albert.",
    )
    parser.add_argument(
        "--sensitivity-rules",
        default=None,
        help="Comma-separated weak-label rules to compare.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain completed label-sensitivity runs.",
    )
    args = parser.parse_args(argv)
    if args.mode == "predict" and not (args.text or args.input_file):
        parser.error("predict requires --text or --input-file")
    if args.mode == "compare" and not (args.prediction_a and args.prediction_b):
        parser.error("compare requires --prediction-a and --prediction-b")
    if args.label_threshold is not None and args.label_threshold < 1:
        parser.error("--label-threshold must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    """Validate effective settings before loading and running a workflow."""
    args = parse_args(argv)
    config = with_overrides(load_config(args.config), vars(args))
    from src.workflows import run_workflow

    run_workflow(args, config)
