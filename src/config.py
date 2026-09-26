"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Validated project configuration.

    Attributes:
        root_dir: Project root directory.
        values: Parsed YAML configuration.
    """

    root_dir: Path
    values: Dict[str, Any]

    def section(self, name: str) -> Dict[str, Any]:
        """Return one configuration section.

        Args:
            name: Top-level section name.

        Returns:
            Mutable dictionary containing the section values.

        Raises:
            KeyError: If the section does not exist.
        """
        if name not in self.values:
            raise KeyError(f"Missing configuration section: {name}")
        return self.values[name]

    def resolve_path(self, value: str) -> Path:
        """Resolve a configured path relative to the project root.

        Args:
            value: Relative or absolute path string.

        Returns:
            Resolved path.
        """
        path = Path(value)
        return path if path.is_absolute() else (self.root_dir / path).resolve()

    @property
    def task_name(self) -> str:
        """Return the configured task name."""
        return str(self.section("task")["name"])

    @property
    def target_column(self) -> str:
        """Return the validated target span column."""
        return str(self.section("task")["target_column"])


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Validated application configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required values are missing or inconsistent.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}

    validate_config(values)
    return AppConfig(root_dir=path.parent.parent, values=values)


def validate_config(values: Dict[str, Any]) -> None:
    """Validate both YAML settings and the effective CLI-overridden settings."""
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    required = {"project", "task", "data", "model", "training", "output"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    for name in required | {"weak_labels"}:
        if name in values and not isinstance(values[name], dict):
            raise ValueError(f"Configuration section '{name}' must be a mapping.")

    task = values["task"]
    supported = task.get("supported_tasks", {})
    task_name = task.get("name")
    if task_name not in supported:
        raise ValueError(
            f"Unsupported task '{task_name}'. Supported tasks: {sorted(supported)}"
        )
    expected_target = supported[task_name]
    if task.get("target_column") != expected_target:
        raise ValueError(
            f"task.target_column must be '{expected_target}' for task '{task_name}'."
        )

    data = values["data"]
    fractions = []
    for key in ("train_size", "validation_size", "test_size"):
        try:
            fraction = float(data[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"data.{key} must be a number in (0, 1).") from exc
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError(f"data.{key} must be a finite number in (0, 1).")
        fractions.append(fraction)
    split_sum = sum(fractions)
    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size must equal 1.0")
    split_strategy = str(data.get("split_strategy", "group"))
    if split_strategy not in {"sentence", "group", "near_duplicate"}:
        raise ValueError("data.split_strategy must be sentence, group, or near_duplicate.")
    if split_strategy == "near_duplicate":
        threshold = float(data.get("near_duplicate_threshold", 0.92))
        if not 0.0 < threshold <= 1.0:
            raise ValueError("data.near_duplicate_threshold must be in (0, 1].")
        if int(data.get("near_duplicate_min_characters", 40)) < 1:
            raise ValueError("data.near_duplicate_min_characters must be positive.")
    fuzzy_policy = str(data.get("fuzzy_selection_policy", "best_window"))
    if fuzzy_policy not in {"best_window", "union"}:
        raise ValueError("data.fuzzy_selection_policy must be best_window or union.")

    model_name = values["model"].get("name")
    supported_models = {
        "tfidf_lr", "tfidf_word_lr", "tfidf_char_lr",
        "tfidf_word_char_lr", "tfidf_linear_svm", "sbert_lr",
        "sbert_linear_svm", "sbert_mlp", "sbert_lgbm",
        "sbert_extra_trees", "albert",
    }
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {sorted(supported_models)}")

    input_modes = {
        "sentence", "previous_target", "target_next", "window", "response",
        "question_target", "question_window",
    }
    input_mode = values["model"].get("input_mode", "sentence")
    if input_mode not in input_modes:
        raise ValueError(f"Unsupported model.input_mode '{input_mode}'.")

    training = values["training"]
    grid = training.get("threshold_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("training.threshold_grid must be a non-empty list.")
    try:
        thresholds = [float(value) for value in grid]
    except (TypeError, ValueError) as exc:
        raise ValueError("training.threshold_grid must contain numbers in [0, 1].") from exc
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("training.threshold_grid must contain finite numbers in [0, 1].")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("training.threshold_grid must be strictly increasing for deterministic ties.")
    if training.get("primary_selection_metric", "pr_auc") not in {
        "pr_auc", "roc_auc", "loss",
    }:
        raise ValueError("training.primary_selection_metric must be pr_auc, roc_auc, or loss.")
    if training.get("threshold_metric", "f1") not in {"f0.5", "f1", "f2"}:
        raise ValueError("training.threshold_metric must be f0.5, f1, or f2.")

    for rule_name, rule in values.get("weak_labels", {}).get("rules", {}).items():
        if int(rule.get("min_annotators", 0)) < 1:
            raise ValueError(
                f"weak_labels.rules.{rule_name}.min_annotators must be >= 1."
            )



def with_overrides(config: AppConfig, arguments: Dict[str, Any]) -> AppConfig:
    """Apply CLI overrides to a copy and revalidate before any output is written."""
    values = deepcopy(config.values)
    destinations = {
        "model_name": ("model", "name"),
        "embedding_model": ("model", "sentence_transformer_name"),
        "embedding_normalization": ("model", "embedding_normalization"),
        "input_mode": ("model", "input_mode"),
        "loss": ("training", "loss"),
        "imbalance_method": ("training", "imbalance_method"),
        "freeze_strategy": ("training", "freeze_strategy"),
        "seed": ("project", "seed"),
    }
    for argument, (section, key) in destinations.items():
        if arguments.get(argument) is not None:
            values[section][key] = arguments[argument]
    validate_config(values)
    return AppConfig(config.root_dir, values)
