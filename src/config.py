"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
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

    required = {"project", "task", "data", "model", "training", "output"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

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
    split_sum = sum(float(data[key]) for key in ("train_size", "validation_size", "test_size"))
    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size must equal 1.0")

    model_name = values["model"].get("name")
    supported_models = {
        "tfidf_lr", "tfidf_linear_svm", "sbert_lr", "sbert_lgbm",
        "sbert_extra_trees", "albert"
    }
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {sorted(supported_models)}")

    return AppConfig(root_dir=path.parent.parent, values=values)
