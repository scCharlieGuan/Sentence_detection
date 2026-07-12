"""Tests for configuration loading."""
from pathlib import Path

from src.config import load_config


def test_default_config_loads() -> None:
    """The distributed configuration should pass validation."""
    config = load_config(Path(__file__).parents[1] / "configs" / "config.yaml")
    assert config.task_name == "toxicity"
    assert config.target_column == "toxicity_copy"
