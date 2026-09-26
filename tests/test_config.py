"""Tests for configuration loading."""
from pathlib import Path

import pytest

from src.config import load_config, validate_config


def test_default_config_loads() -> None:
    """The distributed configuration should pass validation."""
    config = load_config(Path(__file__).parents[1] / "configs" / "config.yaml")
    assert config.task_name == "toxicity"
    assert config.target_column == "toxicity_copy"


@pytest.mark.parametrize("fractions", [(-0.1, 0.5, 0.6), (float("nan"), 0.15, 0.15), (1, 0, 0)])
def test_invalid_split_fractions(fractions):
    config = load_config(Path(__file__).parents[1] / "configs/config.yaml")
    for key, value in zip(("train_size", "validation_size", "test_size"), fractions):
        config.values["data"][key] = value
    with pytest.raises(ValueError, match="finite number"):
        validate_config(config.values)


@pytest.mark.parametrize("grid", [[], [0.5, 0.1], [0.5, 0.5], [float("nan")], [1.1]])
def test_invalid_threshold_grid(grid):
    config = load_config(Path(__file__).parents[1] / "configs/config.yaml")
    config.values["training"]["threshold_grid"] = grid
    with pytest.raises(ValueError, match="threshold_grid"):
        validate_config(config.values)


@pytest.mark.parametrize("values", [[], "text", {"project": None}])
def test_malformed_configuration(values):
    with pytest.raises(ValueError):
        validate_config(values)
