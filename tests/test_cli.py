"""CLI defaults, validation and dispatch boundaries."""
from pathlib import Path

import pytest

from src.cli import parse_args
from src.config import load_config, with_overrides
from src.workflows import WorkflowContext, _run_compare, parse_prediction_paths


def test_default_protocol_is_dissertation_protocol(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = parse_args(["--mode", "prepare"])
    config = load_config(args.config)
    assert config.values["data"]["split_strategy"] == "near_duplicate"
    assert config.values["data"]["near_duplicate_threshold"] == 0.90


@pytest.mark.parametrize("argv", [
    ["--mode", "predict"],
    ["--mode", "compare", "--prediction-a", "a.csv"],
    ["--mode", "baselines", "--label-threshold", "0"],
])
def test_invalid_mode_arguments_fail_early(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_overrides_are_validated_and_do_not_mutate_source():
    config = load_config(parse_args(["--mode", "prepare"]).config)
    modified = with_overrides(config, {"seed": 43, "model_name": "sbert_lr"})
    assert modified.values["project"]["seed"] == 43
    assert config.values["project"]["seed"] == 42
    assert config.values["model"]["name"] == "tfidf_lr"
    with pytest.raises(ValueError, match="Unsupported model"):
        with_overrides(config, {"model_name": "invalid"})


@pytest.mark.parametrize("arguments", [["a.csv"], ["=a.csv"], ["a="], ["a=x.csv", " a=y.csv"]])
def test_prediction_paths_reject_ambiguous_names(arguments):
    with pytest.raises(ValueError):
        parse_prediction_paths(arguments)


def test_compare_does_not_require_prepared_dataset(monkeypatch, tmp_path):
    args = parse_args(["--mode", "compare", "--prediction-a", "a.csv", "--prediction-b", "b.csv"])
    config = load_config(args.config)
    monkeypatch.setattr("src.workflows.compare_prediction_files", lambda *args: {"difference": 0.1})
    monkeypatch.setattr("src.workflows.load_prepared", lambda *_: pytest.fail("Unexpected dataset load"))
    _run_compare(WorkflowContext(args, config, tmp_path, tmp_path, tmp_path))
    assert len(list(tmp_path.glob("model_comparison_*.json"))) == 1
