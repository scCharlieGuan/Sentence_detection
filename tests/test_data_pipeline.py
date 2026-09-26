"""Prepared snapshots must retain row identity and their declared provenance."""
import json

import pandas as pd
import pytest

from src.config import AppConfig
from src.data_pipeline import _dataset_fingerprint, _existing_group_assignments, _file_fingerprint, load_prepared


@pytest.fixture
def snapshot(tmp_path):
    sentences = pd.DataFrame({
        "response_id": ["r1", "r2", "r3"],
        "sentence": ["First.", "Second.", "Third."],
        "label": [0, 1, 0], "label_original": [0, 1, 0],
    })
    assignments = pd.DataFrame({
        "row_index": [0, 1, 2], "response_id": ["r1", "r2", "r3"],
        "label": [0, 1, 0], "split": ["train", "val", "test"],
    })
    directory = tmp_path / "toxicity"
    directory.mkdir()
    sentences.to_csv(directory / "sentences.csv", index=False)
    assignments.to_csv(directory / "splits.csv", index=False)
    (directory / "protocol_manifest.json").write_text(json.dumps({
        "dataset_sha256": _dataset_fingerprint(sentences),
    }), encoding="utf-8")
    config = AppConfig(tmp_path, {"task": {"name": "toxicity"}, "data": {
        "processed_dir": str(tmp_path), "split_strategy": "group", "split_group_column": "response_id",
    }})
    return config, directory, sentences, assignments


def test_shuffled_assignment_rows_preserve_group_membership(snapshot):
    config, directory, _, assignments = snapshot
    assignments.iloc[::-1].to_csv(directory / "splits.csv", index=False)
    _, splits = load_prepared(config)
    assert splits["train"].tolist() == [0]
    assert _existing_group_assignments(directory, "response_id") == {"r1": "train", "r2": "val", "r3": "test"}


def test_changed_sentence_rejected(snapshot):
    config, directory, sentences, _ = snapshot
    sentences.loc[0, "sentence"] = "Changed."
    sentences.to_csv(directory / "sentences.csv", index=False)
    with pytest.raises(ValueError, match="fingerprint"):
        load_prepared(config)


@pytest.mark.parametrize("column,value", [("row_index", 1), ("response_id", "wrong"), ("label", 1), ("split", "unknown")])
def test_corrupt_split_table_rejected(snapshot, column, value):
    config, directory, _, assignments = snapshot
    assignments.loc[0, column] = value
    assignments.to_csv(directory / "splits.csv", index=False)
    with pytest.raises(ValueError):
        load_prepared(config)


def test_split_fingerprint_detects_reassignment(snapshot):
    config, directory, sentences, assignments = snapshot
    (directory / "protocol_manifest.json").write_text(json.dumps({
        "dataset_sha256": _dataset_fingerprint(sentences),
        "split_file_sha256": _file_fingerprint(directory / "splits.csv"),
    }), encoding="utf-8")
    assignments["split"] = ["val", "train", "test"]
    assignments.to_csv(directory / "splits.csv", index=False)
    with pytest.raises(ValueError, match="split file.*fingerprint"):
        load_prepared(config)


def test_atomic_group_cannot_cross_partitions(snapshot):
    config, directory, sentences, assignments = snapshot
    sentences.loc[1, "response_id"] = "r1"
    assignments.loc[1, "response_id"] = "r1"
    sentences.to_csv(directory / "sentences.csv", index=False)
    assignments.to_csv(directory / "splits.csv", index=False)
    (directory / "protocol_manifest.json").write_text(json.dumps({
        "dataset_sha256": _dataset_fingerprint(sentences),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="leak the atomic group"):
        load_prepared(config)
