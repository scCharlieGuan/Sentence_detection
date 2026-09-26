"""Tests for non-destructive label-rule construction."""
import pandas as pd

from src.weak_labels import DEFAULT_LABEL_RULES, apply_label_rules


def test_label_rules_preserve_original_labels() -> None:
    """Sensitivity labels must be new columns rather than destructive rewrites."""
    frame = pd.DataFrame({
        "label": [0, 1, 1],
        "positive_annotators": [0, 1, 2],
        "reliable_positive_annotators": [0, 0, 2],
        "total_annotators": [5, 5, 5],
    })
    labelled = apply_label_rules(frame, DEFAULT_LABEL_RULES)
    assert labelled["label"].tolist() == [0, 1, 1]
    assert labelled["label_original"].tolist() == [0, 1, 1]
    assert labelled["label_consensus_2"].tolist() == [0, 0, 1]
    assert labelled["label_consensus_2_reliable"].tolist() == [0, 0, 1]
