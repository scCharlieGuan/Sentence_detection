"""Tests for text preprocessing and grouped splitting."""
import pandas as pd
import spacy

from src.preprocessing import build_sentence_dataset, normalize_text, split_train_val_test


def test_normalize_text_preserves_hyphen() -> None:
    """Hyphens must survive normalization for reliable span offsets."""
    assert normalize_text("evidence-based  care") == "evidence-based care"


def test_sentence_labels_and_group_safe_split() -> None:
    """Copied spans should label complete sentences and responses must not overlap."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    frame = pd.DataFrame({
        "response": [
            "Kind sentence. Harmful sentence.", "Kind sentence. Harmful sentence.",
            "Safe one. Safe two.", "Another safe response.",
            "Risk here. Safe ending.", "Neutral content only.",
        ],
        "toxicity_copy": ["Harmful sentence.", "N/A", "N/A", "N/A", "Risk here.", "N/A"],
    })
    task = {"name": "toxicity", "text_column": "response", "target_column": "toxicity_copy"}
    data = {
        "entire_response_values": ["all"], "fuzzy_threshold": 0.78,
        "fuzzy_fallback_threshold": 0.68, "min_overlap_ratio": 0.30,
        "max_fuzzy_window": 3, "min_positive_annotators": 1,
        "train_size": 0.5, "validation_size": 0.25, "test_size": 0.25,
    }
    sentences, _ = build_sentence_dataset(frame, task, data, nlp)
    assert sentences.loc[sentences["sentence"] == "Harmful sentence.", "label"].iloc[0] == 1
    splits = split_train_val_test(sentences, data, seed=42)
    groups = {name: set(sentences.loc[idx, "response_id"]) for name, idx in splits.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
