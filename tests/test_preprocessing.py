"""Tests for text preprocessing and grouped splitting."""
import pandas as pd
import spacy

from src.preprocessing import (
    align_copy_to_sentences,
    build_sentence_dataset,
    normalize_text,
    split_sentences,
    split_train_val_test,
)
from src.splitting import NEAR_DUPLICATE_GROUP_COLUMN


def test_normalize_text_preserves_hyphen() -> None:
    """Hyphens must survive normalization for reliable span offsets."""
    assert normalize_text("evidence-based  care") == "evidence-based care"


def test_normalize_text_normalizes_non_breaking_space() -> None:
    """Copy/paste NBSP differences should not force fuzzy alignment."""
    assert normalize_text("evidence-based\u00a0care") == "evidence-based care"


def test_fuzzy_best_window_never_unions_more_than_configured_window() -> None:
    """A span must select one contiguous fuzzy candidate, not their union."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    response = "Alpha detail. Beta detail. Gamma detail. Delta detail."
    sentences = split_sentences(response, nlp)
    result = align_copy_to_sentences(
        response,
        "Alpha detail Beta detail Gamma detail Delta changed",
        sentences,
        nlp,
        ["all"],
        fuzzy_threshold=0.50,
        fallback_threshold=0.40,
        min_overlap_ratio=0.30,
        max_window=2,
        selection_policy="best_window",
    )
    assert result["method"] == "fuzzy_best_window"
    assert 1 <= len(result["matched_indices"]) <= 2
    assert result["matched_indices"] == list(
        range(result["matched_indices"][0], result["matched_indices"][0] + len(result["matched_indices"]))
    )


def test_fuzzy_fallback_uses_the_best_window_score() -> None:
    """The fallback must return the candidate that actually earned its score."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    response = "Alpha concept. Beta concept. Unrelated ending."
    sentences = split_sentences(response, nlp)
    result = align_copy_to_sentences(
        response,
        "Alpha concept and Beta concept together",
        sentences,
        nlp,
        ["all"],
        fuzzy_threshold=1.01,
        fallback_threshold=0.50,
        min_overlap_ratio=0.30,
        max_window=2,
        selection_policy="best_window",
    )
    assert result["method"] == "fuzzy_best_window_fallback"
    assert result["matched_indices"] == [0, 1]


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


def test_agreement_is_scoped_to_question_response_instance() -> None:
    """Reused response text must not pool votes across different questions."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    frame = pd.DataFrame({
        "questionID": ["q1", "q1", "q2", "q2"],
        "responder": ["model", "model", "model", "model"],
        "survey_id": [1, 2, 1, 2],
        "response": ["Shared response."] * 4,
        "toxicity_copy": ["Shared response.", "N/A", "N/A", "N/A"],
    })
    task = {"name": "toxicity", "text_column": "response", "target_column": "toxicity_copy"}
    data = {
        "annotation_group_columns": ["questionID", "responder"],
        "entire_response_values": ["all"],
        "fuzzy_threshold": 0.78,
        "fuzzy_fallback_threshold": 0.68,
        "min_overlap_ratio": 0.30,
        "max_fuzzy_window": 3,
        "min_positive_annotators": 1,
    }
    sentences, alignment = build_sentence_dataset(frame, task, data, nlp)
    assert sentences["response_id"].nunique() == 2
    assert sentences["total_annotators"].tolist() == [2, 2]
    assert sentences["positive_annotators"].tolist() == [1, 0]
    assert {"span_length", "overlap_ratio", "matched_sentence", "crosses_sentence"} <= set(alignment)


def test_near_duplicate_split_keeps_linked_responses_together() -> None:
    """Text-only duplicate components must be atomic across partitions."""
    frame = pd.DataFrame({
        "response_id": list(range(12)),
        "questionID": [f"q{index}" for index in range(12)],
        "sentence": [
            "A distinctive duplicated sentence about supportive counselling.",
            "A distinctive duplicated sentence about supportive counselling.",
            *[f"Unique sufficiently long sentence number {index} for testing groups."
              for index in range(2, 12)],
        ],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    data = {
        "split_strategy": "near_duplicate",
        "split_group_column": "response_id",
        "near_duplicate_base_group_column": "response_id",
        "near_duplicate_threshold": 0.92,
        "near_duplicate_min_characters": 40,
        "train_size": 0.5,
        "validation_size": 0.25,
        "test_size": 0.25,
    }
    splits = split_train_val_test(frame, data, seed=42)
    assert NEAR_DUPLICATE_GROUP_COLUMN in frame
    membership = {
        int(index): name for name, indices in splits.items() for index in indices
    }
    assert membership[0] == membership[1]
    component_sets = {
        name: set(frame.iloc[indices][NEAR_DUPLICATE_GROUP_COLUMN])
        for name, indices in splits.items()
    }
    assert component_sets["train"].isdisjoint(component_sets["val"])
    assert component_sets["train"].isdisjoint(component_sets["test"])
    assert component_sets["val"].isdisjoint(component_sets["test"])
