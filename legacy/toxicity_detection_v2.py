"""Sentence-level toxicity detection for CounselBench.

This script builds weak sentence-level labels from CounselBench toxicity spans,
splits the data into train/validation/test at response level, tunes each model on
validation data, and evaluates the final selected model configuration once on the
held-out test set.
"""

from __future__ import annotations

import ast
import itertools
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import spacy
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================


@dataclass
class Config:
    """Stores experiment configuration.

    Attributes:
        input_path: 
            Path to the CounselBench CSV file.
        output_dir: 
            Directory used for outputs and trained models.
        random_state: 
            Random seed used for reproducible data splits and models.
        train_size: 
            Fraction of responses used for training.
        val_size: 
            Fraction of responses used for validation and tuning.
        test_size: 
            Fraction of responses used for final held-out testing.
        spacy_model: 
            Preferred spaCy model for sentence segmentation.
        sbert_model: 
            SentenceTransformer model used for embedding-based models.
        sbert_batch_size: 
            Batch size for SBERT encoding.
        toxicity_copy_col: 
            Column containing expert-extracted toxicity spans.
        response_col: 
            Column containing the response text.
        response_id_cols: 
            Candidate columns used to construct response IDs.
        min_positive_annotators:
            Minimum number of matched span annotations needed
            to label a sentence as positive. Use 1 for high recall and 2 for a
            cleaner but smaller positive set.
        fuzzy_threshold: 
            Standard fuzzy threshold for sentence/window matching.
        fuzzy_fallback_threshold: 
            Lower threshold used only when a response has
            valid toxicity copies but no copy matches above `fuzzy_threshold`.
        min_overlap_ratio: 
            Minimum span overlap ratio for exact-match alignment.
        max_fuzzy_window: 
            Maximum number of consecutive sentences used for fuzzy
            window matching.
        threshold_grid: 
            Candidate probability thresholds tuned on validation data.
        primary_selection_metric: 
            Metric used to select hyperparameters.
        train_final_on_train_val: 
            Whether to refit the selected configuration on
            train + validation before test evaluation.
        save_models: 
            Whether to save final fitted models.
    """

    input_path: str = "./dataset/CounselBench.csv"
    output_dir: str = "./outputs/toxicity_sentence_classifier_v2"
    random_state: int = 42

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    spacy_model: str = "en_core_web_sm"
    sbert_model: str = "BAAI/bge-base-en-v1.5"
    sbert_batch_size: int = 32

    response_col: str = "response"
    toxicity_copy_col: str = "toxicity_copy"
    response_id_cols: Tuple[str, ...] = (
        "questionID",
        "questionTitle",
        # "answerProvider",
        "responder",
        # "model",
        "response",
    )
    entire_response_values: Tuple[str, ...] = (
        "all",
        "all of it",
        "entire response",
        "the entire response",
        "whole response",
        "the whole thing",
        "the whole response",
        )

    min_positive_annotators: int = 1
    fuzzy_threshold: float = 0.78
    fuzzy_fallback_threshold: float = 0.68
    min_overlap_ratio: float = 0.30
    max_fuzzy_window: int = 3

    threshold_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))
    primary_selection_metric: str = "pr_auc_lift"
    train_final_on_train_val: bool = True
    save_models: bool = True

    model_names: Tuple[str, ...] = (
        "tfidf_lr",
        "tfidf_linear_svm",
        "sbert_lr",
        "sbert_lgbm",
        "sbert_extra_trees",
    )


# ============================================================
# 2. Logging and utility functions
# ============================================================


def setup_logging(output_dir: str) -> None:
    """Configures file and console logging.

    Args:
        output_dir: Directory where `run.log` will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_spacy_model(model_name: str) -> spacy.language.Language:
    """Loads a spaCy English pipeline with sentence segmentation.

    Args:
        model_name: Name of the preferred spaCy model.

    Returns:
        A spaCy language pipeline. If the preferred model is unavailable, a blank
        English pipeline with `sentencizer` is returned.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    return nlp


def normalize_text(text: Any) -> str:
    """Applies lightweight normalization while preserving character offsets.

    The function intentionally does not delete punctuation or hyphens because
    sentence offsets and exact span matching depend on stable character positions.

    Args:
        text: Raw text value.

    Returns:
        Normalized text.
    """
    if pd.isna(text):
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_invalid_copy(text: Any) -> bool:
    """Checks whether an extracted toxicity copy is effectively empty.

    Args:
        text: Raw toxicity copy.

    Returns:
        True if the value is an empty/non-applicable annotation.
    """
    value = normalize_text(text).strip().lower()
    value = value.strip(" .,:;!?\"'`[](){}")

    invalid_values = {
        "",
        "n/a",
        "na",
        "none",
        "nan",
        "null",
        "no",
        "no toxicity",
        "not applicable",
        "nothing",
        "nil",
        "Nil",
    }
    return value in invalid_values


def normalize_for_match_spacy(text: str, nlp: spacy.language.Language) -> str:
    """Normalizes text for fuzzy matching using spaCy tokenization.

    Args:
        text: Raw text.
        nlp: spaCy pipeline used for tokenization.

    Returns:
        Token-normalized text suitable for fuzzy matching.
    """
    doc = nlp(normalize_text(text).lower())
    tokens = []

    for tok in doc:
        if tok.is_space:
            continue
        if tok.is_alpha or tok.is_digit or tok.like_num:
            tokens.append(tok.text)
            continue
        if "-" in tok.text and len(tok.text.strip("-")) > 0:
            tokens.append(tok.text)

    return " ".join(tokens).strip()


def spacy_sentence_split(text: str, nlp: spacy.language.Language) -> List[Tuple[str, int, int]]:
    """Splits a response into sentences with character offsets.

    Args:
        text: Response text.
        nlp: spaCy pipeline with sentence boundary detection.

    Returns:
        A list of `(sentence_text, start_char, end_char)` tuples.
    """
    normalized = normalize_text(text)
    doc = nlp(normalized)
    sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        # Adjust start and end offsets to exclude leading/trailing whitespace
        # This ensures the start and end match the stripped sentence text
        leading_spaces = len(sent.text) - len(sent.text.lstrip())
        trailing_spaces = len(sent.text) - len(sent.text.rstrip())
        start = sent.start_char + leading_spaces
        end = sent.end_char - trailing_spaces
        sentences.append((sent_text, start, end))

    return sentences


# ============================================================
# 3. Span alignment
# ============================================================


def char_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Computes overlap length between two half-open character spans.

    Args:
        a: First character span `(start, end)`.
        b: Second character span `(start, end)`.

    Returns:
        Number of overlapping characters.
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def find_all_exact_spans(response: str, copy: str) -> List[Tuple[int, int]]:
    """Finds all case-insensitive exact spans of a copy in a response.

    Args:
        response: Full response text.
        copy: Expert-extracted copy.

    Returns:
        A list of exact character spans in the normalized response.
    """
    response_norm = normalize_text(response)
    copy_norm = normalize_text(copy)

    if not copy_norm:
        return []

    response_lower = response_norm.lower()
    copy_lower = copy_norm.lower()

    spans = []
    start = 0
    while True:
        idx = response_lower.find(copy_lower, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(copy_norm)))
        start = idx + 1

    return spans


def token_set_score(a_norm: str, b_norm: str) -> float:
    """Computes a token-set containment/Jaccard score.

    Args:
        a_norm: Normalized text A.
        b_norm: Normalized text B.

    Returns:
        Token-level similarity score between 0 and 1.
    """
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    containment = intersection / min(len(a_tokens), len(b_tokens))
    jaccard = intersection / len(a_tokens | b_tokens)
    return float(max(containment * 0.85, jaccard))


def fuzzy_score(a: str, b: str, nlp: spacy.language.Language) -> float:
    """Computes robust fuzzy similarity between two text strings.

    Args:
        a: First text.
        b: Second text.
        nlp: spaCy pipeline used for matching normalization.

    Returns:
        Similarity score between 0 and 1.
    """
    a_norm = normalize_for_match_spacy(a, nlp)
    b_norm = normalize_for_match_spacy(b, nlp)

    if not a_norm or not b_norm:
        return 0.0

    char_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    token_ratio = token_set_score(a_norm, b_norm)

    if a_norm in b_norm or b_norm in a_norm:
        containment_ratio = min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm))
        return float(max(char_ratio, token_ratio, containment_ratio))

    return float(max(char_ratio, token_ratio))


def fuzzy_match_sentence_windows(
    toxicity_copy: str,
    sentences: Sequence[Tuple[str, int, int]],
    nlp: spacy.language.Language,
    threshold: float,
    max_window: int,
) -> Tuple[List[int], float, List[float]]:
    """Matches a toxicity copy to sentences and consecutive sentence windows.

    Args:
        toxicity_copy: Expert-extracted toxicity span.
        sentences: Sentence tuples `(text, start, end)`.
        nlp: spaCy pipeline.
        threshold: Minimum fuzzy score for a match.
        max_window: Maximum consecutive sentence window size.

    Returns:
        A tuple of `(matched_indices, best_score, single_sentence_scores)`.
    """
    matched: Set[int] = set()
    best_score = 0.0

    sentence_scores = [
        fuzzy_score(toxicity_copy, sent_text, nlp) for sent_text, _, _ in sentences
    ]

    for i, score in enumerate(sentence_scores):
        best_score = max(best_score, score)
        if score >= threshold:
            matched.add(i)

    n_sentences = len(sentences)
    for i in range(n_sentences):
        for window_size in range(2, max_window + 1):
            j = i + window_size
            if j > n_sentences:
                break
            window_text = " ".join(sent_text for sent_text, _, _ in sentences[i:j])
            score = fuzzy_score(toxicity_copy, window_text, nlp)
            best_score = max(best_score, score)
            if score >= threshold:
                matched.update(range(i, j))

    return sorted(matched), float(best_score), [float(s) for s in sentence_scores]


def align_copy_to_sentences(
    response: str,
    toxicity_copy: str,
    sentences: Sequence[Tuple[str, int, int]],
    nlp: spacy.language.Language,
    entire_response_values: Tuple[str, ...],
    fuzzy_threshold: float,
    min_overlap_ratio: float,
    max_fuzzy_window: int,
    fallback_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Aligns one toxicity copy to one or more sentences.

    Args:
        response: Full response text.
        toxicity_copy: Expert-extracted toxicity span.
        sentences: Sentence tuples `(text, start, end)`.
        nlp: spaCy pipeline.
        entire_response_values: Tuple of strings representing entire response indicators.
        fuzzy_threshold: Main fuzzy matching threshold.
        min_overlap_ratio: Minimum overlap ratio for exact span mapping.
        max_fuzzy_window: Maximum fuzzy window size.
        fallback_threshold: Optional lower threshold used to keep the best fuzzy
            match when no normal-threshold match is found.

    Returns:
        Alignment dictionary with matched indices, method, score, sentence
        scores, and exact spans.
    """
    toxicity_copy = normalize_text(toxicity_copy)

    if is_invalid_copy(toxicity_copy):
        return {
            "matched_indices": [],
            "method": "empty",
            "score": 0.0,
            "sentence_scores": [],
            "exact_spans": [],
        }

    if not sentences:
        return {
            "matched_indices": [],
            "method": "no_sentence",
            "score": 0.0,
            "sentence_scores": [],
            "exact_spans": [],
        }

    if toxicity_copy.strip().lower() in entire_response_values:
        return {
            "matched_indices": list(range(len(sentences))),
            "method": "all_response",
            "score": 1.0,
            "sentence_scores": [],
            "exact_spans": [(0, len(normalize_text(response)))],
        }

    exact_spans = find_all_exact_spans(response, toxicity_copy)
    if exact_spans:
        matched: Set[int] = set()
        for exact_span in exact_spans:
            copy_len = max(1, exact_span[1] - exact_span[0])
            for i, (_, sent_start, sent_end) in enumerate(sentences):
                sent_len = max(1, sent_end - sent_start)
                overlap = char_overlap((sent_start, sent_end), exact_span)
                if overlap / copy_len >= min_overlap_ratio or overlap / sent_len >= min_overlap_ratio:
                    matched.add(i)

        if matched:
            return {
                "matched_indices": sorted(matched),
                "method": "exact_char_overlap",
                "score": 1.0,
                "sentence_scores": [],
                "exact_spans": exact_spans,
            }

    matched_indices, best_score, sentence_scores = fuzzy_match_sentence_windows(
        toxicity_copy=toxicity_copy,
        sentences=sentences,
        nlp=nlp,
        threshold=fuzzy_threshold,
        max_window=max_fuzzy_window,
    )

    if matched_indices:
        return {
            "matched_indices": matched_indices,
            "method": "fuzzy_sentence_or_window",
            "score": best_score,
            "sentence_scores": sentence_scores,
            "exact_spans": exact_spans,
        }

    if fallback_threshold is not None and sentence_scores and best_score >= fallback_threshold:
        best_idx = int(np.argmax(sentence_scores))
        return {
            "matched_indices": [best_idx],
            "method": "fuzzy_best_fallback",
            "score": best_score,
            "sentence_scores": sentence_scores,
            "exact_spans": exact_spans,
        }

    return {
        "matched_indices": [],
        "method": "unmatched",
        "score": best_score,
        "sentence_scores": sentence_scores,
        "exact_spans": exact_spans,
    }


# ============================================================
# 4. Weak label generation
# ============================================================


def parse_copy_cell(value: Any) -> List[str]:
    """Parses one toxicity-copy cell into candidate text spans.

    Args:
        value: Raw cell value from the toxicity-copy column.

    Returns:
        A list of candidate toxicity copies from the cell.
    """
    if is_invalid_copy(value):
        return []

    text = normalize_text(value)
    candidates = [text]

    # if text looks like a list or dict, try to parse it
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                candidates.extend(str(x) for x in parsed)
            elif isinstance(parsed, dict):
                candidates.extend(str(x) for x in parsed.values())
        except (SyntaxError, ValueError):
            pass

    if "\n" in text:
        candidates.extend(part.strip(" -•*\t") for part in text.split("\n"))

    # clean duplicates and invalid entries
    cleaned = []
    seen = set()
    for candidate in candidates:
        candidate = normalize_text(candidate).strip(" \t\n\r-•*")
        if is_invalid_copy(candidate):
            continue
        key = candidate.lower()
        if key not in seen:
            cleaned.append(candidate)
            seen.add(key)

    return cleaned


def add_response_id(cb: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Adds a stable response ID used for grouped splitting.

    Args:
        cb: Raw CounselBench DataFrame.
        cfg: Experiment configuration.

    Returns:
        A DataFrame with `response_id` and normalized response text.
    """
    cb = cb.copy()
    cb[cfg.response_col] = cb[cfg.response_col].map(normalize_text)

    # available_cols = [col for col in cfg.response_id_cols if col in cb.columns]
    # if cfg.response_col not in available_cols:
    #     available_cols.append(cfg.response_col)

    # key_series = cb[available_cols].fillna("").astype(str).agg("||".join, axis=1)
    key_series = cb[cfg.response_col].fillna("").astype(str)
    cb["response_id"] = pd.factorize(key_series, sort=False)[0]
    return cb


def build_sentence_dataset(
    cb: pd.DataFrame,
    cfg: Config,
    nlp: spacy.language.Language,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constructs a sentence-level weak-label dataset.

    The label is based on aggregated span-to-sentence alignments. A sentence is
    positive when at least `cfg.min_positive_annotators` extracted toxicity spans
    align to it.

    Args:
        cb: CounselBench DataFrame with response and toxicity-copy columns.
        cfg: Experiment configuration.
        nlp: spaCy pipeline.

    Returns:
        A tuple `(sentence_df, alignment_df)`.
    """
    required_cols = [cfg.response_col, cfg.toxicity_copy_col]
    missing = [col for col in required_cols if col not in cb.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cb = add_response_id(cb, cfg)
    cb[cfg.toxicity_copy_col] = cb[cfg.toxicity_copy_col].map(normalize_text) 

    rows = []
    alignment_logs = []
    alignment_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

    grouped = cb.groupby(["response_id", cfg.response_col], sort=False)

    for (response_id, response), group in tqdm(grouped, desc="Building weak labels"):
        # split response into sentences with start position and end position
        sentences = spacy_sentence_split(response, nlp)

        match_counts = np.zeros(len(sentences), dtype=int)

        # generate toxicity copies in the group
        raw_copies = []
        for value in group[cfg.toxicity_copy_col].tolist():
            raw_copies.extend(parse_copy_cell(value))

        response_has_valid_copy = int(len(raw_copies) > 0)

        for copy in raw_copies:
            cache_key = (response_id, copy.lower())
            if cache_key not in alignment_cache:
                alignment_cache[cache_key] = align_copy_to_sentences(
                    response=response,
                    toxicity_copy=copy,
                    sentences=sentences,
                    nlp=nlp,
                    entire_response_values=cfg.entire_response_values,
                    fuzzy_threshold=cfg.fuzzy_threshold,
                    min_overlap_ratio=cfg.min_overlap_ratio,
                    max_fuzzy_window=cfg.max_fuzzy_window,
                    fallback_threshold=cfg.fuzzy_fallback_threshold,
                )

            align = alignment_cache[cache_key]
            for idx in align["matched_indices"]:
                if 0 <= idx < len(match_counts):
                    match_counts[idx] += 1

            alignment_logs.append(
                {
                    "response_id": response_id,
                    "toxicity_copy": copy,
                    "method": align["method"],
                    "score": align["score"],
                    "matched_indices": json.dumps(align["matched_indices"]),
                    "sentence_scores": json.dumps(align["sentence_scores"]),
                    "exact_spans": json.dumps(align["exact_spans"]),
                    "response": response,
                }
            )

        labels = (match_counts >= cfg.min_positive_annotators).astype(int)
        response_label = int(labels.max()) if len(labels) else 0

        for sent_id, (sent, start, end) in enumerate(sentences):
            rows.append(
                {
                    "response_id": response_id,
                    "sentence_id": sent_id,
                    "sentence": sent,
                    "start": start,
                    "end": end,
                    "label": int(labels[sent_id]),
                    "match_count": int(match_counts[sent_id]),
                    "response_label": response_label,
                    "response_has_valid_copy": response_has_valid_copy,
                    "response": response,
                }
            )

    # generate final dataframes
    sent_df = pd.DataFrame(rows)
    align_df = pd.DataFrame(alignment_logs)

    if sent_df.empty:
        raise ValueError("No sentences were generated. Check response text and spaCy segmentation.")

    return sent_df, align_df


# ============================================================
# 5. Splitting
# ============================================================


def safe_stratified_split(
    response_df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits response IDs with stratification when feasible.

    Args:
        response_df: DataFrame with `response_id` and `response_label`.
        test_size: Fraction assigned to the second split.
        random_state: Random seed.

    Returns:
        `(left_response_ids, right_response_ids)`.
    """
    ids = response_df["response_id"].values
    y = response_df["response_label"].values

    stratify = y if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() >= 2 else None

    left_ids, right_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return left_ids, right_ids


def split_train_val_test(sent_df: pd.DataFrame, cfg: Config) -> Dict[str, np.ndarray]:
    """Creates train/validation/test splits at response level.

    Args:
        sent_df: Sentence-level dataset.
        cfg: Experiment configuration.

    Returns:
        Dictionary with integer row indices for train, validation, and test.
    """
    total = cfg.train_size + cfg.val_size + cfg.test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    response_df = (
        sent_df.groupby("response_id", as_index=False)["label"]
        .max()
        .rename(columns={"label": "response_label"})
    )

    train_ids, temp_ids = safe_stratified_split(
        response_df=response_df,
        test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.random_state,
    )

    temp_df = response_df[response_df["response_id"].isin(temp_ids)].copy()
    val_fraction_of_temp = cfg.val_size / (cfg.val_size + cfg.test_size)
    test_size_in_temp = 1.0 - val_fraction_of_temp

    val_ids, test_ids = safe_stratified_split(
        response_df=temp_df,
        test_size=test_size_in_temp,
        random_state=cfg.random_state + 1,
    )

    splits = {
        "train": sent_df.index[sent_df["response_id"].isin(train_ids)].to_numpy(),
        "val": sent_df.index[sent_df["response_id"].isin(val_ids)].to_numpy(),
        "test": sent_df.index[sent_df["response_id"].isin(test_ids)].to_numpy(),
    }

    return splits


# ============================================================
# 6. Metrics
# ============================================================


def best_threshold_by_f1(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold_grid: Sequence[float],
) -> Tuple[float, float]:
    """Finds the threshold with the best validation F1.

    Args:
        y_true: Binary labels.
        prob: Positive-class probabilities.
        threshold_grid: Candidate thresholds.

    Returns:
        `(best_threshold, best_f1)`.
    """
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in threshold_grid:
        pred = (prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def sentence_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Computes sentence-level classification metrics.

    Args:
        y_true: Binary labels.
        prob: Positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        Metric dictionary including PR-AUC lift.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= threshold).astype(int)

    prevalence = float(np.mean(y_true)) if len(y_true) else np.nan
    pr_auc = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc_lift = pr_auc / prevalence if prevalence and prevalence > 0 else np.nan

    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_lift": pr_auc_lift,
        "threshold": float(threshold),
    }


def response_metrics(eval_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Computes response-level retrieval-style metrics.

    Args:
        eval_df: Evaluation DataFrame with `response_id`, `label`, and `prob`.
        threshold: Decision threshold.

    Returns:
        Response-level recall metrics.
    """
    results = []

    for response_id, group in eval_df.groupby("response_id"):
        y = group["label"].to_numpy().astype(int)
        p = group["prob"].to_numpy().astype(float)
        has_toxic = int(y.max()) if len(y) else 0
        order = np.argsort(-p)

        if has_toxic:
            found_any = int((((p >= threshold).astype(int)) * y).max())
            top1_hit = int(y[order[:1]].max())
            top3_hit = int(y[order[: min(3, len(order))]].max())
        else:
            found_any = np.nan
            top1_hit = np.nan
            top3_hit = np.nan

        results.append(
            {
                "response_id": response_id,
                "has_toxic": has_toxic,
                "found_any": found_any,
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
        )

    result_df = pd.DataFrame(results)
    toxic_only = result_df[result_df["has_toxic"] == 1]

    return {
        "response_found_any_recall": toxic_only["found_any"].mean() if len(toxic_only) else np.nan,
        "top1_recall": toxic_only["top1_hit"].mean() if len(toxic_only) else np.nan,
        "top3_recall": toxic_only["top3_hit"].mean() if len(toxic_only) else np.nan,
        "n_toxic_responses": float(len(toxic_only)),
    }


def evaluate_predictions(
    sent_df: pd.DataFrame,
    indices: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    model_name: str,
    split: str,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates predictions and returns metrics plus prediction rows.

    Args:
        sent_df: Sentence-level dataset.
        indices: Row indices being evaluated.
        prob: Positive-class probabilities for the rows.
        threshold: Decision threshold.
        model_name: Model name.
        split: Split name.
        params: Hyperparameters used by the model.

    Returns:
        `(metrics, prediction_df)`.
    """
    eval_df = sent_df.loc[indices].copy()
    eval_df["prob"] = prob
    eval_df["pred"] = (eval_df["prob"] >= threshold).astype(int)
    eval_df["model"] = model_name
    eval_df["split"] = split
    eval_df["params_json"] = json.dumps(params, sort_keys=True)

    metrics = {
        "model": model_name,
        "split": split,
        "params_json": json.dumps(params, sort_keys=True),
        **sentence_metrics(eval_df["label"].to_numpy(), prob, threshold),
        **response_metrics(eval_df, threshold),
    }
    return metrics, eval_df


# ============================================================
# 7. Models and hyperparameter grids
# ============================================================


def iter_param_grid(grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    """Yields dictionaries from a parameter grid.

    Args:
        grid: Mapping from parameter names to candidate values.

    Yields:
        Parameter dictionaries.
    """
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def make_tfidf_lr(params: Dict[str, Any], random_state: int) -> Pipeline:
    """Builds a TF-IDF + Logistic Regression pipeline.

    Args:
        params: Hyperparameters.
        random_state: Random seed.

    Returns:
        A scikit-learn Pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=params.get("ngram_range", (1, 2)),
                    min_df=params.get("min_df", 2),
                    max_df=params.get("max_df", 0.95),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=params.get("C", 1.0),
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_tfidf_linear_svm(params: Dict[str, Any], random_state: int) -> Pipeline:
    """Builds a calibrated TF-IDF + Linear SVM pipeline.

    Args:
        params: Hyperparameters.
        random_state: Random seed.

    Returns:
        A scikit-learn Pipeline with calibrated probabilities.
    """
    base = LinearSVC(
        C=params.get("C", 1.0),
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000,
    )
    calibrated = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=params.get("ngram_range", (1, 2)),
                    min_df=params.get("min_df", 2),
                    max_df=params.get("max_df", 0.95),
                    sublinear_tf=True,
                ),
            ),
            ("clf", calibrated),
        ]
    )


def make_sbert_lr(params: Dict[str, Any], random_state: int) -> LogisticRegression:
    """Builds Logistic Regression for SBERT embeddings.

    Args:
        params: Hyperparameters.
        random_state: Random seed.

    Returns:
        LogisticRegression classifier.
    """
    return LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )


def make_sbert_lgbm(params: Dict[str, Any], random_state: int) -> LGBMClassifier:
    """Builds a LightGBM classifier for SBERT embeddings.

    Args:
        params: Hyperparameters.
        random_state: Random seed.

    Returns:
        LGBMClassifier.
    """
    return LGBMClassifier(
        objective="binary",
        n_estimators=params.get("n_estimators", 200),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 15),
        min_child_samples=params.get("min_child_samples", 10),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1,
    )


def make_sbert_extra_trees(params: Dict[str, Any], random_state: int) -> ExtraTreesClassifier:
    """Builds an ExtraTrees classifier for SBERT embeddings.

    Args:
        params: Hyperparameters.
        random_state: Random seed.

    Returns:
        ExtraTreesClassifier.
    """
    return ExtraTreesClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def get_model_grids() -> Dict[str, List[Dict[str, Any]]]:
    """Defines compact hyperparameter grids for all models.

    Returns:
        Mapping from model name to list of parameter dictionaries.
    """
    grids = {
        "tfidf_lr": list(
            iter_param_grid(
                {
                    "ngram_range": [(1, 1), (1, 2)],
                    "min_df": [1, 2],
                    "C": [0.3, 1.0, 3.0],
                }
            )
        ),
        "tfidf_linear_svm": list(
            iter_param_grid(
                {
                    "ngram_range": [(1, 1), (1, 2)],
                    "min_df": [1, 2],
                    "C": [0.3, 1.0],
                }
            )
        ),
        "sbert_lr": list(iter_param_grid({"C": [0.1, 0.3, 1.0, 3.0]})),
        "sbert_lgbm": list(
            iter_param_grid(
                {
                    "n_estimators": [150, 300],
                    "learning_rate": [0.03, 0.05],
                    "num_leaves": [7, 15],
                    "min_child_samples": [5, 10],
                }
            )
        ),
        "sbert_extra_trees": list(
            iter_param_grid(
                {
                    "n_estimators": [300],
                    "max_depth": [None, 8],
                    "min_samples_leaf": [1, 2],
                }
            )
        ),
    }
    return grids


def fit_model(
    model_name: str,
    params: Dict[str, Any],
    train_idx: np.ndarray,
    y: np.ndarray,
    x_text: np.ndarray,
    x_emb: Optional[np.ndarray],
    random_state: int,
) -> Any:
    """Fits one model candidate.

    Args:
        model_name: Name of the model family.
        params: Hyperparameters.
        train_idx: Training row indices.
        y: Full label array.
        x_text: Full sentence text array.
        x_emb: Full SBERT embedding matrix, if needed.
        random_state: Random seed.

    Returns:
        Fitted model.
    """
    if model_name == "tfidf_lr":
        model = make_tfidf_lr(params, random_state)
        model.fit(x_text[train_idx], y[train_idx])
        return model

    if model_name == "tfidf_linear_svm":
        model = make_tfidf_linear_svm(params, random_state)
        model.fit(x_text[train_idx], y[train_idx])
        return model

    if x_emb is None:
        raise ValueError(f"Embeddings are required for model: {model_name}")

    if model_name == "sbert_lr":
        model = make_sbert_lr(params, random_state)
    elif model_name == "sbert_lgbm":
        model = make_sbert_lgbm(params, random_state)
    elif model_name == "sbert_extra_trees":
        model = make_sbert_extra_trees(params, random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(x_emb[train_idx], y[train_idx])
    return model


def predict_proba_model(
    model: Any,
    model_name: str,
    indices: np.ndarray,
    x_text: np.ndarray,
    x_emb: Optional[np.ndarray],
) -> np.ndarray:
    """Predicts positive-class probabilities for one fitted model.

    Args:
        model: Fitted model.
        model_name: Model family name.
        indices: Row indices to predict.
        x_text: Full sentence text array.
        x_emb: Full SBERT embedding matrix, if needed.

    Returns:
        Positive-class probability array.
    """
    if model_name.startswith("tfidf"):
        return model.predict_proba(x_text[indices])[:, 1]

    if x_emb is None:
        raise ValueError(f"Embeddings are required for model: {model_name}")

    return model.predict_proba(x_emb[indices])[:, 1]


# ============================================================
# 8. Training, tuning, and evaluation
# ============================================================


def save_error_analysis(
    eval_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    split: str,
    top_n: int = 40,
) -> None:
    """Saves false positives and false negatives for inspection.

    Args:
        eval_df: Prediction DataFrame with `label`, `pred`, and `prob`.
        output_dir: Output directory.
        model_name: Model name.
        split: Split name.
        top_n: Number of errors to save per error type.
    """
    fp = (
        eval_df[(eval_df["label"] == 0) & (eval_df["pred"] == 1)]
        .sort_values("prob", ascending=False)
        .head(top_n)
    )
    fn = (
        eval_df[(eval_df["label"] == 1) & (eval_df["pred"] == 0)]
        .sort_values("prob", ascending=True)
        .head(top_n)
    )

    fp.to_csv(os.path.join(output_dir, f"{model_name}_{split}_false_positives.csv"), index=False)
    fn.to_csv(os.path.join(output_dir, f"{model_name}_{split}_false_negatives.csv"), index=False)


def tune_one_model(
    model_name: str,
    param_grid: List[Dict[str, Any]],
    sent_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    x_text: np.ndarray,
    x_emb: Optional[np.ndarray],
    cfg: Config,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Tunes one model family on the validation split.

    Args:
        model_name: Model family name.
        param_grid: Hyperparameter candidates.
        sent_df: Sentence-level dataset.
        splits: Train/validation/test row indices.
        x_text: Full sentence text array.
        x_emb: Full SBERT embedding matrix.
        cfg: Experiment configuration.

    Returns:
        `(best_record, validation_metrics_df, validation_predictions_df)`.
    """
    y = sent_df["label"].to_numpy().astype(int)
    metric_rows = []
    pred_frames = []
    best_record: Optional[Dict[str, Any]] = None

    for candidate_id, params in enumerate(param_grid):
        logging.info("Tuning %s candidate %d/%d: %s", model_name, candidate_id + 1, len(param_grid), params)

        model = fit_model(
            model_name=model_name,
            params=params,
            train_idx=splits["train"],
            y=y,
            x_text=x_text,
            x_emb=x_emb,
            random_state=cfg.random_state,
        )

        val_prob = predict_proba_model(model, model_name, splits["val"], x_text, x_emb)
        threshold, _ = best_threshold_by_f1(
            y_true=y[splits["val"]],
            prob=val_prob,
            threshold_grid=cfg.threshold_grid,
        )
        metrics, pred_df = evaluate_predictions(
            sent_df=sent_df,
            indices=splits["val"],
            prob=val_prob,
            threshold=threshold,
            model_name=model_name,
            split="val",
            params=params,
        )
        metrics["candidate_id"] = candidate_id
        pred_df["candidate_id"] = candidate_id
        metric_rows.append(metrics)
        pred_frames.append(pred_df)

        selection_score = metrics.get(cfg.primary_selection_metric, np.nan)
        if best_record is None or np.nan_to_num(selection_score, nan=-np.inf) > np.nan_to_num(
            best_record["selection_score"], nan=-np.inf
        ):
            best_record = {
                "model": model_name,
                "candidate_id": candidate_id,
                "params": params,
                "threshold": threshold,
                "selection_score": selection_score,
                "val_metrics": metrics,
            }

    if best_record is None:
        raise RuntimeError(f"No candidate was evaluated for model: {model_name}")

    return best_record, pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def final_evaluate_one_model(
    best_record: Dict[str, Any],
    sent_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    x_text: np.ndarray,
    x_emb: Optional[np.ndarray],
    cfg: Config,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Refits the selected model and evaluates it on test data once.

    Args:
        best_record: Selected hyperparameters and threshold from validation.
        sent_df: Sentence-level dataset.
        splits: Train/validation/test row indices.
        x_text: Full sentence text array.
        x_emb: Full SBERT embedding matrix.
        cfg: Experiment configuration.

    Returns:
        `(final_model, test_metrics, test_predictions_df)`.
    """
    y = sent_df["label"].to_numpy().astype(int)
    model_name = best_record["model"]
    params = best_record["params"]
    threshold = best_record["threshold"]

    if cfg.train_final_on_train_val:
        train_idx = np.concatenate([splits["train"], splits["val"]])
    else:
        train_idx = splits["train"]

    final_model = fit_model(
        model_name=model_name,
        params=params,
        train_idx=train_idx,
        y=y,
        x_text=x_text,
        x_emb=x_emb,
        random_state=cfg.random_state,
    )

    test_prob = predict_proba_model(final_model, model_name, splits["test"], x_text, x_emb)
    test_metrics, test_pred_df = evaluate_predictions(
        sent_df=sent_df,
        indices=splits["test"],
        prob=test_prob,
        threshold=threshold,
        model_name=model_name,
        split="test",
        params=params,
    )
    test_metrics["candidate_id"] = best_record["candidate_id"]
    return final_model, test_metrics, test_pred_df


def run_experiment(cfg: Config) -> None:
    """Runs the full experiment pipeline.

    Args:
        cfg: Experiment configuration.
    """
    setup_logging(cfg.output_dir)
    logging.info("Config:\n%s", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    cb = pd.read_csv(cfg.input_path)
    nlp = load_spacy_model(cfg.spacy_model)

    sent_df, align_df = build_sentence_dataset(cb, cfg, nlp)
    sent_df.to_csv(os.path.join(cfg.output_dir, "sentence_weak_labels.csv"), index=False)
    align_df.to_csv(os.path.join(cfg.output_dir, "alignment_log.csv"), index=False)

    logging.info("Sentence dataset size: %d", len(sent_df))
    logging.info("Positive sentence rate: %.4f", sent_df["label"].mean())
    logging.info("Positive response rate: %.4f", sent_df.groupby("response_id")["label"].max().mean())
    if len(align_df):
        logging.info("Alignment methods:\n%s", align_df["method"].value_counts())

    splits = split_train_val_test(sent_df, cfg)
    split_summary = []
    for split_name, idx in splits.items():
        split_summary.append(
            {
                "split": split_name,
                "n_sentences": len(idx),
                "n_responses": sent_df.loc[idx, "response_id"].nunique(),
                "positive_sentence_rate": sent_df.loc[idx, "label"].mean(),
                "positive_response_rate": sent_df.loc[idx].groupby("response_id")["label"].max().mean(),
            }
        )
    split_summary_df = pd.DataFrame(split_summary)
    split_summary_df.to_csv(os.path.join(cfg.output_dir, "split_summary.csv"), index=False)
    logging.info("Split summary:\n%s", split_summary_df)

    x_text = sent_df["sentence"].to_numpy()
    x_emb = None

    if any(name.startswith("sbert") for name in cfg.model_names):
        logging.info("Encoding sentences with SBERT: %s", cfg.sbert_model)
        encoder = SentenceTransformer(cfg.sbert_model)
        x_emb = encoder.encode(
            x_text.tolist(),
            batch_size=cfg.sbert_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        with open(os.path.join(cfg.output_dir, "sbert_model.txt"), "w", encoding="utf-8") as f:
            f.write(cfg.sbert_model)

    grids = get_model_grids()
    val_metric_frames = []
    val_pred_frames = []
    test_metric_rows = []
    test_pred_frames = []
    best_records = []

    for model_name in cfg.model_names:
        if model_name not in grids:
            raise ValueError(f"No grid defined for model: {model_name}")

        best_record, val_metrics, val_preds = tune_one_model(
            model_name=model_name,
            param_grid=grids[model_name],
            sent_df=sent_df,
            splits=splits,
            x_text=x_text,
            x_emb=x_emb,
            cfg=cfg,
        )

        final_model, test_metrics, test_preds = final_evaluate_one_model(
            best_record=best_record,
            sent_df=sent_df,
            splits=splits,
            x_text=x_text,
            x_emb=x_emb,
            cfg=cfg,
        )

        best_records.append(
            {
                "model": model_name,
                "candidate_id": best_record["candidate_id"],
                "params_json": json.dumps(best_record["params"], sort_keys=True),
                "threshold": best_record["threshold"],
                "selection_metric": cfg.primary_selection_metric,
                "selection_score": best_record["selection_score"],
            }
        )
        val_metric_frames.append(val_metrics)
        val_pred_frames.append(val_preds)
        test_metric_rows.append(test_metrics)
        test_pred_frames.append(test_preds)

        save_error_analysis(test_preds, cfg.output_dir, model_name, "test")

        if cfg.save_models:
            joblib.dump(final_model, os.path.join(cfg.output_dir, f"{model_name}_final.joblib"))

    val_metrics_df = pd.concat(val_metric_frames, ignore_index=True)
    val_preds_df = pd.concat(val_pred_frames, ignore_index=True)
    test_metrics_df = pd.DataFrame(test_metric_rows)
    test_preds_df = pd.concat(test_pred_frames, ignore_index=True)
    best_records_df = pd.DataFrame(best_records)

    val_metrics_df.to_csv(os.path.join(cfg.output_dir, "validation_metrics_all_candidates.csv"), index=False)
    val_preds_df.to_csv(os.path.join(cfg.output_dir, "validation_predictions_all_candidates.csv"), index=False)
    test_metrics_df.to_csv(os.path.join(cfg.output_dir, "test_metrics_selected_models.csv"), index=False)
    test_preds_df.to_csv(os.path.join(cfg.output_dir, "test_predictions_selected_models.csv"), index=False)
    best_records_df.to_csv(os.path.join(cfg.output_dir, "best_model_configurations.csv"), index=False)

    logging.info("Best configurations:\n%s", best_records_df)
    logging.info("Test metrics:\n%s", test_metrics_df)
    print("\nBest configurations")
    print(best_records_df)
    print("\nTest metrics")
    print(test_metrics_df.sort_values(cfg.primary_selection_metric, ascending=False))


# ============================================================
# 9. Inference
# ============================================================


def predict_toxic_sentences(
    response: str,
    model: Any,
    model_type: str,
    nlp: spacy.language.Language,
    threshold: float,
    sbert_model: Optional[SentenceTransformer] = None,
) -> pd.DataFrame:
    """Predicts toxic sentences in one response.

    Args:
        response: Raw response text.
        model: Fitted sentence-level classifier.
        model_type: Model type, such as `tfidf_lr` or `sbert_lr`.
        nlp: spaCy pipeline for sentence segmentation.
        threshold: Probability threshold.
        sbert_model: SentenceTransformer encoder required for SBERT models.

    Returns:
        DataFrame with sentence spans, probabilities, and predicted labels.
    """
    sentences = spacy_sentence_split(response, nlp)
    sent_texts = [sent for sent, _, _ in sentences]

    if not sent_texts:
        return pd.DataFrame(
            columns=["sentence_id", "sentence", "start", "end", "toxicity_probability", "predicted_label"]
        )

    if model_type.startswith("tfidf"):
        prob = model.predict_proba(sent_texts)[:, 1]
    elif model_type.startswith("sbert"):
        if sbert_model is None:
            raise ValueError("sbert_model is required for SBERT-based inference.")
        emb = sbert_model.encode(
            sent_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        prob = model.predict_proba(emb)[:, 1]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    rows = []
    for i, ((sent, start, end), p) in enumerate(zip(sentences, prob)):
        rows.append(
            {
                "sentence_id": i,
                "sentence": sent,
                "start": start,
                "end": end,
                "toxicity_probability": float(p),
                "predicted_label": int(p >= threshold),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment(Config())
