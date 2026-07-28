"""Text normalization, span alignment, weak labels, and grouped splitting."""
from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SentenceSpan = Tuple[str, int, int]


def load_spacy_model(model_name: str) -> spacy.language.Language:
    """Load an English spaCy pipeline with sentence segmentation.

    Args:
        model_name: Preferred installed spaCy model.

    Returns:
        Loaded pipeline, or a blank English sentencizer fallback.
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
    """Normalize whitespace without changing punctuation semantics.

    Args:
        text: Raw value.

    Returns:
        Normalized string.
    """
    if pd.isna(text):
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def is_invalid_copy(text: Any) -> bool:
    """Return whether a span annotation is empty or non-applicable.

    Args:
        text: Raw span value.

    Returns:
        True for empty and standard non-applicable markers.
    """
    value = normalize_text(text).lower().strip(" .,:;!?\"'`[](){}")
    return value in {"", "n/a", "na", "none", "nan", "null", "no", "not applicable", "nothing", "nil", "no toxicity"}


def normalize_for_match(text: str, nlp: spacy.language.Language) -> str:
    """Create a token-normalized representation for fuzzy matching.

    Args:
        text: Raw text.
        nlp: spaCy tokenizer.

    Returns:
        Lowercase token string that preserves meaningful hyphenated tokens.
    """
    tokens: List[str] = []
    for token in nlp(normalize_text(text).lower()):
        if token.is_space:
            continue
        if token.is_alpha or token.is_digit or token.like_num:
            tokens.append(token.text)
        elif "-" in token.text and token.text.strip("-"):
            tokens.append(token.text)
    return " ".join(tokens)


def split_sentences(text: str, nlp: spacy.language.Language) -> List[SentenceSpan]:
    """Split text into sentences while retaining character offsets.

    Args:
        text: Response text.
        nlp: spaCy pipeline.

    Returns:
        Tuples of sentence text, start offset, and end offset.
    """
    normalized = normalize_text(text)
    sentences: List[SentenceSpan] = []
    for sent in nlp(normalized).sents:
        cleaned = sent.text.strip()
        if not cleaned:
            continue
        leading = len(sent.text) - len(sent.text.lstrip())
        trailing = len(sent.text) - len(sent.text.rstrip())
        sentences.append((cleaned, sent.start_char + leading, sent.end_char - trailing))
    return sentences


def parse_copy_cell(value: Any) -> List[str]:
    """Parse one annotation cell into unique candidate spans.

    Args:
        value: Raw target-column value.

    Returns:
        Candidate copied spans.
    """
    if is_invalid_copy(value):
        return []
    text = normalize_text(value)
    candidates = [text]
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                candidates.extend(str(item) for item in parsed)
            elif isinstance(parsed, dict):
                candidates.extend(str(item) for item in parsed.values())
        except (SyntaxError, ValueError):
            pass
    if "\n" in text:
        candidates.extend(part.strip(" -•*\t") for part in text.splitlines())
    output, seen = [], set()
    for candidate in candidates:
        candidate = normalize_text(candidate).strip(" \t\n\r-•*")
        key = candidate.lower()
        if not is_invalid_copy(candidate) and key not in seen:
            output.append(candidate)
            seen.add(key)
    return output


def _fuzzy_score(a: str, b: str, nlp: spacy.language.Language) -> float:
    a_norm, b_norm = normalize_for_match(a, nlp), normalize_for_match(b, nlp)
    if not a_norm or not b_norm:
        return 0.0
    char_score = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_set, b_set = set(a_norm.split()), set(b_norm.split())
    intersection = len(a_set & b_set)
    token_score = max(
        intersection / min(len(a_set), len(b_set)),
        intersection / len(a_set | b_set),
    )
    containment = min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm)) if a_norm in b_norm or b_norm in a_norm else 0.0
    return float(max(char_score, token_score, containment))


def align_copy_to_sentences(
    response: str,
    copy: str,
    sentences: Sequence[SentenceSpan],
    nlp: spacy.language.Language,
    entire_response_values: Sequence[str],
    fuzzy_threshold: float,
    fallback_threshold: Optional[float],
    min_overlap_ratio: float,
    max_window: int,
) -> Dict[str, Any]:
    """Align an expert-copied span to complete response sentences.

    Args:
        response: Full response.
        copy: Copied target span.
        sentences: Sentence spans from the response.
        nlp: spaCy pipeline.
        entire_response_values: Markers meaning the whole response.
        fuzzy_threshold: Main fuzzy threshold.
        fallback_threshold: Lower threshold for a single best fallback.
        min_overlap_ratio: Character-overlap threshold for exact spans.
        max_window: Largest consecutive sentence window considered.

    Returns:
        Alignment metadata and matched sentence indices.
    """
    if is_invalid_copy(copy) or not sentences:
        return {"matched_indices": [], "method": "empty", "score": 0.0}
    if normalize_text(copy).lower() in {value.lower() for value in entire_response_values}:
        return {"matched_indices": list(range(len(sentences))), "method": "all_response", "score": 1.0}

    response_lower, copy_lower = normalize_text(response).lower(), normalize_text(copy).lower()
    exact_start = response_lower.find(copy_lower)
    if exact_start >= 0:
        exact = (exact_start, exact_start + len(copy_lower))
        matched: Set[int] = set()
        for index, (_, start, end) in enumerate(sentences):
            overlap = max(0, min(end, exact[1]) - max(start, exact[0]))
            if overlap / max(1, exact[1] - exact[0]) >= min_overlap_ratio or overlap / max(1, end - start) >= min_overlap_ratio:
                matched.add(index)
        if matched:
            return {"matched_indices": sorted(matched), "method": "exact_char_overlap", "score": 1.0}

    scores = [_fuzzy_score(copy, sentence, nlp) for sentence, _, _ in sentences]
    matched = {index for index, score in enumerate(scores) if score >= fuzzy_threshold}
    best_score = max(scores, default=0.0)
    for start in range(len(sentences)):
        for size in range(2, max_window + 1):
            end = start + size
            if end > len(sentences):
                break
            score = _fuzzy_score(copy, " ".join(item[0] for item in sentences[start:end]), nlp)
            best_score = max(best_score, score)
            if score >= fuzzy_threshold:
                matched.update(range(start, end))
    if matched:
        return {"matched_indices": sorted(matched), "method": "fuzzy_sentence_or_window", "score": best_score}
    if fallback_threshold is not None and scores and best_score >= fallback_threshold:
        return {"matched_indices": [int(np.argmax(scores))], "method": "fuzzy_best_fallback", "score": best_score}
    return {"matched_indices": [], "method": "unmatched", "score": best_score}


def build_sentence_dataset(frame: pd.DataFrame, task: Dict[str, Any], data_cfg: Dict[str, Any], nlp: spacy.language.Language) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build sentence-level weak labels from repeated expert span annotations.

    Args:
        frame: CounselBench rows.
        task: Task configuration.
        data_cfg: Data and alignment configuration.
        nlp: spaCy pipeline.

    Returns:
        Sentence dataset and alignment audit log.
    """
    response_col, target_col = task["text_column"], task["target_column"]
    work = frame.copy()
    work[response_col] = work[response_col].map(normalize_text)
    # Identical response text receives one group ID, preventing cross-split duplication.
    work["response_id"] = pd.factorize(work[response_col], sort=False)[0]
    rows, logs = [], []
    grouped = work.groupby(["response_id", response_col], sort=False)
    for (response_id, response), group in tqdm(grouped, desc=f"Building {task['name']} labels"):
        sentences = split_sentences(response, nlp)
        # Agreement must count annotators, not copied fragments. Otherwise one
        # annotator who supplies several overlapping fragments is counted more
        # than once and can manufacture false consensus.
        sentence_annotators: List[Set[str]] = [set() for _ in sentences]
        for row_index, annotation in group.iterrows():
            annotator_id = str(annotation.get("survey_id", row_index))
            copies = parse_copy_cell(annotation[target_col])
            annotator_matches: Set[int] = set()
            for copy in copies:
                result = align_copy_to_sentences(
                    response, copy, sentences, nlp,
                    data_cfg["entire_response_values"],
                    float(data_cfg["fuzzy_threshold"]),
                    float(data_cfg["fuzzy_fallback_threshold"]),
                    float(data_cfg["min_overlap_ratio"]),
                    int(data_cfg["max_fuzzy_window"]),
                )
                annotator_matches.update(result["matched_indices"])
                logs.append({
                    "response_id": response_id,
                    "source_row_index": int(row_index),
                    "annotator_id": annotator_id,
                    "task": task["name"],
                    "target_column": target_col,
                    "copy": copy,
                    "method": result["method"],
                    "score": result["score"],
                    "matched_indices": json.dumps(result["matched_indices"]),
                })
            for index in annotator_matches:
                sentence_annotators[index].add(annotator_id)
        counts = np.asarray([len(values) for values in sentence_annotators], dtype=int)
        labels = (counts >= int(data_cfg["min_positive_annotators"])).astype(int)
        first = group.iloc[0]
        metadata_columns = (
            "questionID", "questionTitle", "questionText", "topic", "responder"
        )
        metadata = {
            column: normalize_text(first[column])
            for column in metadata_columns
            if column in group.columns
        }
        for sentence_id, (sentence, start, end) in enumerate(sentences):
            rows.append({
                "response_id": response_id, "sentence_id": sentence_id,
                "sentence": sentence, "start": start, "end": end,
                "label": int(labels[sentence_id]), "match_count": int(counts[sentence_id]),
                "positive_annotators": int(counts[sentence_id]),
                "total_annotators": int(group["survey_id"].nunique())
                if "survey_id" in group.columns else int(len(group)),
                "response": response,
                **metadata,
            })
    sentence_df = pd.DataFrame(rows)
    if sentence_df.empty:
        raise ValueError("No sentences were generated from the configured response column.")
    return sentence_df, pd.DataFrame(logs)


def split_train_val_test(sentence_df: pd.DataFrame, data_cfg: Dict[str, Any], seed: int) -> Dict[str, np.ndarray]:
    """Split response groups into train, validation, and test sets.

    Args:
        sentence_df: Sentence-level rows.
        data_cfg: Split fractions.
        seed: Random seed.

    Returns:
        Mapping from split name to sentence-row indices.
    """
    group_column = str(data_cfg.get("split_group_column", "response_id"))
    if group_column not in sentence_df:
        raise ValueError(f"Configured split_group_column is absent: {group_column}")
    response_df = sentence_df.groupby(group_column, as_index=False)["label"].max()
    ids, labels = response_df[group_column].to_numpy(), response_df["label"].to_numpy()
    stratify = labels if len(np.unique(labels)) > 1 and pd.Series(labels).value_counts().min() >= 2 else None
    # first split the response groups into a training set and a temporary set for validation/test
    # stratifying if possible
    train_ids, temp_ids = train_test_split(
        ids, test_size=float(data_cfg["validation_size"]) + float(data_cfg["test_size"]),
        random_state=seed, stratify=stratify,
    )
    # further split the temporary set into validation and test sets, stratifying if possible
    temp = response_df[response_df[group_column].isin(temp_ids)]
    temp_labels = temp["label"].to_numpy()
    temp_stratify = temp_labels if len(np.unique(temp_labels)) > 1 and pd.Series(temp_labels).value_counts().min() >= 2 else None
    test_fraction = float(data_cfg["test_size"]) / (float(data_cfg["validation_size"]) + float(data_cfg["test_size"]))
    val_ids, test_ids = train_test_split(
        temp[group_column].to_numpy(), test_size=test_fraction,
        random_state=seed + 1, stratify=temp_stratify,
    )

    return {
        name: sentence_df.index[sentence_df[group_column].isin(group_ids)].to_numpy()
        for name, group_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
