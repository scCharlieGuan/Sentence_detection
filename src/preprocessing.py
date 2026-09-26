"""Text cleaning, sentence splitting, span alignment, and weak-label construction."""
from __future__ import annotations

import ast
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from src.splitting import split_train_val_test

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
        logging.warning(
            "spaCy model '%s' is unavailable; using spacy.blank('en') with the "
            "rule-based sentencizer. Sentence boundaries may differ.",
            model_name,
        )
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
    # CounselBench contains non-breaking spaces. Treat every horizontal Unicode
    # whitespace character consistently so copy/paste spans do not need fuzzy
    # matching merely because one side contains U+00A0.
    value = re.sub(r"[^\S\n]+", " ", value)
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
    return _fuzzy_score_normalized(a_norm, b_norm)


def _fuzzy_score_normalized(a_norm: str, b_norm: str) -> float:
    """Score two already normalized strings without repeated spaCy passes."""
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


def _token_overlap_ratio(
    source: str,
    target: str,
    nlp: spacy.language.Language,
) -> float:
    """Measure how much of an annotation's normalized vocabulary is retained."""
    source_tokens = set(normalize_for_match(source, nlp).split())
    target_tokens = set(normalize_for_match(target, nlp).split())
    if not source_tokens:
        return 0.0
    return float(len(source_tokens & target_tokens) / len(source_tokens))


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
    selection_policy: str = "best_window",
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
        Alignment metadata and matched sentence indices. ``best_window`` selects
        one contiguous candidate and therefore cannot silently union unrelated
        windows. ``union`` is retained only for reproducing legacy experiments.
    """
    if selection_policy not in {"best_window", "union"}:
        raise ValueError("selection_policy must be 'best_window' or 'union'.")
    if is_invalid_copy(copy) or not sentences:
        return {
            "matched_indices": [], "method": "empty", "score": 0.0,
            "overlap_ratio": 0.0,
            "runner_up_score": 0.0,
            "score_margin": 0.0,
            "candidate_count": 0,
            "failure_reason": "invalid_copy" if is_invalid_copy(copy) else "no_sentences",
        }
    if normalize_text(copy).lower() in {value.lower() for value in entire_response_values}:
        return {
            "matched_indices": list(range(len(sentences))),
            "method": "all_response",
            "score": 1.0,
            "overlap_ratio": 1.0,
            "runner_up_score": 0.0,
            "score_margin": 1.0,
            "candidate_count": 1,
            "failure_reason": "",
        }

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
            covered = sum(
                max(0, min(end, exact[1]) - max(start, exact[0]))
                for _, start, end in sentences
            )
            return {
                "matched_indices": sorted(matched),
                "method": "exact_char_overlap",
                "score": 1.0,
                "overlap_ratio": float(min(1.0, covered / max(1, len(copy_lower)))),
                "runner_up_score": 0.0,
                "score_margin": 1.0,
                "candidate_count": 1,
                "failure_reason": "",
            }

    copy_norm = normalize_for_match(copy, nlp)
    sentence_norms = [normalize_for_match(sentence, nlp) for sentence, _, _ in sentences]
    candidates: List[Dict[str, Any]] = []
    for start in range(len(sentences)):
        for size in range(1, max_window + 1):
            end = start + size
            if end > len(sentences):
                break
            window_norm = " ".join(sentence_norms[start:end])
            score = _fuzzy_score_normalized(copy_norm, window_norm)
            matched_text = " ".join(sentences[index][0] for index in range(start, end))
            candidates.append({
                "start": start,
                "end": end,
                "size": size,
                "score": float(score),
                "overlap_ratio": _token_overlap_ratio(copy, matched_text, nlp),
                "matched_text": matched_text,
            })
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["score"], -item["overlap_ratio"], item["size"], item["start"]
        ),
    )
    best = ranked[0] if ranked else None
    runner_up = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    best_score = float(best["score"]) if best is not None else 0.0

    if selection_policy == "union":
        matched = {
            index
            for candidate in candidates
            if candidate["score"] >= fuzzy_threshold
            for index in range(candidate["start"], candidate["end"])
        }
        if matched:
            matched_text = " ".join(sentences[index][0] for index in sorted(matched))
            return {
                "matched_indices": sorted(matched),
                "method": "fuzzy_sentence_or_window_union",
                "score": best_score,
                "overlap_ratio": _token_overlap_ratio(copy, matched_text, nlp),
                "runner_up_score": runner_up,
                "score_margin": best_score - runner_up,
                "candidate_count": len(candidates),
                "failure_reason": "",
            }

    if best is not None and best_score >= fuzzy_threshold:
        matched_indices = list(range(int(best["start"]), int(best["end"])))
        return {
            "matched_indices": matched_indices,
            "method": "fuzzy_best_window",
            "score": best_score,
            "overlap_ratio": float(best["overlap_ratio"]),
            "runner_up_score": runner_up,
            "score_margin": best_score - runner_up,
            "candidate_count": len(candidates),
            "failure_reason": "",
        }
    if fallback_threshold is not None and best is not None and best_score >= fallback_threshold:
        matched_indices = list(range(int(best["start"]), int(best["end"])))
        return {
            "matched_indices": matched_indices,
            "method": "fuzzy_best_window_fallback",
            "score": best_score,
            "overlap_ratio": float(best["overlap_ratio"]),
            "runner_up_score": runner_up,
            "score_margin": best_score - runner_up,
            "candidate_count": len(candidates),
            "failure_reason": "",
        }
    return {
        "matched_indices": [], "method": "unmatched", "score": best_score,
        "overlap_ratio": 0.0,
        "runner_up_score": runner_up,
        "score_margin": best_score - runner_up,
        "candidate_count": len(candidates),
        "failure_reason": "below_fuzzy_fallback_threshold",
    }


def _annotation_group_columns(
    frame: pd.DataFrame,
    response_column: str,
    data_cfg: Dict[str, Any],
) -> List[str]:
    """Choose the response instance used for agreement aggregation.

    Identical generic response text can occur under different questions. Those
    instances must not pool annotator votes, so configured metadata columns are
    preferred and raw response text remains part of the key.
    """
    configured = list(data_cfg.get("annotation_group_columns", ["questionID", "responder"]))
    available = [column for column in configured if column in frame.columns]
    return [*available, response_column] if available else [response_column]


def _is_reliable_alignment(result: Dict[str, Any], data_cfg: Dict[str, Any]) -> bool:
    """Apply the declared conservative alignment reliability definition."""
    reliable_methods = set(data_cfg.get("reliable_alignment_methods", ["exact_char_overlap"]))
    if result["method"] in reliable_methods:
        return True
    return bool(
        result["method"] in {"fuzzy_best_fallback", "fuzzy_best_window_fallback"}
        and float(result["score"]) >= float(data_cfg.get("reliable_fuzzy_threshold", 0.90))
        and float(result["overlap_ratio"]) >= float(data_cfg.get("reliable_min_overlap_ratio", 0.50))
    )


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

    group_columns = _annotation_group_columns(work, response_col, data_cfg)
    group_keys = pd.MultiIndex.from_frame(work[group_columns])
    work["response_id"] = pd.factorize(group_keys, sort=False)[0]
    work["response_text_id"] = pd.factorize(work[response_col], sort=False)[0]

    rows: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    grouped = work.groupby(["response_id", "response_text_id", response_col], sort=False)
    for (response_id, response_text_id, response), group in tqdm(
        grouped, desc=f"Building {task['name']} labels"
    ):
        sentences = split_sentences(response, nlp)
        sentence_annotators: List[Set[str]] = [set() for _ in sentences]
        reliable_annotators: List[Set[str]] = [set() for _ in sentences]

        for row_index, annotation in group.iterrows():
            annotator_id = str(annotation.get("survey_id", row_index))
            copies = parse_copy_cell(annotation[target_col])
            annotator_matches: Set[int] = set()
            annotator_reliable_matches: Set[int] = set()
            for copy in copies:
                result = align_copy_to_sentences(
                    response, copy, sentences, nlp,
                    data_cfg["entire_response_values"],
                    float(data_cfg["fuzzy_threshold"]),
                    float(data_cfg["fuzzy_fallback_threshold"]),
                    float(data_cfg["min_overlap_ratio"]),
                    int(data_cfg["max_fuzzy_window"]),
                    str(data_cfg.get("fuzzy_selection_policy", "best_window")),
                )
                annotator_matches.update(result["matched_indices"])
                reliable = _is_reliable_alignment(result, data_cfg)
                if reliable:
                    annotator_reliable_matches.update(result["matched_indices"])
                matched_text = " || ".join(
                    sentences[index][0] for index in result["matched_indices"]
                )
                method = str(result["method"])
                method_group = (
                    "exact" if method == "exact_char_overlap"
                    else "fuzzy" if method.startswith("fuzzy")
                    else method
                )
                logs.append({
                    "response_id": response_id,
                    "response_text_id": response_text_id,
                    "source_row_index": int(row_index),
                    "annotator_id": annotator_id,
                    "task": task["name"],
                    "target_column": target_col,
                    "original_span": copy,
                    "copy": copy,
                    "span_length": len(copy),
                    "alignment_method": method_group,
                    "method": result["method"],
                    "score": result["score"],
                    "runner_up_score": result.get("runner_up_score"),
                    "score_margin": result.get("score_margin"),
                    "candidate_count": result.get("candidate_count"),
                    "fuzzy_similarity_score": result["score"] if method.startswith("fuzzy") else None,
                    "overlap_ratio": result["overlap_ratio"],
                    "failure_reason": result.get("failure_reason", ""),
                    "matched_indices": json.dumps(result["matched_indices"]),
                    "matched_sentence": matched_text,
                    "matched_sentence_count": len(result["matched_indices"]),
                    "crosses_sentence": len(result["matched_indices"]) > 1,
                    "reliable_alignment": reliable,
                    **{
                        column: normalize_text(annotation[column])
                        for column in ("questionID", "responder", "topic")
                        if column in group.columns
                    },
                    **{
                        column: normalize_text(annotation[column])
                        for column in (
                            target_col.replace("_copy", "_score"),
                            target_col.replace("_copy", "_reason"),
                        )
                        if column in group.columns
                    },
                })
            for index in annotator_matches:
                sentence_annotators[index].add(annotator_id)
            for index in annotator_reliable_matches:
                reliable_annotators[index].add(annotator_id)

        counts = np.asarray([len(values) for values in sentence_annotators], dtype=int)
        reliable_counts = np.asarray([len(values) for values in reliable_annotators], dtype=int)
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
                "response_id": response_id,
                "response_text_id": response_text_id,
                "sentence_id": sentence_id,
                "sentence": sentence, "start": start, "end": end,
                "label": int(labels[sentence_id]),
                "label_original": int(labels[sentence_id]),
                "match_count": int(counts[sentence_id]),
                "positive_annotators": int(counts[sentence_id]),
                "reliable_positive_annotators": int(reliable_counts[sentence_id]),
                "total_annotators": int(group["survey_id"].nunique())
                if "survey_id" in group.columns else int(len(group)),
                "agreement_ratio": float(counts[sentence_id] / max(1, group["survey_id"].nunique()))
                if "survey_id" in group.columns else float(counts[sentence_id] / max(1, len(group))),
                "response": response,
                **metadata,
            })
    sentence_df = pd.DataFrame(rows)
    if sentence_df.empty:
        raise ValueError("No sentences were generated from the configured response column.")
    return sentence_df, pd.DataFrame(logs)
