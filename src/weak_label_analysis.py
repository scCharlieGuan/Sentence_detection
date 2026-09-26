"""Weak-label diagnostics and manual-audit exports."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import ensure_dir, save_json
from src.weak_labels import (
    annotator_agreement_summary,
    apply_label_rules,
    label_column_name,
    label_rule_summary,
    save_label_datasets,
)


def _parse_indices(value: Any) -> list[int]:
    """Parse JSON-encoded matched sentence indices."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"matched_indices must contain a JSON list: {value!r}")
    return [int(index) for index in parsed]


def summarize_alignment(alignment_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize exact, fuzzy, unmatched, and cross-sentence alignments.

    Args:
        alignment_df: One row per annotation span.

    Returns:
        JSON-serializable alignment diagnostics.
    """
    if alignment_df.empty:
        return {"total_spans": 0, "alignment_groups": {}}
    frame = alignment_df.copy()
    if "alignment_method" not in frame:
        frame["alignment_method"] = frame["method"].map(
            lambda value: "exact" if value == "exact_char_overlap"
            else "fuzzy" if str(value).startswith("fuzzy")
            else str(value)
        )
    if "matched_sentence_count" not in frame:
        frame["matched_sentence_count"] = frame["matched_indices"].map(
            lambda value: len(_parse_indices(value))
        )
    if "crosses_sentence" not in frame:
        frame["crosses_sentence"] = frame["matched_sentence_count"] > 1

    groups: dict[str, Any] = {}
    for method, group in frame.groupby("alignment_method", dropna=False):
        score = pd.to_numeric(
            group["score"] if "score" in group else pd.Series(np.nan, index=group.index),
            errors="coerce",
        )
        overlap = pd.to_numeric(
            group["overlap_ratio"]
            if "overlap_ratio" in group else pd.Series(np.nan, index=group.index),
            errors="coerce",
        )
        groups[str(method)] = {
            "spans": int(len(group)),
            "ratio": float(len(group) / len(frame)),
            "matched_sentence_assignments": int(group["matched_sentence_count"].sum()),
            "cross_sentence_spans": int(group["crosses_sentence"].astype(bool).sum()),
            "mean_similarity_score": float(score.mean()) if score.notna().any() else None,
            "median_similarity_score": float(score.median()) if score.notna().any() else None,
            "mean_overlap_ratio": float(overlap.mean()) if overlap.notna().any() else None,
        }
    fuzzy = frame[frame["alignment_method"] == "fuzzy"]
    unmatched = frame[frame["alignment_method"] == "unmatched"]
    fuzzy_scores = pd.to_numeric(
        fuzzy.get("fuzzy_similarity_score", fuzzy.get("score", pd.Series(dtype=float))),
        errors="coerce",
    ).dropna()
    source_score_columns = [column for column in frame if column.endswith("_score")]
    return {
        "total_spans": int(len(frame)),
        "matched_spans": int(len(frame) - len(unmatched)),
        "matched_ratio": float(1.0 - len(unmatched) / len(frame)),
        "unmatched_spans": int(len(unmatched)),
        "unmatched_ratio": float(len(unmatched) / len(frame)),
        "fuzzy_spans": int(len(fuzzy)),
        "fuzzy_cross_sentence_spans": int(fuzzy["crosses_sentence"].astype(bool).sum()),
        "fuzzy_similarity_distribution": {
            "min": float(fuzzy_scores.min()),
            "p10": float(fuzzy_scores.quantile(0.10)),
            "median": float(fuzzy_scores.median()),
            "p90": float(fuzzy_scores.quantile(0.90)),
            "max": float(fuzzy_scores.max()),
        } if len(fuzzy_scores) else {},
        "failure_reasons": {
            str(key): int(value)
            for key, value in frame.get(
                "failure_reason", pd.Series("not_recorded", index=frame.index)
            ).fillna("").replace("", "not_applicable").value_counts().items()
        },
        "source_score_by_alignment_method": {
            column: {
                str(score_value): {
                    str(method): int(count)
                    for method, count in group["alignment_method"].value_counts().items()
                }
                for score_value, group in frame.groupby(column, dropna=False)
            }
            for column in source_score_columns
        },
        "alignment_groups": groups,
    }


def _load_predictions(
    prediction_files: Mapping[str, Path],
    sentence_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Load only predictions that match the current prepared row identities."""
    predictions: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, float] = {}
    for name, path in prediction_files.items():
        resolved = Path(path).resolve()
        if not resolved.exists():
            continue
        frame = pd.read_csv(resolved)
        required = {"row_index", "probability", "prediction"}
        if not required.issubset(frame):
            raise ValueError(f"{resolved} lacks prediction columns: {sorted(required - set(frame))}")
        if frame["row_index"].duplicated().any():
            raise ValueError(f"{resolved} contains duplicate row_index values.")
        row_indices = frame["row_index"].to_numpy(dtype=int)
        if len(row_indices) and (row_indices.min() < 0 or row_indices.max() >= len(sentence_df)):
            continue
        if "sentence" in frame:
            current_sentences = sentence_df.iloc[row_indices]["sentence"].astype(str).to_numpy()
            if not np.array_equal(current_sentences, frame["sentence"].astype(str).to_numpy()):
                # Old predictions must never be silently attached after data rebuilding.
                continue
        predictions[name] = frame[["row_index", "probability", "prediction"]].copy()
        metrics_path = resolved.parent / "test_metrics.json"
        threshold = 0.5
        if metrics_path.exists():
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            threshold = float(
                report.get("threshold", report.get("test_metrics", {}).get("threshold", 0.5))
            )
        thresholds[name] = threshold
    return predictions, thresholds


def _normalized_sentence(text: Any) -> str:
    """Build a conservative key for detecting label conflicts."""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def _alignment_sentence_rows(
    sentences: pd.DataFrame,
    alignment: pd.DataFrame,
) -> pd.DataFrame:
    """Expand spans to matched sentences while retaining unmatched spans."""
    spans = alignment.copy()
    spans["_indices"] = spans["matched_indices"].map(_parse_indices)
    spans["sentence_id"] = spans["_indices"].map(lambda values: values or [pd.NA])
    spans = spans.explode("sentence_id", ignore_index=True)

    sentence_metadata = sentences.copy()
    sentence_metadata["row_index"] = np.arange(len(sentence_metadata), dtype=int)
    matched = spans[spans["sentence_id"].notna()].copy()
    matched["sentence_id"] = matched["sentence_id"].astype(int)
    matched = matched.merge(
        sentence_metadata,
        on=["response_id", "sentence_id"],
        how="left",
        validate="many_to_one",
        suffixes=("_span", ""),
    )

    response_metadata = (
        sentence_metadata.sort_values("sentence_id")
        .drop_duplicates("response_id")
        .drop(columns=["sentence_id", "sentence", "start", "end", "row_index"], errors="ignore")
    )
    unmatched = spans[spans["sentence_id"].isna()].copy().merge(
        response_metadata,
        on="response_id",
        how="left",
        suffixes=("_span", ""),
    )
    unmatched["row_index"] = pd.NA
    unmatched["sentence"] = ""

    represented = set(matched["row_index"].dropna().astype(int))
    unrepresented = sentence_metadata[~sentence_metadata["row_index"].isin(represented)].copy()
    unrepresented["method"] = "no_positive_span"
    unrepresented["alignment_method"] = "none"
    unrepresented["original_span"] = ""
    unrepresented["annotator_id"] = ""
    return pd.concat([matched, unmatched, unrepresented], ignore_index=True, sort=False)


def build_weak_label_audit(
    sentence_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    prediction_files: Mapping[str, Path] | None = None,
    strict_rule: str = "consensus_2_reliable",
    strong_disagreement: float = 0.40,
    borderline_margin: float = 0.10,
) -> pd.DataFrame:
    """Build a span-aware manual audit table with risk flags.

    Args:
        sentence_df: Sentence dataset containing all derived label columns.
        alignment_df: Span alignment log.
        prediction_files: Optional short model name to prediction CSV mapping.
        strict_rule: Derived rule shown as ``strict_weak_label``.
        strong_disagreement: Absolute TF-IDF/ALBERT probability gap to flag.
        borderline_margin: Distance from each saved decision threshold.

    Returns:
        Audit rows covering every sentence and every unmatched span.
    """
    audit = _alignment_sentence_rows(sentence_df, alignment_df)
    strict_column = label_column_name(strict_rule)
    if strict_column not in audit:
        raise ValueError(f"Strict label column is absent: {strict_column}")
    audit["original_weak_label"] = audit["label_original"]
    audit["strict_weak_label"] = audit[strict_column]
    if "original_span" not in audit and "copy" in audit:
        audit["original_span"] = audit["copy"]

    predictions, thresholds = _load_predictions(prediction_files or {}, sentence_df)
    for name, frame in predictions.items():
        audit = audit.merge(
            frame.rename(columns={
                "probability": f"{name}_probability",
                "prediction": f"{name}_prediction",
            }),
            on="row_index",
            how="left",
            validate="many_to_one",
        )

    sentence_keys = sentence_df["sentence"].map(_normalized_sentence)
    conflicts = set(
        sentence_keys.groupby(sentence_keys).filter(
            lambda values: sentence_df.loc[values.index, "label_original"].nunique() > 1
        )
    )
    audit["flag_unmatched_span"] = audit["alignment_method"].eq("unmatched")
    audit["flag_fuzzy_alignment"] = audit["alignment_method"].eq("fuzzy")
    audit["flag_conflicting_labels"] = audit["sentence"].map(_normalized_sentence).isin(conflicts)
    audit["flag_single_annotator_positive"] = (
        audit["original_weak_label"].eq(1) & audit["positive_annotators"].eq(1)
    )
    audit["flag_multi_annotator_positive"] = (
        audit["original_weak_label"].eq(1) & audit["positive_annotators"].ge(2)
    )
    audit["flag_exact_aligned_positive"] = (
        audit["original_weak_label"].eq(1) & audit["alignment_method"].eq("exact")
    )
    audit["flag_fuzzy_aligned_positive"] = (
        audit["original_weak_label"].eq(1) & audit["alignment_method"].eq("fuzzy")
    )
    audit["flag_context_dependent_proxy"] = audit.get(
        "crosses_sentence", pd.Series(False, index=audit.index)
    ).fillna(False).astype(bool)

    prediction_columns = [f"{name}_prediction" for name in predictions]
    if prediction_columns:
        available_votes = audit[prediction_columns].notna().sum(axis=1)
        audit["flag_models_agree_negative_label_positive"] = (
            audit["original_weak_label"].eq(1)
            & available_votes.eq(len(prediction_columns))
            & audit[prediction_columns].fillna(1).eq(0).all(axis=1)
        )
    else:
        audit["flag_models_agree_negative_label_positive"] = False

    if {"tfidf_probability", "albert_probability"}.issubset(audit):
        audit["flag_tfidf_albert_strong_disagreement"] = (
            audit["tfidf_probability"] - audit["albert_probability"]
        ).abs().ge(strong_disagreement)
    else:
        audit["flag_tfidf_albert_strong_disagreement"] = False

    borderline_columns: list[pd.Series] = []
    for name, threshold in thresholds.items():
        borderline_columns.append(
            audit[f"{name}_probability"].sub(threshold).abs().le(borderline_margin)
        )
    audit["flag_low_confidence_borderline"] = (
        pd.concat(borderline_columns, axis=1).all(axis=1)
        if borderline_columns else False
    )
    audit["flag_potentially_noisy_label"] = (
        audit["flag_unmatched_span"]
        | audit["flag_fuzzy_alignment"]
        | audit["flag_conflicting_labels"]
        | audit["flag_single_annotator_positive"]
        | audit["flag_models_agree_negative_label_positive"]
    )
    flag_columns = [column for column in audit if column.startswith("flag_")]
    audit["audit_flags"] = audit.apply(
        lambda row: ";".join(column.removeprefix("flag_") for column in flag_columns if bool(row[column])),
        axis=1,
    )

    preferred = [
        "audit_flags", "questionID", "response_id", "responder", "topic", "response",
        "sentence_id", "sentence", "original_span", "annotator_id", "alignment_method",
        "method", "score", "fuzzy_similarity_score", "overlap_ratio", "span_length",
        "crosses_sentence", "positive_annotators", "total_annotators", "agreement_ratio",
        "original_weak_label", "strict_weak_label", "tfidf_probability",
        "albert_probability", *flag_columns,
    ]
    existing = [column for column in preferred if column in audit]
    remaining = [column for column in audit if column not in existing and column != "_indices"]
    return audit[existing + remaining].sort_values(
        ["response_id", "sentence_id"], na_position="last"
    )


def run_weak_label_analysis(
    sentence_df: pd.DataFrame,
    splits: Mapping[str, Sequence[int]],
    alignment_df: pd.DataFrame,
    rules: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    prediction_files: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Run the complete weak-label analysis and save concentrated outputs."""
    output_dir = ensure_dir(output_dir)
    sentences = apply_label_rules(sentence_df, rules)
    rule_names = list(rules)

    alignment_path = output_dir / "alignment_summary.json"
    agreement_path = output_dir / "annotator_agreement.csv"
    agreement_sentences_path = output_dir / "annotator_agreement_sentences.csv"
    rule_summary_path = output_dir / "label_rule_summary.csv"
    audit_path = output_dir / "weak_label_audit.csv"
    model_summary_path = output_dir / "model_by_label_rule.csv"

    save_json(summarize_alignment(alignment_df), alignment_path)
    annotator_agreement_summary(sentences, rule_names).to_csv(agreement_path, index=False)
    agreement_columns = [
        column for column in (
            "response_id", "response_text_id", "sentence_id", "questionID", "responder",
            "sentence", "positive_annotators", "reliable_positive_annotators",
            "total_annotators", "agreement_ratio", "label_original",
            *(label_column_name(name) for name in rule_names),
        ) if column in sentences
    ]
    sentences[agreement_columns].to_csv(agreement_sentences_path, index=False)
    label_rule_summary(sentences, splits, rules).to_csv(rule_summary_path, index=False)
    save_label_datasets(sentences, rules, output_dir)
    build_weak_label_audit(
        sentences, alignment_df, prediction_files=prediction_files
    ).to_csv(audit_path, index=False, encoding="utf-8-sig")
    if not model_summary_path.exists():
        pd.DataFrame(columns=[
            "label_rule", "model", "seed", "train_positive_samples",
            "positive_rate", "pr_auc", "roc_auc", "precision", "recall", "f1",
        ]).to_csv(model_summary_path, index=False)
    return {
        "alignment_summary": alignment_path,
        "annotator_agreement": agreement_path,
        "annotator_agreement_sentences": agreement_sentences_path,
        "label_rule_summary": rule_summary_path,
        "weak_label_audit": audit_path,
        "model_by_label_rule": model_summary_path,
    }
