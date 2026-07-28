"""Dataset diagnostics, strong lexical baselines, and error-analysis exports."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import (
    aggregate_seed_metrics,
    bootstrap_metric_difference,
    build_prediction_frame,
    classification_metrics,
    find_best_threshold,
    mcnemar_exact,
)
from src.input_features import build_model_inputs
from src.utils import ensure_dir, save_json


def _normalise_duplicate_key(text: str) -> str:
    """Normalize surface variation for conservative duplicate checks."""
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def audit_dataset(
    sentence_df: pd.DataFrame,
    splits: Mapping[str, np.ndarray],
    alignment_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Audit split integrity, duplicates, weak labels, and alignment quality.

    Args:
        sentence_df: Prepared sentence dataset.
        splits: Shared train/validation/test indices.
        alignment_df: Optional span-alignment log.

    Returns:
        JSON-serializable diagnostic report.
    """
    expected = set(range(len(sentence_df)))
    split_sets = {name: set(np.asarray(indices, dtype=int)) for name, indices in splits.items()}
    overlap = {
        f"{left}_{right}": len(split_sets[left] & split_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    union = set().union(*split_sets.values())

    group_sets = {
        name: set(sentence_df.iloc[list(indices)]["response_id"].tolist())
        for name, indices in split_sets.items()
    }
    group_overlap = {
        f"{left}_{right}": len(group_sets[left] & group_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }

    split_lookup = {}
    for name, indices in splits.items():
        split_lookup.update({int(index): name for index in indices})
    duplicate_frame = sentence_df[["sentence", "label"]].copy()
    duplicate_frame["key"] = duplicate_frame["sentence"].map(_normalise_duplicate_key)
    duplicate_frame["split"] = [split_lookup.get(index, "missing") for index in range(len(sentence_df))]
    duplicate_groups = duplicate_frame.groupby("key", dropna=False)
    cross_split_duplicates = int(
        sum(group["split"].nunique() > 1 for _, group in duplicate_groups if len(group) > 1)
    )
    conflicting_text_labels = int(
        sum(group["label"].nunique() > 1 for _, group in duplicate_groups if len(group) > 1)
    )

    split_summary: Dict[str, Any] = {}
    for name, indices in splits.items():
        part = sentence_df.iloc[np.asarray(indices, dtype=int)]
        split_summary[name] = {
            "rows": int(len(part)),
            "responses": int(part["response_id"].nunique()),
            "positive": int(part["label"].sum()),
            "prevalence": float(part["label"].mean()),
        }

    near_duplicate_summary: Dict[str, Any] = {}
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50_000
    )
    duplicate_features = vectorizer.fit_transform(sentence_df["sentence"].fillna(""))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_indices = np.asarray(splits[left], dtype=int)
        right_indices = np.asarray(splits[right], dtype=int)
        similarity = cosine_similarity(
            duplicate_features[right_indices], duplicate_features[left_indices]
        )
        maxima = similarity.max(axis=1) if similarity.size else np.asarray([])
        near_duplicate_summary[f"{left}_{right}"] = {
            "right_rows_similarity_gte_0.90": int(np.sum(maxima >= 0.90)),
            "right_rows_similarity_gte_0.95": int(np.sum(maxima >= 0.95)),
            "maximum_similarity": float(maxima.max()) if len(maxima) else 0.0,
        }

    surface = pd.DataFrame(index=sentence_df.index)
    surface["label"] = sentence_df["label"].astype(int)
    surface["characters"] = sentence_df["sentence"].astype(str).str.len()
    surface["tokens"] = sentence_df["sentence"].astype(str).str.split().str.len()
    surface["punctuation"] = sentence_df["sentence"].astype(str).str.count(r"[!?.,;:]")
    surface["has_negation"] = sentence_df["sentence"].astype(str).str.contains(
        r"\b(?:no|not|never|n't)\b", case=False, regex=True
    ).astype(int)
    surface["has_should"] = sentence_df["sentence"].astype(str).str.contains(
        r"\bshould\b", case=False, regex=True
    ).astype(int)
    surface_summary = {
        str(label): {
            column: float(value)
            for column, value in group.drop(columns="label").mean().items()
        }
        for label, group in surface.groupby("label")
    }

    report: Dict[str, Any] = {
        "rows": int(len(sentence_df)),
        "responses": int(sentence_df["response_id"].nunique()),
        "positive": int(sentence_df["label"].sum()),
        "prevalence": float(sentence_df["label"].mean()),
        "split_summary": split_summary,
        "row_overlap": overlap,
        "group_overlap": group_overlap,
        "all_rows_assigned_once": union == expected and sum(map(len, split_sets.values())) == len(expected),
        "missing_rows": int(len(expected - union)),
        "exact_or_normalized_duplicate_groups_across_splits": cross_split_duplicates,
        "near_duplicate_cross_split": near_duplicate_summary,
        "conflicting_labels_for_identical_sentence_text": conflicting_text_labels,
        "multi_sentence_positive_span_proxy": int(
            sentence_df.groupby("response_id")["label"].sum().gt(1).sum()
        ),
        "surface_feature_means_by_label": surface_summary,
    }
    for candidate_group in ("questionID",):
        if candidate_group in sentence_df:
            candidate_sets = {
                name: set(
                    sentence_df.iloc[np.asarray(indices, dtype=int)][candidate_group]
                    .dropna()
                    .astype(str)
                )
                for name, indices in splits.items()
            }
            report[f"{candidate_group}_overlap"] = {
                f"{left}_{right}": int(len(candidate_sets[left] & candidate_sets[right]))
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            }
    for source_column in ("responder", "topic"):
        if source_column in sentence_df:
            source = (
                sentence_df.groupby(source_column)["label"]
                .agg(["count", "sum", "mean"])
                .sort_values("mean", ascending=False)
            )
            report[f"prevalence_by_{source_column}"] = {
                str(index): {
                    "rows": int(row["count"]),
                    "positive": int(row["sum"]),
                    "prevalence": float(row["mean"]),
                }
                for index, row in source.iterrows()
            }
    if "positive_annotators" in sentence_df:
        positives = sentence_df.loc[sentence_df["label"] == 1, "positive_annotators"]
        report["positive_annotator_count"] = {
            "median": float(positives.median()) if len(positives) else 0.0,
            "single_annotator_ratio": float((positives == 1).mean()) if len(positives) else 0.0,
        }
    if alignment_df is not None and not alignment_df.empty:
        method_column = "method" if "method" in alignment_df else "match_method"
        report["alignment"] = {
            "total_spans": int(len(alignment_df)),
            "methods": {
                str(key): int(value)
                for key, value in alignment_df[method_column].value_counts(dropna=False).items()
            },
            "unmatched_ratio": float((alignment_df[method_column] == "unmatched").mean()),
            "fuzzy_or_fallback_ratio": float(
                alignment_df[method_column].astype(str).str.startswith("fuzzy").mean()
            ),
        }
    return report


def _baseline_pipeline(
    representation: str,
    classifier: str,
    class_weight: str | None,
    seed: int,
    c_value: float,
) -> Pipeline:
    """Construct one declared lexical baseline without fitting."""
    word_uni = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), min_df=2, max_df=0.95, sublinear_tf=True
    )
    word_bi = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100_000,
        sublinear_tf=True,
    )
    representations: Dict[str, Any] = {
        "word_unigram": word_uni,
        "word_bigram": word_bi,
        "char": char,
        "word_char": FeatureUnion([("word", word_bi), ("char", char)]),
    }
    if classifier == "lr":
        estimator: Any = LogisticRegression(
            C=c_value,
            max_iter=3000,
            class_weight=class_weight,
            solver="liblinear",
            random_state=seed,
        )
    elif classifier == "svm":
        estimator = CalibratedClassifierCV(
            LinearSVC(
                C=c_value,
                class_weight=class_weight,
                random_state=seed,
                max_iter=5000,
            ),
            method="sigmoid",
            cv=3,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    return Pipeline([("features", representations[representation]), ("clf", estimator)])


BASELINE_SPECS: Tuple[Tuple[str, str, str | None], ...] = (
    ("word_unigram", "lr", None),
    ("word_bigram", "lr", None),
    ("char", "lr", None),
    ("word_char", "lr", None),
    ("word_unigram", "svm", None),
    ("word_bigram", "svm", None),
    ("word_bigram", "lr", "balanced"),
    ("word_char", "svm", "balanced"),
)


def _top_linear_features(model: Pipeline, top_n: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """Extract positive/negative unigram and bigram weights from word LR."""
    vectorizer = model.named_steps["features"]
    classifier = model.named_steps["clf"]
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(classifier, "coef_"):
        return {}
    names = np.asarray(vectorizer.get_feature_names_out())
    weights = np.asarray(classifier.coef_[0])
    output: Dict[str, List[Dict[str, Any]]] = {}
    for size, label in ((1, "unigram"), (2, "bigram")):
        mask = np.asarray([len(name.split()) == size for name in names])
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        order = indices[np.argsort(weights[indices])]
        output[f"negative_{label}"] = [
            {"feature": str(names[index]), "weight": float(weights[index])}
            for index in order[:top_n]
        ]
        output[f"positive_{label}"] = [
            {"feature": str(names[index]), "weight": float(weights[index])}
            for index in order[-top_n:][::-1]
        ]
    return output


def run_baseline_suite(
    sentence_df: pd.DataFrame,
    splits: Mapping[str, np.ndarray],
    report_dir: Path,
    seeds: Sequence[int],
    threshold_grid: Sequence[float],
    input_mode: str = "sentence",
) -> pd.DataFrame:
    """Run eight declared TF-IDF baselines on the shared protocol.

    Regularization is selected by validation PR-AUC, then the decision threshold
    is selected by validation F1. The fitted train-only model is not refit after
    threshold selection, keeping its probability scale unchanged.
    """
    texts = np.asarray(build_model_inputs(sentence_df, input_mode), dtype=object)
    labels = sentence_df["label"].to_numpy(dtype=int)
    train_idx = np.asarray(splits["train"], dtype=int)
    val_idx = np.asarray(splits["val"], dtype=int)
    test_idx = np.asarray(splits["test"], dtype=int)
    ensure_dir(report_dir)

    rows: List[Dict[str, Any]] = []
    features_written = False
    for representation, classifier, class_weight in BASELINE_SPECS:
        experiment = f"{representation}_{classifier}_{class_weight or 'unweighted'}"
        run_metrics: List[Mapping[str, float]] = []
        for seed in seeds:
            candidates: List[Tuple[float, float, Pipeline, np.ndarray]] = []
            for c_value in (0.01, 0.1, 1.0, 10.0):
                model = _baseline_pipeline(
                    representation, classifier, class_weight, int(seed), c_value
                )
                model.fit(texts[train_idx], labels[train_idx])
                val_probability = model.predict_proba(texts[val_idx])[:, 1]
                val_metrics = classification_metrics(labels[val_idx], val_probability, 0.5)
                candidates.append((val_metrics["pr_auc"], c_value, model, val_probability))
            _, best_c, best_model, val_probability = max(candidates, key=lambda item: item[0])
            threshold, _ = find_best_threshold(
                labels[val_idx], val_probability, threshold_grid, metric="f1"
            )
            test_probability = best_model.predict_proba(texts[test_idx])[:, 1]
            metrics = classification_metrics(labels[test_idx], test_probability, threshold)
            run_metrics.append({key: value for key, value in metrics.items() if np.isscalar(value)})
            rows.append({
                "experiment": experiment,
                "representation": representation,
                "classifier": classifier,
                "class_weight": class_weight or "none",
                "seed": int(seed),
                "best_c": float(best_c),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            })
            prediction = build_prediction_frame(sentence_df, test_idx, test_probability, threshold)
            prediction.to_csv(report_dir / f"{experiment}_seed_{seed}_predictions.csv", index=False)

            if (
                not features_written
                and representation == "word_bigram"
                and classifier == "lr"
                and class_weight is None
            ):
                save_json(_top_linear_features(best_model), report_dir / "tfidf_top_features.json")
                features_written = True
        save_json(
            aggregate_seed_metrics(run_metrics),
            report_dir / f"{experiment}_seed_summary.json",
        )
    result = pd.DataFrame(rows)
    result.to_csv(report_dir / "baseline_results.csv", index=False)
    return result


def export_error_audit(
    sentence_df: pd.DataFrame,
    prediction_files: Mapping[str, Path],
    output_path: Path,
    per_category: int = 50,
) -> pd.DataFrame:
    """Export confidence-ranked disagreements for manual label review.

    Args:
        sentence_df: Full sentence data.
        prediction_files: Model name to prediction CSV.
        output_path: Destination CSV.
        per_category: Maximum rows per audit category.

    Returns:
        Combined audit frame.
    """
    merged: pd.DataFrame | None = None
    for model_name, path in prediction_files.items():
        frame = pd.read_csv(path)
        required = {"row_index", "label", "probability", "prediction"}
        if not required.issubset(frame):
            raise ValueError(f"{path} is missing prediction columns: {sorted(required - set(frame))}")
        part = frame[["row_index", "label", "probability", "prediction"]].rename(
            columns={
                "probability": f"{model_name}_probability",
                "prediction": f"{model_name}_prediction",
            }
        )
        merged = part if merged is None else merged.merge(part, on=["row_index", "label"])
    if merged is None:
        raise ValueError("At least one prediction file is required.")

    metadata = sentence_df.copy()
    metadata["row_index"] = np.arange(len(metadata))
    merged = merged.merge(metadata, on=["row_index", "label"], how="left")
    probability_columns = [column for column in merged if column.endswith("_probability")]
    prediction_columns = [column for column in merged if column.endswith("_prediction")]
    merged["confidence"] = np.max(
        np.abs(merged[probability_columns].to_numpy(dtype=float) - 0.5), axis=1
    )
    merged["mean_probability"] = merged[probability_columns].mean(axis=1)

    categories: List[pd.DataFrame] = []

    def add_category(name: str, mask: pd.Series, ascending: bool = False) -> None:
        selected = merged.loc[mask].sort_values("confidence", ascending=ascending).head(per_category).copy()
        selected.insert(0, "audit_category", name)
        categories.append(selected)

    if len(prediction_columns) >= 1:
        primary = prediction_columns[0]
        add_category("high_confidence_true_positive", (merged.label == 1) & (merged[primary] == 1))
        add_category("high_confidence_false_positive", (merged.label == 0) & (merged[primary] == 1))
        add_category("high_confidence_false_negative", (merged.label == 1) & (merged[primary] == 0))
    add_category("low_confidence_boundary", pd.Series(True, index=merged.index), ascending=True)
    if len(prediction_columns) >= 2:
        for left_index, first in enumerate(prediction_columns):
            for second in prediction_columns[left_index + 1:]:
                add_category(
                    f"{first.removesuffix('_prediction')}_correct_{second.removesuffix('_prediction')}_wrong",
                    (merged[first] == merged.label) & (merged[second] != merged.label),
                )
                add_category(
                    f"{second.removesuffix('_prediction')}_correct_{first.removesuffix('_prediction')}_wrong",
                    (merged[second] == merged.label) & (merged[first] != merged.label),
                )
    if len(prediction_columns) >= 3:
        add_category("models_disagree", merged[prediction_columns].nunique(axis=1) > 1)
    add_category(
        "all_models_wrong",
        (merged[prediction_columns].to_numpy() != merged[["label"]].to_numpy()).all(axis=1),
    )
    output = pd.concat(categories, ignore_index=True)
    ensure_dir(output_path.parent)
    output.to_csv(output_path, index=False)
    return output


def compare_prediction_files(
    prediction_a: Path,
    prediction_b: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare paired saved predictions with bootstrap and McNemar tests."""
    a = pd.read_csv(prediction_a)
    b = pd.read_csv(prediction_b)
    paired = a[["row_index", "label", "probability", "prediction"]].merge(
        b[["row_index", "label", "probability", "prediction"]],
        on=["row_index", "label"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    y_true = paired["label"].to_numpy(dtype=int)
    return {
        "rows": int(len(paired)),
        "paired_bootstrap_pr_auc": bootstrap_metric_difference(
            y_true,
            paired["probability_a"].to_numpy(),
            paired["probability_b"].to_numpy(),
            metric="pr_auc",
            seed=seed,
        ),
        "paired_bootstrap_roc_auc": bootstrap_metric_difference(
            y_true,
            paired["probability_a"].to_numpy(),
            paired["probability_b"].to_numpy(),
            metric="roc_auc",
            seed=seed,
        ),
        "mcnemar_exact": mcnemar_exact(
            y_true,
            paired["prediction_a"].to_numpy(),
            paired["prediction_b"].to_numpy(),
        ),
    }
