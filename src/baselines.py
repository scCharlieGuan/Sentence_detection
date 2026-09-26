"""Strong, explicit TF-IDF baselines for thesis comparisons."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import (
    aggregate_seed_metrics,
    bootstrap_metric_intervals,
    build_prediction_frame,
    classification_metrics,
    find_best_threshold,
)
from src.input_features import build_model_inputs
from src.utils import ensure_dir, save_json


BASELINE_SPECS: Tuple[Tuple[str, str, str, str | None, float], ...] = (
    ("word_unigram", "word_unigram", "lr", None, 0.95),
    ("word_bigram", "word_bigram", "lr", None, 0.95),
    ("word_bigram_no_frequency_filter", "word_bigram", "lr", None, 1.00),
    ("word_bigram_high_frequency_filter", "word_bigram", "lr", None, 0.80),
    ("char", "char", "lr", None, 0.95),
    ("word_char", "word_char", "lr", None, 0.95),
    ("word_unigram_svm", "word_unigram", "svm", None, 0.95),
    ("word_bigram_svm", "word_bigram", "svm", None, 0.95),
    ("word_bigram_balanced", "word_bigram", "lr", "balanced", 0.95),
    ("word_char_svm_balanced", "word_char", "svm", "balanced", 0.95),
)


def _baseline_pipeline(
    representation: str,
    classifier: str,
    class_weight: str | None,
    seed: int,
    c_value: float,
    max_df: float = 0.95,
) -> Pipeline:
    """Construct one declared lexical baseline without fitting."""
    representations: Dict[str, Any] = {
        "word_unigram": TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), min_df=2,
            max_df=max_df, sublinear_tf=True,
        ),
        "word_bigram": TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_df=max_df, sublinear_tf=True,
        ),
        "char": TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=100_000, sublinear_tf=True,
        ),
    }
    representations["word_char"] = FeatureUnion([
        ("word", representations["word_bigram"]),
        ("char", representations["char"]),
    ])
    if classifier == "lr":
        estimator: Any = LogisticRegression(
            C=c_value, max_iter=3000, class_weight=class_weight,
            solver="liblinear", random_state=seed,
        )
    elif classifier == "svm":
        estimator = CalibratedClassifierCV(
            LinearSVC(
                C=c_value, class_weight=class_weight,
                random_state=seed, max_iter=5000,
            ),
            method="sigmoid",
            cv=3,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    return Pipeline([("features", representations[representation]), ("clf", estimator)])


def _top_linear_features(
    model: Pipeline,
    sentence_df: pd.DataFrame | None = None,
    top_n: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract positive and negative word weights from linear regression."""
    vectorizer = model.named_steps["features"]
    classifier = model.named_steps["clf"]
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(classifier, "coef_"):
        return {}
    names = np.asarray(vectorizer.get_feature_names_out())
    weights = np.asarray(classifier.coef_[0])
    output: Dict[str, List[Dict[str, Any]]] = {}
    for size, label in ((1, "unigram"), (2, "bigram")):
        indices = np.flatnonzero([len(name.split()) == size for name in names])
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
    if sentence_df is not None:
        metadata_terms: set[str] = set()
        for column in ("responder", "topic"):
            if column not in sentence_df:
                continue
            for value in sentence_df[column].dropna().astype(str).str.casefold().unique():
                metadata_terms.update(part for part in value.replace("-", " ").split() if len(part) >= 3)
        candidate_indices = [
            index for index, name in enumerate(names)
            if any(term in str(name).split() for term in metadata_terms)
        ]
        candidate_indices = sorted(candidate_indices, key=lambda index: abs(weights[index]), reverse=True)
        output["potential_responder_or_topic_features"] = [
            {"feature": str(names[index]), "weight": float(weights[index])}
            for index in candidate_indices[:top_n]
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
    """Run the declared TF-IDF suite under one shared data protocol."""
    texts = np.asarray(build_model_inputs(sentence_df, input_mode), dtype=object)
    labels = sentence_df["label"].to_numpy(dtype=int)
    train_idx = np.asarray(splits["train"], dtype=int)
    val_idx = np.asarray(splits["val"], dtype=int)
    test_idx = np.asarray(splits["test"], dtype=int)
    ensure_dir(report_dir)

    rows: List[Dict[str, Any]] = []
    features_written = False
    bootstrap_cache: Dict[str, Dict[str, Any]] = {}
    prior = float(labels[train_idx].mean())
    prior_probability = np.full(len(test_idx), prior, dtype=float)
    prior_metrics = classification_metrics(labels[test_idx], prior_probability, threshold=0.5)
    rows.append({
        "experiment": "class_prior_majority",
        "representation": "none",
        "classifier": "class_prior",
        "class_weight": "none",
        "max_df": None,
        "seed": int(seeds[0]) if seeds else 42,
        "best_c": None,
        **{key: value for key, value in prior_metrics.items() if key != "confusion_matrix"},
    })
    build_prediction_frame(sentence_df, test_idx, prior_probability, 0.5).to_csv(
        report_dir / "class_prior_majority_predictions.csv", index=False
    )
    prior_signature = hashlib.sha256(prior_probability.tobytes()).hexdigest()
    bootstrap_cache[prior_signature] = bootstrap_metric_intervals(
        labels[test_idx], prior_probability, 0.5,
        seed=int(seeds[0]) if seeds else 42,
    )
    save_json({
        "prediction_sha256": prior_signature,
        **bootstrap_cache[prior_signature],
    }, report_dir / "class_prior_majority_bootstrap_ci.json")

    for experiment, representation, classifier, class_weight, max_df in BASELINE_SPECS:
        run_metrics: List[Mapping[str, float]] = []
        for seed in seeds:
            candidates: List[Tuple[float, float, Pipeline, np.ndarray]] = []
            for c_value in (0.01, 0.1, 1.0, 10.0):
                model = _baseline_pipeline(
                    representation, classifier, class_weight, int(seed), c_value, max_df
                )
                model.fit(texts[train_idx], labels[train_idx])
                val_probability = model.predict_proba(texts[val_idx])[:, 1]
                pr_auc = classification_metrics(
                    labels[val_idx], val_probability, 0.5
                )["pr_auc"]
                candidates.append((pr_auc, c_value, model, val_probability))
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
                "max_df": float(max_df),
                "seed": int(seed),
                "best_c": float(best_c),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            })
            build_prediction_frame(
                sentence_df, test_idx, test_probability, threshold
            ).to_csv(report_dir / f"{experiment}_seed_{seed}_predictions.csv", index=False)
            signature = hashlib.sha256(
                np.round(test_probability, decimals=12).tobytes()
                + f"|threshold={threshold:.12g}".encode("ascii")
            ).hexdigest()
            reused = signature in bootstrap_cache
            if not reused:
                bootstrap_cache[signature] = bootstrap_metric_intervals(
                    labels[test_idx], test_probability, threshold, seed=int(seed)
                )
            save_json({
                "prediction_sha256": signature,
                "reused_identical_prediction_interval": reused,
                **bootstrap_cache[signature],
            }, report_dir / f"{experiment}_seed_{seed}_bootstrap_ci.json")
            if (
                not features_written and representation == "word_bigram"
                and classifier == "lr" and class_weight is None
            ):
                save_json({
                    "vocabulary_settings": {
                        "analyzer": "word",
                        "ngram_range": [1, 2],
                        "min_df": 2,
                        "max_df": float(max_df),
                        "sublinear_tf": True,
                    },
                    **_top_linear_features(best_model, sentence_df),
                }, report_dir / "tfidf_top_features.json")
                features_written = True
        save_json(
            aggregate_seed_metrics(run_metrics),
            report_dir / f"{experiment}_seed_summary.json",
        )
    result = pd.DataFrame(rows)
    result.to_csv(report_dir / "baseline_results.csv", index=False)
    save_json({
        "input_mode": input_mode,
        "selection_data": "validation only",
        "threshold_data": "validation only",
        "test_rows": int(len(test_idx)),
        "model_seed_note": (
            "Seeds repeat model fitting on one fixed split; they do not estimate split uncertainty."
        ),
        "specifications": [
            {
                "experiment": name,
                "representation": representation,
                "classifier": classifier,
                "class_weight": class_weight or "none",
                "max_df": max_df,
            }
            for name, representation, classifier, class_weight, max_df in BASELINE_SPECS
        ],
    }, report_dir / "baseline_protocol.json")
    return result
