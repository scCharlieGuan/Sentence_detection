"""Unified model construction and persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AlbertForSequenceClassification, AlbertTokenizerFast


def build_model(model_name: str, model_cfg: Dict[str, Any], seed: int) -> Any:
    """Build one supported classifier.

    Args:
        model_name: Configured model family.
        model_cfg: Model configuration.
        seed: Random seed.

    Returns:
        Scikit-learn estimator or ALBERT model.

    Raises:
        ValueError: If the model name is unsupported.
    """
    if model_name == "tfidf_lr":
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)),
        ])
    if model_name == "tfidf_linear_svm":
        calibrated = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000),
            cv=3, method="sigmoid",
        )
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("clf", calibrated),
        ])
    if model_name == "sbert_lr":
        return LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
    if model_name == "sbert_lgbm":
        return LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.03, num_leaves=15, class_weight="balanced", random_state=seed, verbosity=-1)
    if model_name == "sbert_extra_trees":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    if model_name == "albert":
        return AlbertForSequenceClassification.from_pretrained(
            model_cfg["pretrained_model_name"], num_labels=int(model_cfg.get("num_labels", 2))
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_albert_components(model_cfg: Dict[str, Any]) -> Tuple[AlbertTokenizerFast, AlbertForSequenceClassification]:
    """Build the configured ALBERT tokenizer and classifier.

    Args:
        model_cfg: Model configuration.

    Returns:
        Tokenizer and model.
    """
    name = model_cfg["pretrained_model_name"]
    tokenizer = AlbertTokenizerFast.from_pretrained(name)
    model = AlbertForSequenceClassification.from_pretrained(name, num_labels=int(model_cfg.get("num_labels", 2)))
    return tokenizer, model


def save_classical_model(model: Any, path: Path) -> None:
    """Persist a scikit-learn-compatible estimator.

    Args:
        model: Fitted estimator.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classical_model(path: Path) -> Any:
    """Load a persisted classical estimator.

    Args:
        path: Joblib model file.

    Returns:
        Loaded estimator.
    """
    return joblib.load(path)
