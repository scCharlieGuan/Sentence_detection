"""Training for TF-IDF and fixed sentence-embedding classifiers."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import (
    bootstrap_metric_intervals,
    build_prediction_frame,
    classification_metrics,
    find_best_threshold,
)
from src.input_features import build_model_inputs
from src.models import build_model, save_classical_model
from src.utils import ensure_dir, save_json


def _is_sbert(model_name: str) -> bool:
    """Return whether a model consumes fixed sentence-transformer embeddings."""
    return model_name.startswith("sbert")


def _prediction_probability(model: Any, features: Any) -> np.ndarray:
    """Return calibrated positive probabilities from a classical estimator."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Configured estimator must expose predict_proba.")
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def _prepare_sbert_features(
    texts: Sequence[str],
    model_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, SentenceTransformer]:
    """Encode fixed SBERT features according to declared normalization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model_cfg["sentence_transformer_name"], device=device)
    normalization = str(model_cfg.get("embedding_normalization", "l2"))
    if normalization not in {"raw", "l2", "standardized"}:
        raise ValueError("embedding_normalization must be raw, l2, or standardized.")
    features = encoder.encode(
        list(texts),
        batch_size=int(model_cfg.get("sbert_batch_size", 32)),
        convert_to_numpy=True,
        normalize_embeddings=normalization == "l2",
        show_progress_bar=True,
    )
    return np.asarray(features, dtype=np.float32), encoder


def tune_classical_model(
    model: Any,
    model_name: str,
    train_features: Any,
    train_labels: np.ndarray,
    training_cfg: Dict[str, Any],
    groups: Optional[np.ndarray] = None,
    seed: int = 42,
    wrapped: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Tune an estimator using training-only group-aware cross-validation."""
    tuning_cfg: Dict[str, Any] = training_cfg.get("hyperparameter_tuning", {})
    if not tuning_cfg.get("enabled", False):
        model.fit(train_features, train_labels)
        return model, {}, {"enabled": False}

    param_grid = copy.deepcopy(
        tuning_cfg.get("models", {}).get(model_name, {}).get("params", {})
    )
    for key, value in list(param_grid.items()):
        if key.endswith("ngram_range"):
            param_grid[key] = [tuple(item) for item in value]
    if wrapped:
        param_grid = {
            key if key.startswith("clf__") else f"clf__{key}": value
            for key, value in param_grid.items()
        }
    if not param_grid:
        model.fit(train_features, train_labels)
        return model, {}, {"enabled": False, "reason": "no_parameter_grid"}

    requested_folds = int(tuning_cfg.get("cv_folds", 5))
    minimum_class = int(np.bincount(train_labels).min())
    cv_folds = max(2, min(requested_folds, minimum_class))
    cv: Any = (
        StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    )
    common = {
        "estimator": model,
        "scoring": str(tuning_cfg.get("scoring", "average_precision")),
        "cv": cv,
        "n_jobs": int(tuning_cfg.get("n_jobs", -1)),
        "verbose": int(tuning_cfg.get("verbose", 0)),
        "refit": True,
        "error_score": "raise",
    }
    if model_name in set(tuning_cfg.get("random_search_models", [])):
        model_cfg = tuning_cfg.get("models", {}).get(model_name, {})
        search: Any = RandomizedSearchCV(
            **common,
            param_distributions=param_grid,
            n_iter=int(model_cfg.get("n_iter", tuning_cfg.get("n_iter", 40))),
            random_state=seed,
        )
        method = "random_search"
    else:
        search = GridSearchCV(**common, param_grid=param_grid)
        method = "grid_search"
    search.fit(
        train_features,
        train_labels,
        **({"groups": groups} if groups is not None else {}),
    )
    summary = {
        "enabled": True,
        "search_method": method,
        "scoring": common["scoring"],
        "cv_folds": cv_folds,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "n_candidates": int(len(search.cv_results_["params"])),
    }
    return search.best_estimator_, dict(search.best_params_), summary


def train_classical(
    sentence_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_dir: Path,
    report_dir: Path,
) -> Tuple[Any, float]:
    """Train one TF-IDF or fixed-embedding classifier under one protocol."""
    started = time.perf_counter()
    model_cfg, training_cfg = config["model"], config["training"]
    model_name = str(model_cfg["name"])
    seed = int(config["project"]["seed"])
    input_mode = str(model_cfg.get("input_mode", "sentence"))
    texts = np.asarray(build_model_inputs(sentence_df, input_mode), dtype=object)
    labels = sentence_df["label"].to_numpy(dtype=int)
    features: Any = texts
    encoder: SentenceTransformer | None = None
    if _is_sbert(model_name):
        features, encoder = _prepare_sbert_features(texts.tolist(), model_cfg)

    train_indices = np.asarray(splits["train"], dtype=int)
    val_indices = np.asarray(splits["val"], dtype=int)
    test_indices = np.asarray(splits["test"], dtype=int)
    base_model = build_model(model_name, model_cfg, seed)
    standardized = _is_sbert(model_name) and model_cfg.get("embedding_normalization") == "standardized"
    if standardized:
        base_model = Pipeline([("scaler", StandardScaler()), ("clf", base_model)])

    group_column = str(config["data"].get("split_group_column", "response_id"))
    train_groups = sentence_df.iloc[train_indices][group_column].to_numpy()
    tuned_model, best_params, tuning_summary = tune_classical_model(
        model=base_model,
        model_name=model_name,
        train_features=features[train_indices],
        train_labels=labels[train_indices],
        training_cfg=training_cfg,
        groups=train_groups,
        seed=seed,
        wrapped=standardized,
    )

    val_probability = _prediction_probability(tuned_model, features[val_indices])
    threshold_metric = str(training_cfg.get("threshold_metric", "f1"))
    threshold, validation_score = find_best_threshold(
        labels[val_indices], val_probability, training_cfg["threshold_grid"],
        metric=threshold_metric,
    )
    if bool(training_cfg.get("train_final_on_train_val", False)):
        raise ValueError(
            "train_final_on_train_val changes probability calibration after threshold selection."
        )
    test_probability = _prediction_probability(tuned_model, features[test_indices])
    validation_metrics = classification_metrics(labels[val_indices], val_probability, threshold)
    test_metrics = classification_metrics(labels[test_indices], test_probability, threshold)

    ensure_dir(model_dir)
    ensure_dir(report_dir)
    save_classical_model(tuned_model, model_dir / "model.joblib")
    build_prediction_frame(
        sentence_df, val_indices, val_probability, threshold
    ).to_csv(report_dir / "validation_predictions.csv", index=False)
    build_prediction_frame(
        sentence_df, test_indices, test_probability, threshold
    ).to_csv(report_dir / "test_predictions.csv", index=False)
    save_json(
        bootstrap_metric_intervals(
            labels[test_indices], test_probability, threshold, seed=seed
        ),
        report_dir / "test_bootstrap_ci.json",
    )
    save_json({
        "model_name": model_name,
        "input_mode": input_mode,
        "seed": seed,
        "best_params": best_params,
        "hyperparameter_tuning": tuning_summary,
        "threshold": threshold,
        "threshold_metric": threshold_metric,
        "validation_threshold_score": validation_score,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "runtime_seconds": float(time.perf_counter() - started),
        "dataset_sha256": config.get("_protocol_manifest", {}).get("dataset_sha256"),
        "split_strategy": config.get("_protocol_manifest", {}).get("split_strategy"),
        "actual_split_group_column": config.get("_protocol_manifest", {}).get(
            "actual_split_group_column"
        ),
        "effective_model_config": model_cfg,
        "effective_training_config": training_cfg,
    }, report_dir / "test_metrics.json")
    save_json(
        {"model_name": model_name, "best_params": best_params, **tuning_summary},
        report_dir / "hyperparameter_tuning.json",
    )
    if encoder is not None:
        (model_dir / "sentence_transformer_name.txt").write_text(
            str(model_cfg["sentence_transformer_name"]), encoding="utf-8"
        )
        save_json({
            "sentence_transformer_name": model_cfg["sentence_transformer_name"],
            "embedding_normalization": model_cfg.get("embedding_normalization", "l2"),
        }, model_dir / "embedding_config.json")
    return tuned_model, threshold
