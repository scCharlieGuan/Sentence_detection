"""Leakage-safe training workflows for lexical, SBERT, and ALBERT models."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.dataset import TransformerTextDataset
from src.evaluate import build_prediction_frame, classification_metrics, find_best_threshold
from src.input_features import build_model_inputs, token_length_summary
from src.models import build_albert_components, build_model, save_classical_model
from src.utils import ensure_dir, save_json, select_device


def _is_sbert(model_name: str) -> bool:
    """Return whether a model consumes fixed sentence-transformer embeddings."""
    return model_name.startswith("sbert")


def _prediction_probability(model: Any, features: Any) -> np.ndarray:
    """Return calibrated positive probabilities from a classical estimator."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Configured estimator must expose predict_proba for threshold selection.")
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def _prepare_sbert_features(
    texts: Sequence[str],
    model_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, SentenceTransformer]:
    """Encode fixed SBERT features according to the declared normalization."""
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


def train_classical(
    sentence_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_dir: Path,
    report_dir: Path,
) -> Tuple[Any, float]:
    """Train a TF-IDF or fixed-embedding classifier under one protocol.

    Hyperparameters use only training-fold cross-validation. The decision
    threshold uses validation data. The model is deliberately not refit on
    train+validation because doing so changes its probability scale after the
    threshold has already been selected.
    """
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
    if _is_sbert(model_name) and model_cfg.get("embedding_normalization") == "standardized":
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
        wrapped=_is_sbert(model_name)
        and model_cfg.get("embedding_normalization") == "standardized",
    )

    val_probability = _prediction_probability(tuned_model, features[val_indices])
    threshold_metric = str(training_cfg.get("threshold_metric", "f1"))
    threshold, validation_threshold_score = find_best_threshold(
        labels[val_indices],
        val_probability,
        training_cfg["threshold_grid"],
        metric=threshold_metric,
    )
    if bool(training_cfg.get("train_final_on_train_val", False)):
        raise ValueError(
            "train_final_on_train_val is incompatible with a validation-selected "
            "probability threshold. Keep it false or use nested out-of-fold calibration."
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
    report = {
        "model_name": model_name,
        "input_mode": input_mode,
        "seed": seed,
        "best_params": best_params,
        "hyperparameter_tuning": tuning_summary,
        "threshold": threshold,
        "threshold_metric": threshold_metric,
        "validation_threshold_score": validation_threshold_score,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    save_json(report, report_dir / "test_metrics.json")
    save_json(
        {"model_name": model_name, "best_params": best_params, **tuning_summary},
        report_dir / "hyperparameter_tuning.json",
    )
    if encoder is not None:
        (model_dir / "sentence_transformer_name.txt").write_text(
            str(model_cfg["sentence_transformer_name"]), encoding="utf-8"
        )
        save_json(
            {
                "sentence_transformer_name": model_cfg["sentence_transformer_name"],
                "embedding_normalization": model_cfg.get("embedding_normalization", "l2"),
            },
            model_dir / "embedding_config.json",
        )
    return tuned_model, threshold


def _freeze_albert(model: torch.nn.Module, strategy: str, last_groups: int) -> None:
    """Apply a declared ALBERT freezing strategy.

    ALBERT shares parameters across logical layers. Consequently, unfreezing a
    final logical layer is not equivalent to BERT; ``last_groups`` operates on
    physical layer groups and often equals full-encoder tuning for albert-base.
    """
    if strategy == "full":
        return
    for parameter in model.albert.parameters():
        parameter.requires_grad = False
    if strategy == "classifier":
        return
    if strategy != "last_groups":
        raise ValueError("freeze_strategy must be classifier, last_groups, or full.")
    groups = model.albert.encoder.albert_layer_groups
    for group in list(groups)[-max(1, last_groups):]:
        for parameter in group.parameters():
            parameter.requires_grad = True
    if getattr(model.albert, "pooler", None) is not None:
        for parameter in model.albert.pooler.parameters():
            parameter.requires_grad = True


def _albert_optimizer(
    model: torch.nn.Module,
    training_cfg: Mapping[str, Any],
) -> AdamW:
    """Create encoder/head learning-rate groups."""
    encoder_lr = float(training_cfg.get("encoder_learning_rate", training_cfg["learning_rate"]))
    head_lr = float(training_cfg.get("head_learning_rate", encoder_lr))
    encoder_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("classifier") and parameter.requires_grad
    ]
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("classifier") and parameter.requires_grad
    ]
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": encoder_lr})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_lr})
    return AdamW(groups, weight_decay=float(training_cfg.get("weight_decay", 0.01)))


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    """Compute exactly one imbalance-aware loss."""
    if loss_name in {"cross_entropy", "weighted_cross_entropy"}:
        weights = class_weights if loss_name == "weighted_cross_entropy" else None
        return functional.cross_entropy(logits, labels, weight=weights)
    if loss_name == "focal":
        per_example = functional.cross_entropy(logits, labels, reduction="none")
        probability = torch.exp(-per_example)
        return ((1.0 - probability) ** focal_gamma * per_example).mean()
    raise ValueError("loss must be cross_entropy, weighted_cross_entropy, or focal.")


def _evaluate_albert(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> Tuple[float, np.ndarray]:
    """Return loss and probabilities without updating model state."""
    model.eval()
    total_loss, total_examples = 0.0, 0
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = _classification_loss(
                logits, labels, loss_name, class_weights, focal_gamma
            )
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return (
        total_loss / max(1, total_examples),
        np.concatenate(probabilities) if probabilities else np.asarray([], dtype=float),
    )


def train_albert(
    sentence_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_dir: Path,
    report_dir: Path,
) -> Tuple[Any, float]:
    """Fine-tune ALBERT with dynamic padding and ranking-based checkpointing."""
    model_cfg, training_cfg = config["model"], config["training"]
    seed = int(config["project"]["seed"])
    tokenizer, model = build_albert_components(model_cfg)
    device = select_device()
    model.to(device)

    input_mode = str(model_cfg.get("input_mode", "sentence"))
    texts = build_model_inputs(sentence_df, input_mode)
    labels = sentence_df["label"].to_numpy(dtype=int)
    max_length = int(model_cfg.get("max_length", 128))
    ensure_dir(report_dir)
    save_json(
        token_length_summary(texts, tokenizer, max_length),
        report_dir / "token_length_summary.json",
    )

    freeze_strategy = str(training_cfg.get("freeze_strategy", "full"))
    _freeze_albert(model, freeze_strategy, int(training_cfg.get("unfreeze_last_groups", 1)))
    loss_name = str(training_cfg.get("loss", "cross_entropy"))
    imbalance_method = str(training_cfg.get("imbalance_method", "none"))
    if loss_name == "weighted_cross_entropy" and imbalance_method == "weighted_sampler":
        raise ValueError("Do not combine class-weighted loss with WeightedRandomSampler.")
    if imbalance_method not in {"none", "weighted_sampler"}:
        raise ValueError("imbalance_method must be none or weighted_sampler.")

    train_labels = labels[np.asarray(splits["train"], dtype=int)]
    counts = np.bincount(train_labels, minlength=2)
    weights = len(train_labels) / (2.0 * np.maximum(counts, 1))
    class_weights = (
        torch.tensor(weights, dtype=torch.float32, device=device)
        if loss_name == "weighted_cross_entropy" else None
    )
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    def make_loader(indices: np.ndarray, train: bool) -> DataLoader:
        indices = np.asarray(indices, dtype=int)
        dataset = TransformerTextDataset(
            [texts[index] for index in indices],
            labels[indices].tolist(),
            tokenizer,
            max_length,
        )
        sampler = None
        shuffle = train
        if train and imbalance_method == "weighted_sampler":
            sample_weights = weights[labels[indices]]
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=int(training_cfg.get("batch_size", 16)),
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collator,
            generator=generator,
        )

    train_loader = make_loader(splits["train"], True)
    val_loader = make_loader(splits["val"], False)
    test_loader = make_loader(splits["test"], False)
    optimizer = _albert_optimizer(model, training_cfg)
    epochs = int(training_cfg.get("epochs", 6))
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    update_steps = int(np.ceil(len(train_loader) / accumulation)) * epochs
    warmup_steps = int(float(training_cfg.get("warmup_ratio", 0.1)) * update_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, update_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"
    )
    focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
    selection_metric = str(training_cfg.get("primary_selection_metric", "pr_auc"))
    if selection_metric not in {"pr_auc", "roc_auc", "loss"}:
        raise ValueError("primary_selection_metric must be pr_auc, roc_auc, or loss.")

    best_score = -np.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    history = []
    patience = int(training_cfg.get("patience", 2))
    patience_left = patience
    clip_norm = float(training_cfg.get("gradient_clip_norm", 1.0))
    val_labels = labels[np.asarray(splits["val"], dtype=int)]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_examples = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_labels = batch.pop("labels")
            with torch.autocast(
                device_type=device.type,
                enabled=scaler.is_enabled(),
                dtype=torch.float16,
            ):
                logits = model(**batch).logits
                loss = _classification_loss(
                    logits, batch_labels, loss_name, class_weights, focal_gamma
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # AMP skips optimizer.step when it detects inf/nan gradients.
                # Advancing the scheduler in that case would silently shorten
                # warmup and produces PyTorch's scheduler-order warning.
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * len(batch_labels)
            total_examples += len(batch_labels)

        val_loss, val_probability = _evaluate_albert(
            model, val_loader, device, loss_name, class_weights, focal_gamma
        )
        val_pr_auc = float(average_precision_score(val_labels, val_probability))
        val_roc_auc = float(roc_auc_score(val_labels, val_probability))
        epoch_score = {
            "pr_auc": val_pr_auc,
            "roc_auc": val_roc_auc,
            "loss": -val_loss,
        }[selection_metric]
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_examples),
            "validation_loss": val_loss,
            "validation_pr_auc": val_pr_auc,
            "validation_roc_auc": val_roc_auc,
        })
        logging.info(
            "ALBERT epoch %d/%d train_loss=%.4f val_loss=%.4f val_pr_auc=%.4f val_roc_auc=%.4f",
            epoch, epochs, history[-1]["train_loss"], val_loss, val_pr_auc, val_roc_auc,
        )
        if epoch_score > best_score:
            best_score = epoch_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_left = patience
        else:
            patience_left -= 1
            if bool(training_cfg.get("use_patience", True)) and patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("ALBERT training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    val_loss, val_probability = _evaluate_albert(
        model, val_loader, device, loss_name, class_weights, focal_gamma
    )
    threshold, threshold_score = find_best_threshold(
        val_labels,
        val_probability,
        training_cfg["threshold_grid"],
        metric=str(training_cfg.get("threshold_metric", "f1")),
    )
    test_loss, test_probability = _evaluate_albert(
        model, test_loader, device, loss_name, class_weights, focal_gamma
    )
    test_indices = np.asarray(splits["test"], dtype=int)
    val_indices = np.asarray(splits["val"], dtype=int)
    validation_metrics = classification_metrics(val_labels, val_probability, threshold)
    test_metrics = classification_metrics(labels[test_indices], test_probability, threshold)
    test_metrics["loss"] = test_loss

    ensure_dir(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    build_prediction_frame(
        sentence_df, val_indices, val_probability, threshold
    ).to_csv(report_dir / "validation_predictions.csv", index=False)
    build_prediction_frame(
        sentence_df, test_indices, test_probability, threshold
    ).to_csv(report_dir / "test_predictions.csv", index=False)
    save_json(history, report_dir / "training_history.json")
    save_json({
        "model_name": "albert",
        "input_mode": input_mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_validation_selection_score": best_score,
        "threshold": threshold,
        "threshold_metric": training_cfg.get("threshold_metric", "f1"),
        "validation_threshold_score": threshold_score,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_configuration": {
            "loss": loss_name,
            "imbalance_method": imbalance_method,
            "freeze_strategy": freeze_strategy,
            "encoder_learning_rate": training_cfg.get(
                "encoder_learning_rate", training_cfg["learning_rate"]
            ),
            "head_learning_rate": training_cfg.get(
                "head_learning_rate", training_cfg["learning_rate"]
            ),
        },
    }, report_dir / "test_metrics.json")
    return model, threshold


def train_model(
    sentence_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_dir: Path,
    report_dir: Path,
) -> Tuple[Any, float]:
    """Dispatch to the configured backend without changing public API."""
    ensure_dir(report_dir)
    if config["model"]["name"] == "albert":
        return train_albert(sentence_df, splits, config, model_dir, report_dir)
    return train_classical(sentence_df, splits, config, model_dir, report_dir)


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
    """Tune an estimator using training-only (group-aware) cross-validation."""
    tuning_cfg: Dict[str, Any] = training_cfg.get("hyperparameter_tuning", {})
    if not tuning_cfg.get("enabled", False):
        model.fit(train_features, train_labels)
        return model, {}, {"enabled": False}

    raw_grid = tuning_cfg.get("models", {}).get(model_name, {}).get("params", {})
    param_grid = copy.deepcopy(raw_grid)
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
    cv: Any
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
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
    fit_kwargs = {"groups": groups} if groups is not None else {}
    search.fit(train_features, train_labels, **fit_kwargs)
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
