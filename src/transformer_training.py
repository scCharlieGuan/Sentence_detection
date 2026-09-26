"""ALBERT fine-tuning, checkpoint selection, and evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.dataset import TransformerTextDataset
from src.evaluate import (
    bootstrap_metric_intervals,
    build_prediction_frame,
    classification_metrics,
    find_best_threshold,
)
from src.input_features import build_model_inputs, token_length_summary
from src.models import build_albert_components
from src.utils import ensure_dir, save_json, select_device


def _freeze_albert(model: torch.nn.Module, strategy: str, last_groups: int) -> None:
    """Apply a declared ALBERT freezing strategy."""
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
    """Create separate encoder and classifier learning-rate groups."""
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
    """Compute exactly one declared imbalance-aware loss."""
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
    """Return loss and positive probabilities without updating state."""
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


def _make_loader(
    indices: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    tokenizer: Any,
    max_length: int,
    training_cfg: Mapping[str, Any],
    collator: DataCollatorWithPadding,
    generator: torch.Generator,
    sample_class_weights: np.ndarray,
    train: bool,
    imbalance_method: str,
) -> DataLoader:
    """Build one deterministic dynamically padded data loader."""
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
        sample_weights = sample_class_weights[labels[indices]]
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


def train_albert(
    sentence_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_dir: Path,
    report_dir: Path,
) -> Tuple[Any, float]:
    """Fine-tune ALBERT with dynamic padding and validation checkpointing."""
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
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "texts": texts,
        "labels": labels,
        "tokenizer": tokenizer,
        "max_length": max_length,
        "training_cfg": training_cfg,
        "collator": collator,
        "generator": generator,
        "sample_class_weights": weights,
        "imbalance_method": imbalance_method,
    }
    train_loader = _make_loader(splits["train"], train=True, **loader_kwargs)
    val_loader = _make_loader(splits["val"], train=False, **loader_kwargs)
    test_loader = _make_loader(splits["test"], train=False, **loader_kwargs)

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
    history: list[dict[str, float | int]] = []
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
    save_json(
        bootstrap_metric_intervals(
            labels[test_indices], test_probability, threshold, seed=seed
        ),
        report_dir / "test_bootstrap_ci.json",
    )
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
        "dataset_sha256": config.get("_protocol_manifest", {}).get("dataset_sha256"),
        "split_strategy": config.get("_protocol_manifest", {}).get("split_strategy"),
        "actual_split_group_column": config.get("_protocol_manifest", {}).get(
            "actual_split_group_column"
        ),
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
