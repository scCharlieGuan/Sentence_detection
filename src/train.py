"""Training workflows for classical, SBERT, and ALBERT classifiers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.dataset import TransformerTextDataset
from src.evaluate import classification_metrics, find_best_threshold
from src.models import build_albert_components, build_model, save_classical_model
from src.utils import ensure_dir, select_device, save_json


def _is_sbert(model_name: str) -> bool:
    return model_name.startswith("sbert")


def train_classical(sentence_df: pd.DataFrame, splits: Dict[str, np.ndarray], config: Dict[str, Any], model_dir: Path, report_dir: Path) -> Tuple[Any, float]:
    """Train a configured classical or SBERT classifier.

    Args:
        sentence_df: Weakly labelled sentences.
        splits: Group-safe split indices.
        config: Full configuration dictionary.
        model_dir: Task-specific model output directory.
        report_dir: Task-specific report directory.

    Returns:
        Fitted model and validation-selected threshold.
    """
    model_cfg, training_cfg = config["model"], config["training"]
    model_name, seed = model_cfg["name"], int(config["project"]["seed"])
    texts = sentence_df["sentence"].to_numpy()
    labels = sentence_df["label"].to_numpy(dtype=int)
    features: Any = texts
    encoder = None
    if _is_sbert(model_name):
        encoder = SentenceTransformer(model_cfg["sentence_transformer_name"])
        features = encoder.encode(
            texts.tolist(), batch_size=int(model_cfg["sbert_batch_size"]),
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
        )
    model = build_model(model_name, model_cfg, seed)
    model.fit(features[splits["train"]], labels[splits["train"]])
    val_prob = model.predict_proba(features[splits["val"]])[:, 1]
    threshold, _ = find_best_threshold(labels[splits["val"]], val_prob, training_cfg["threshold_grid"])
    train_idx = np.concatenate([splits["train"], splits["val"]]) if training_cfg.get("train_final_on_train_val", True) else splits["train"]
    model = build_model(model_name, model_cfg, seed)
    model.fit(features[train_idx], labels[train_idx])
    test_prob = model.predict_proba(features[splits["test"]])[:, 1]
    metrics = classification_metrics(labels[splits["test"]], test_prob, threshold)
    ensure_dir(model_dir)
    save_classical_model(model, model_dir / "model.joblib")
    save_json({"model_name": model_name, "threshold": threshold, "metrics": metrics}, report_dir / "test_metrics.json")
    if encoder is not None:
        (model_dir / "sentence_transformer_name.txt").write_text(model_cfg["sentence_transformer_name"], encoding="utf-8")
    return model, threshold


def _predict_albert(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            probabilities.extend(torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy())
    return np.asarray(probabilities)


def train_albert(sentence_df: pd.DataFrame, splits: Dict[str, np.ndarray], config: Dict[str, Any], model_dir: Path, report_dir: Path) -> Tuple[Any, float]:
    """Fine-tune ALBERT with validation threshold selection and early stopping.

    Args:
        sentence_df: Weakly labelled sentences.
        splits: Group-safe split indices.
        config: Full configuration dictionary.
        model_dir: Task-specific model directory.
        report_dir: Task-specific report directory.

    Returns:
        Fitted ALBERT model and selected threshold.
    """
    model_cfg, training_cfg = config["model"], config["training"]
    tokenizer, model = build_albert_components(model_cfg)
    device = select_device()
    model.to(device)
    texts, labels = sentence_df["sentence"].tolist(), sentence_df["label"].astype(int).tolist()
    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TransformerTextDataset(
            [texts[i] for i in indices], [labels[i] for i in indices], tokenizer,
            int(model_cfg["max_length"]),
        )
        return DataLoader(dataset, batch_size=int(training_cfg["batch_size"]), shuffle=shuffle)
    train_loader = make_loader(splits["train"], True)
    val_loader = make_loader(splits["val"], False)
    test_loader = make_loader(splits["test"], False)
    optimizer = AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]), weight_decay=float(training_cfg["weight_decay"]))
    total_steps = max(1, len(train_loader) * int(training_cfg["epochs"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    best_state, best_f1, patience_left = None, -1.0, int(training_cfg["patience"])
    threshold = 0.5
    for epoch in range(int(training_cfg["epochs"])):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_prob = _predict_albert(model, val_loader, device)
        threshold, val_f1 = find_best_threshold(np.asarray([labels[i] for i in splits["val"]]), val_prob, training_cfg["threshold_grid"])
        logging.info("ALBERT epoch %d validation F1 %.4f", epoch + 1, val_f1)
        if val_f1 > best_f1:
            best_f1, patience_left = val_f1, int(training_cfg["patience"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    test_prob = _predict_albert(model, test_loader, device)
    test_labels = np.asarray([labels[i] for i in splits["test"]])
    metrics = classification_metrics(test_labels, test_prob, threshold)
    ensure_dir(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_json({"model_name": "albert", "threshold": threshold, "metrics": metrics}, report_dir / "test_metrics.json")
    return model, threshold


def train_model(sentence_df: pd.DataFrame, splits: Dict[str, np.ndarray], config: Dict[str, Any], model_dir: Path, report_dir: Path) -> Tuple[Any, float]:
    """Dispatch to the configured training backend.

    Args:
        sentence_df: Weakly labelled sentences.
        splits: Group-safe split indices.
        config: Full configuration dictionary.
        model_dir: Model output directory.
        report_dir: Report output directory.

    Returns:
        Fitted model and threshold.
    """
    ensure_dir(report_dir)
    if config["model"]["name"] == "albert":
        return train_albert(sentence_df, splits, config, model_dir, report_dir)
    return train_classical(sentence_df, splits, config, model_dir, report_dir)
