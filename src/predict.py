"""Inference for saved sentence-level classifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AlbertForSequenceClassification, AlbertTokenizerFast

from src.input_features import build_model_inputs
from src.evaluate import build_prediction_frame
from src.preprocessing import load_spacy_model, split_sentences
from src.utils import select_device


def predict_response(response: str, config: Dict[str, Any], model_dir: Path, threshold: float) -> pd.DataFrame:
    """Predict problematic sentences in one response.

    Args:
        response: Raw response text.
        config: Full project configuration.
        model_dir: Directory containing the trained model.
        threshold: Positive-class decision threshold.

    Returns:
        Sentence-level predictions and probabilities.
    """
    nlp = load_spacy_model(config["data"]["spacy_model"])
    spans = split_sentences(response, nlp)
    sentences = [item[0] for item in spans]
    if not sentences:
        return pd.DataFrame(columns=["sentence_id", "sentence", "start", "end", "probability", "prediction"])
    inference_frame = pd.DataFrame([
        {
            "response_id": 0,
            "sentence_id": index,
            "sentence": sentence,
            "response": response,
            "questionText": "",
        }
        for index, sentence in enumerate(sentences)
    ])
    texts = build_model_inputs(
        inference_frame,
        str(config["model"].get("input_mode", "sentence")),
    )
    model_name = config["model"]["name"]
    if model_name == "albert":
        tokenizer = AlbertTokenizerFast.from_pretrained(model_dir)
        model = AlbertForSequenceClassification.from_pretrained(model_dir)
        device = select_device()
        model.to(device).eval()
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=int(config["model"]["max_length"]), return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            probability = torch.softmax(model(**encoded).logits, dim=-1)[:, 1].cpu().numpy()
    else:
        model = joblib.load(model_dir / "model.joblib")
        features: Any = texts
        if model_name.startswith("sbert"):
            encoder = SentenceTransformer(config["model"]["sentence_transformer_name"])
            normalization = str(config["model"].get("embedding_normalization", "l2"))
            features = encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalization == "l2",
            )
        probability = model.predict_proba(features)[:, 1]
    return pd.DataFrame([
        {
            "sentence_id": index, "sentence": sentence, "start": start, "end": end,
            "probability": float(prob), "prediction": int(prob >= threshold),
            "task": config["task"]["name"],
        }
        for index, ((sentence, start, end), prob) in enumerate(zip(spans, probability))
    ])


def predict_dataset(
    sentence_df: pd.DataFrame,
    indices: Any,
    config: Dict[str, Any],
    model_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    """Run a saved model on declared prepared rows for paired analysis.

    Args:
        sentence_df: Full prepared dataset.
        indices: Positional rows, normally the shared test split.
        config: Model and input configuration used during training.
        model_dir: Saved model directory.
        threshold: Validation-selected threshold.

    Returns:
        Prediction frame containing stable ``row_index`` values.
    """
    selected_indices = pd.Series(indices, dtype=int).to_numpy()
    all_texts = build_model_inputs(
        sentence_df,
        str(config["model"].get("input_mode", "sentence")),
    )
    texts = [all_texts[index] for index in selected_indices]
    model_name = str(config["model"]["name"])
    if model_name == "albert":
        tokenizer = AlbertTokenizerFast.from_pretrained(model_dir)
        model = AlbertForSequenceClassification.from_pretrained(model_dir)
        device = select_device()
        model.to(device).eval()
        chunks = []
        batch_size = int(config["training"].get("batch_size", 16))
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=int(config["model"].get("max_length", 128)),
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                chunks.append(
                    torch.softmax(model(**encoded).logits, dim=-1)[:, 1].cpu().numpy()
                )
        probability = np.concatenate(chunks) if chunks else np.asarray([])
    else:
        model = joblib.load(model_dir / "model.joblib")
        features: Any = texts
        if model_name.startswith("sbert"):
            encoder_name_path = model_dir / "sentence_transformer_name.txt"
            encoder_name = (
                encoder_name_path.read_text(encoding="utf-8").strip()
                if encoder_name_path.exists()
                else config["model"]["sentence_transformer_name"]
            )
            encoder = SentenceTransformer(encoder_name)
            normalization = str(config["model"].get("embedding_normalization", "l2"))
            features = encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalization == "l2",
            )
        probability = model.predict_proba(features)[:, 1]
    return build_prediction_frame(
        sentence_df, selected_indices, probability, threshold
    )
