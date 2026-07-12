"""Inference for saved sentence-level classifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AlbertForSequenceClassification, AlbertTokenizerFast

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
    texts = [item[0] for item in spans]
    if not texts:
        return pd.DataFrame(columns=["sentence_id", "sentence", "start", "end", "probability", "prediction"])
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
            features = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        probability = model.predict_proba(features)[:, 1]
    return pd.DataFrame([
        {
            "sentence_id": index, "sentence": sentence, "start": start, "end": end,
            "probability": float(prob), "prediction": int(prob >= threshold),
            "task": config["task"]["name"],
        }
        for index, ((sentence, start, end), prob) in enumerate(zip(spans, probability))
    ])
