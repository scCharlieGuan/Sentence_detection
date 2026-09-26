"""Public training dispatcher with model-specific backends."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.classical_training import train_classical, tune_classical_model
from src.transformer_training import train_albert
from src.utils import ensure_dir


def train_model(
    sentence_df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_dir: Path,
    report_dir: Path,
) -> Tuple[Any, float]:
    """Dispatch to the configured training backend."""
    ensure_dir(report_dir)
    if config["model"]["name"] == "albert":
        return train_albert(sentence_df, splits, config, model_dir, report_dir)
    return train_classical(sentence_df, splits, config, model_dir, report_dir)


__all__ = ["train_albert", "train_classical", "train_model", "tune_classical_model"]
