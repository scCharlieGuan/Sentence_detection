"""General utilities shared across the project."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Seed applied to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select CUDA when available, otherwise CPU.

    Returns:
        Selected PyTorch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed.

    Args:
        path: Directory path.

    Returns:
        The same path for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(log_path: Path, level: str = "INFO") -> None:
    """Configure console and file logging.

    Args:
        log_path: Destination log file.
        level: Logging level name.
    """
    ensure_dir(log_path.parent)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def save_json(data: Any, path: Path) -> None:
    """Save JSON-serializable data.

    Args:
        data: Object to serialize.
        path: Destination file.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
