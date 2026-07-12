"""Dataset loading and schema validation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_data(file_path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    """Load CounselBench data and validate required columns.

    Args:
        file_path: Path to the source CSV file.
        required_columns: Columns required by the selected task.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        ValueError: If required columns are absent or the file is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. Place CounselBench.csv at this path "
            "or update data.raw_path in the configuration."
        )
    frame = pd.read_csv(file_path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {file_path}")
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame
