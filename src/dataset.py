"""PyTorch dataset utilities for transformer models."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TransformerTextDataset(Dataset):
    """Tokenized sentence classification dataset."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        """Initialize the dataset.

        Args:
            texts: Sentence texts.
            labels: Binary labels.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Tokenize one example.

        Args:
            index: Example index.

        Returns:
            Model inputs with a labels tensor.
        """
        encoded = self.tokenizer(
            self.texts[index], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item
