"""Compatibility imports for the renamed training module."""
from src.training import train_albert, train_classical, train_model, tune_classical_model

__all__ = ["train_albert", "train_classical", "train_model", "tune_classical_model"]
