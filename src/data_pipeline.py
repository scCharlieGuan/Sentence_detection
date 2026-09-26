"""Prepared-dataset persistence for the command-line workflows."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.config import AppConfig
from src.preprocessing import build_sentence_dataset, load_spacy_model
from src.splitting import (
    ensure_split_group_column,
    split_train_val_test,
    validate_splits,
)
from src.utils import ensure_dir, save_json
from src.weak_labels import apply_label_rules, configured_label_rules


def processed_task_dir(app_config: AppConfig) -> Path:
    """Return the configured processed directory for the active task."""
    return (
        app_config.resolve_path(app_config.section("data")["processed_dir"])
        / app_config.task_name
    )


def _existing_group_assignments(
    directory: Path,
    group_column: str,
) -> dict[str, str] | None:
    """Recover persisted group-to-split assignments when they are unambiguous."""
    sentence_path = directory / "sentences.csv"
    split_path = directory / "splits.csv"
    if not sentence_path.exists() or not split_path.exists():
        return None
    sentences = pd.read_csv(sentence_path)
    splits = pd.read_csv(split_path)
    if group_column not in sentences or len(sentences) != len(splits):
        return None
    splits = _validate_split_table(sentences, splits)
    joined = pd.DataFrame({
        "group": sentences[group_column].astype(str),
        "split": splits["split"].astype(str),
    })
    if joined.groupby("group")["split"].nunique().max() > 1:
        return None
    return joined.drop_duplicates("group").set_index("group")["split"].to_dict()


def _reuse_or_create_splits(
    sentence_df: pd.DataFrame,
    data_cfg: Mapping[str, Any],
    seed: int,
    directory: Path,
) -> tuple[dict[str, np.ndarray], bool]:
    """Preserve declared question partitions when rebuilding weak labels."""
    group_column = ensure_split_group_column(sentence_df, data_cfg)
    if group_column is None:
        return split_train_val_test(sentence_df, data_cfg, seed), False
    assignments = _existing_group_assignments(directory, group_column)
    if assignments is not None:
        groups = sentence_df[group_column].astype(str)
        if set(groups.unique()) == set(assignments):
            splits = {
                name: sentence_df.index[groups.map(assignments).eq(name)].to_numpy(dtype=int)
                for name in ("train", "val", "test")
            }
            validate_splits(sentence_df, splits, group_columns=[group_column])
            return splits, True
    return split_train_val_test(sentence_df, data_cfg, seed), False


def _dataset_fingerprint(sentence_df: pd.DataFrame) -> str:
    """Hash protocol-critical sentence and label fields."""
    columns = [
        column for column in (
            "response_id", "response_text_id", "sentence_id", "questionID",
            "responder", "sentence", "label_original",
        ) if column in sentence_df
    ]
    payload = sentence_df[columns].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_fingerprint(path: Path) -> str:
    """Hash a source file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    """Capture versions that can change preprocessing or splitting."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "scikit-learn", "spacy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def prepare_data(app_config: AppConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build weak labels and persist one deterministic experimental dataset."""
    values = app_config.values
    task, data = values["task"], values["data"]
    required = [task["text_column"], task["target_column"]]
    raw_path = app_config.resolve_path(data["raw_path"])
    frame = load_data(raw_path, required)
    nlp = load_spacy_model(data["spacy_model"])
    sentence_df, alignment_df = build_sentence_dataset(frame, task, data, nlp)
    sentence_df = apply_label_rules(sentence_df, configured_label_rules(values))

    directory = ensure_dir(processed_task_dir(app_config))
    seed = int(values["project"]["seed"])
    splits, reused = _reuse_or_create_splits(sentence_df, data, seed, directory)
    sentence_df.to_csv(directory / "sentences.csv", index=False)
    alignment_df.to_csv(directory / "alignment_log.csv", index=False)

    split_names = np.full(len(sentence_df), "", dtype=object)
    for name, indices in splits.items():
        split_names[np.asarray(indices, dtype=int)] = name
    actual_group_column = ensure_split_group_column(sentence_df, data)
    split_columns = list(dict.fromkeys(
        column for column in (
            "response_id", "response_text_id", actual_group_column,
        ) if column in sentence_df
    ))
    split_frame = sentence_df[split_columns].copy()
    split_frame.insert(0, "row_index", np.arange(len(sentence_df), dtype=int))
    split_frame["label"] = sentence_df["label"].astype(int)
    split_frame["split"] = split_names
    split_frame.to_csv(directory / "splits.csv", index=False)

    diagnostics = validate_splits(
        sentence_df,
        splits,
        group_columns=[
            actual_group_column,
            "response_id",
            "response_text_id",
        ] if actual_group_column is not None else ["response_id", "response_text_id"],
    )
    save_json({
        "task": task["name"],
        "seed": seed,
        "rows": int(len(sentence_df)),
        "responses": int(sentence_df["response_id"].nunique()),
        "unique_response_texts": int(sentence_df["response_text_id"].nunique()),
        "annotation_group_columns": data.get(
            "annotation_group_columns", ["questionID", "responder"]
        ),
        "split_group_column": data.get("split_group_column", "response_id"),
        "split_strategy": data.get("split_strategy", "group"),
        "actual_split_group_column": actual_group_column,
        "reused_existing_group_assignments": reused,
        "dataset_sha256": _dataset_fingerprint(sentence_df),
        "split_file_sha256": _file_fingerprint(directory / "splits.csv"),
        "raw_data_sha256": _file_fingerprint(raw_path),
        "effective_config_sha256": hashlib.sha256(
            json.dumps(values, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "preprocessing": {
            "requested_spacy_model": data.get("spacy_model"),
            "actual_spacy_model": nlp.meta.get("name", "unknown"),
            "actual_spacy_version": nlp.meta.get("version", "unknown"),
            "pipeline_components": list(nlp.pipe_names),
            "fuzzy_threshold": data.get("fuzzy_threshold"),
            "fuzzy_fallback_threshold": data.get("fuzzy_fallback_threshold"),
            "fuzzy_selection_policy": data.get("fuzzy_selection_policy", "best_window"),
            "max_fuzzy_window": data.get("max_fuzzy_window"),
            "min_positive_annotators": data.get("min_positive_annotators"),
        },
        "environment": _package_versions(),
        "split_diagnostics": diagnostics,
        "test_is_model_selection_forbidden": True,
    }, directory / "protocol_manifest.json")
    return sentence_df, splits


def _validate_split_table(
    sentences: pd.DataFrame, assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Align persisted assignments by row identity, never by CSV file order."""
    if not {"row_index", "split"}.issubset(assignments.columns):
        raise ValueError("Prepared splits require row_index and split columns.")
    indices = assignments["row_index"]
    if (
        not pd.api.types.is_integer_dtype(indices)
        or len(assignments) != len(sentences)
        or indices.duplicated().any()
        or set(indices) != set(range(len(sentences)))
    ):
        raise ValueError("Prepared split assignments do not cover every sentence exactly once.")
    if set(assignments["split"]) != {"train", "val", "test"}:
        raise ValueError("Prepared splits must contain non-empty train, val, and test partitions.")
    ordered = assignments.sort_values("row_index").reset_index(drop=True)
    for column in ("response_id", "response_text_id", "questionID", "near_duplicate_group_id", "label"):
        if column in sentences and column in ordered:
            if not sentences[column].astype(str).reset_index(drop=True).equals(ordered[column].astype(str)):
                raise ValueError(f"Prepared split row identities disagree in column '{column}'.")
    return ordered


def load_prepared(app_config: AppConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load prepared sentence rows and validate their persisted split indices."""
    directory = processed_task_dir(app_config)
    sentence_path, split_path = directory / "sentences.csv", directory / "splits.csv"
    if not sentence_path.exists() or not split_path.exists():
        raise FileNotFoundError("Prepared data not found. Run --mode prepare first.")
    sentence_df = pd.read_csv(sentence_path)
    split_df = _validate_split_table(sentence_df, pd.read_csv(split_path))
    manifest_path = directory / "protocol_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        expected = manifest.get("dataset_sha256")
        if expected and _dataset_fingerprint(sentence_df) != expected:
            raise ValueError("Prepared sentence dataset does not match its manifest fingerprint.")
        expected_split = manifest.get("split_file_sha256")
        if expected_split and _file_fingerprint(split_path) != expected_split:
            raise ValueError("Prepared split file does not match its manifest fingerprint.")
        app_config.values["_protocol_manifest"] = manifest
    splits = {
        name: split_df.loc[split_df["split"] == name, "row_index"].to_numpy(dtype=int)
        for name in ("train", "val", "test")
    }
    actual_group_column = ensure_split_group_column(sentence_df, app_config.section("data"))
    diagnostics = validate_splits(
        sentence_df,
        splits,
        group_columns=[actual_group_column] if actual_group_column is not None else [],
    )
    if actual_group_column is not None and any(
        diagnostics["group_overlap"].get(actual_group_column, {}).values()
    ):
        raise ValueError(f"Prepared splits leak the atomic group '{actual_group_column}'.")
    return sentence_df, splits
