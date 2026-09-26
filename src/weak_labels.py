"""Explicit weak-label rules and agreement summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import ensure_dir


DEFAULT_LABEL_RULES: dict[str, dict[str, Any]] = {
    "any_support": {"min_annotators": 1, "reliable_only": False},
    "consensus_2": {"min_annotators": 2, "reliable_only": False},
    "consensus_2_reliable": {"min_annotators": 2, "reliable_only": True},
}


def configured_label_rules(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read label rules from configuration with research-safe defaults."""
    configured = config.get("weak_labels", {}).get("rules", DEFAULT_LABEL_RULES)
    return {str(name): dict(definition) for name, definition in configured.items()}


def label_column_name(rule_name: str) -> str:
    """Return the stable dataset column for a named weak-label rule."""
    return f"label_{rule_name}"


def apply_label_rules(
    sentence_df: pd.DataFrame,
    rules: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Append derived labels without overwriting the original label.

    Args:
        sentence_df: Sentence data with annotator support counts.
        rules: Named definitions containing ``min_annotators`` and optionally
            ``reliable_only``.

    Returns:
        A copy containing ``label_original`` and one column per rule.

    Raises:
        ValueError: If support columns or a rule definition are invalid.
    """
    required = {"positive_annotators", "total_annotators"}
    missing = sorted(required - set(sentence_df))
    if missing:
        raise ValueError(f"Sentence data lacks weak-label columns: {missing}")

    output = sentence_df.copy()
    output["label_original"] = output.get("label_original", output["label"]).astype(int)
    if "reliable_positive_annotators" not in output:
        output["reliable_positive_annotators"] = 0
    output["agreement_ratio"] = output.get(
        "agreement_ratio",
        output["positive_annotators"] / output["total_annotators"].clip(lower=1),
    ).astype(float)

    for name, definition in (rules or DEFAULT_LABEL_RULES).items():
        minimum = int(definition.get("min_annotators", 1))
        if minimum < 1:
            raise ValueError(f"Rule '{name}' requires min_annotators >= 1.")
        support_column = (
            "reliable_positive_annotators"
            if bool(definition.get("reliable_only", False))
            else "positive_annotators"
        )
        output[label_column_name(name)] = (
            output[support_column].astype(int) >= minimum
        ).astype(int)
    return output


def support_bucket(count: int) -> str:
    """Map an annotator support count to the thesis reporting buckets."""
    return "3+" if int(count) >= 3 else str(int(count))


def annotator_agreement_summary(
    sentence_df: pd.DataFrame,
    rule_names: Sequence[str],
) -> pd.DataFrame:
    """Summarize sentence counts and positive rates by support bucket.

    Args:
        sentence_df: Data returned by :func:`apply_label_rules`.
        rule_names: Rules whose positive rates should be reported.

    Returns:
        One row for each of ``0``, ``1``, ``2``, and ``3+``.
    """
    frame = sentence_df.copy()
    frame["support_bucket"] = frame["positive_annotators"].map(support_bucket)
    rows: list[dict[str, Any]] = []
    for bucket in ("0", "1", "2", "3+"):
        group = frame[frame["support_bucket"] == bucket]
        row: dict[str, Any] = {
            "support_bucket": bucket,
            "sentence_count": int(len(group)),
            "mean_agreement_ratio": float(group["agreement_ratio"].mean())
            if len(group) else float("nan"),
            "original_positive_count": int(group["label_original"].sum()),
            "original_positive_rate": float(group["label_original"].mean())
            if len(group) else float("nan"),
        }
        for name in rule_names:
            column = label_column_name(name)
            row[f"{name}_positive_count"] = int(group[column].sum())
            row[f"{name}_positive_rate"] = (
                float(group[column].mean()) if len(group) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def label_rule_summary(
    sentence_df: pd.DataFrame,
    splits: Mapping[str, Sequence[int]],
    rules: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Describe dataset size and prevalence for every label rule and split."""
    rows: list[dict[str, Any]] = []
    for name, definition in rules.items():
        column = label_column_name(name)
        for split_name, indices in (("all", np.arange(len(sentence_df))), *splits.items()):
            selected = sentence_df.iloc[np.asarray(indices, dtype=int)]
            positives = int(selected[column].sum())
            rows.append({
                "label_rule": name,
                "split": split_name,
                "min_annotators": int(definition.get("min_annotators", 1)),
                "reliable_only": bool(definition.get("reliable_only", False)),
                "samples": int(len(selected)),
                "positive_samples": positives,
                "positive_rate": float(positives / len(selected)) if len(selected) else float("nan"),
            })
    return pd.DataFrame(rows)


def save_label_datasets(
    sentence_df: pd.DataFrame,
    rules: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Save independent, non-destructive sentence datasets for each rule."""
    paths: dict[str, Path] = {}
    for name in rules:
        rule_dir = ensure_dir(output_dir / "label_rules" / name)
        version = sentence_df.copy()
        version["label"] = version[label_column_name(name)].astype(int)
        path = rule_dir / "sentences.csv"
        version.to_csv(path, index=False)
        paths[name] = path
    return paths
