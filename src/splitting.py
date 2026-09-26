"""Group-aware, duplicate-aware dataset splitting and leakage checks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


NEAR_DUPLICATE_GROUP_COLUMN = "near_duplicate_group_id"


class _DisjointSet:
    """Small deterministic union-find used for text-only duplicate components."""

    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _normalise_duplicate_text(text: Any) -> str:
    """Normalize text without using labels or split membership."""
    return " ".join(str(text).casefold().split())


def build_near_duplicate_groups(
    sentence_df: pd.DataFrame,
    base_group_column: str,
    similarity_threshold: float = 0.92,
    min_fuzzy_characters: int = 40,
) -> pd.Series:
    """Cluster base groups linked by exact or fuzzy sentence duplicates.

    The graph uses sentence text only. Exact normalized duplicates of any length
    are linked; fuzzy matching is restricted to longer strings to avoid making
    generic fragments such as "Thank you" connect most of the corpus.
    Connected components are atomic split groups.
    """
    if base_group_column not in sentence_df:
        raise ValueError(f"Near-duplicate base group is absent: {base_group_column}")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1].")
    base_groups = sentence_df[base_group_column].astype(str)
    unique_groups = sorted(base_groups.unique())
    disjoint = _DisjointSet(unique_groups)
    normalized = sentence_df["sentence"].fillna("").map(_normalise_duplicate_text)

    exact = pd.DataFrame({"text": normalized, "group": base_groups})
    for _, duplicates in exact[exact["text"].ne("")].groupby("text", sort=False):
        groups = sorted(duplicates["group"].unique())
        for group in groups[1:]:
            disjoint.union(groups[0], group)

    eligible = normalized.str.len().ge(int(min_fuzzy_characters))
    fuzzy_texts = normalized[eligible].reset_index(drop=True)
    fuzzy_groups = base_groups[eligible].reset_index(drop=True)
    if len(fuzzy_texts) >= 2 and fuzzy_texts.nunique() >= 2:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        features = vectorizer.fit_transform(fuzzy_texts)
        neighbours = NearestNeighbors(
            metric="cosine", radius=max(0.0, 1.0 - float(similarity_threshold)),
            algorithm="brute", n_jobs=-1,
        ).fit(features)
        distances, indices = neighbours.radius_neighbors(features, return_distance=True)
        for row_index, (row_distances, row_neighbours) in enumerate(zip(distances, indices)):
            left_group = str(fuzzy_groups.iloc[row_index])
            for distance, neighbour_index in zip(row_distances, row_neighbours):
                if int(neighbour_index) <= row_index:
                    continue
                if 1.0 - float(distance) + 1e-12 < similarity_threshold:
                    continue
                disjoint.union(left_group, str(fuzzy_groups.iloc[int(neighbour_index)]))

    roots = base_groups.map(disjoint.find)
    root_ids = {root: index for index, root in enumerate(sorted(roots.unique()))}
    return roots.map(root_ids).astype(int).rename(NEAR_DUPLICATE_GROUP_COLUMN)


def ensure_split_group_column(
    sentence_df: pd.DataFrame,
    data_cfg: Mapping[str, Any],
) -> str | None:
    """Materialize and return the actual atomic grouping column for a strategy."""
    strategy = str(data_cfg.get("split_strategy", "group"))
    if strategy == "sentence":
        return None
    if strategy == "group":
        group_column = str(data_cfg.get("split_group_column", "response_id"))
    elif strategy == "near_duplicate":
        group_column = NEAR_DUPLICATE_GROUP_COLUMN
        if group_column not in sentence_df:
            base_column = str(
                data_cfg.get(
                    "near_duplicate_base_group_column",
                    data_cfg.get("split_group_column", "response_id"),
                )
            )
            sentence_df[group_column] = build_near_duplicate_groups(
                sentence_df,
                base_group_column=base_column,
                similarity_threshold=float(data_cfg.get("near_duplicate_threshold", 0.92)),
                min_fuzzy_characters=int(data_cfg.get("near_duplicate_min_characters", 40)),
            )
    else:
        raise ValueError("split_strategy must be sentence, group, or near_duplicate.")
    if group_column not in sentence_df:
        raise ValueError(f"Configured split group column is absent: {group_column}")
    return group_column


def _stratification_labels(labels: np.ndarray) -> np.ndarray | None:
    """Return labels only when stratification is statistically possible."""
    counts = pd.Series(labels).value_counts()
    return labels if len(counts) > 1 and int(counts.min()) >= 2 else None


def _size_aware_group_assignment(
    sentence_df: pd.DataFrame,
    group_column: str,
    data_cfg: Mapping[str, Any],
    seed: int,
    attempts: int = 500,
) -> dict[str, np.ndarray]:
    """Assign uneven components while targeting row and positive counts.

    Near-duplicate connected components can be extremely uneven. Randomly
    splitting component IDs can put a giant component in the test set, so this
    bounded seeded search optimizes sentence counts and prevalence explicitly.
    Labels influence assignment balance but never duplicate construction.
    """
    summary = sentence_df.groupby(group_column)["label"].agg(rows="size", positive="sum")
    split_names = ("train", "val", "test")
    fractions = np.asarray([
        float(data_cfg["train_size"]),
        float(data_cfg["validation_size"]),
        float(data_cfg["test_size"]),
    ])
    target_rows = fractions * len(sentence_df)
    target_positive = fractions * float(sentence_df["label"].sum())
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_assignment: dict[Any, str] | None = None

    for _ in range(max(1, attempts)):
        jitter = pd.Series(rng.random(len(summary)), index=summary.index)
        order = (
            summary.assign(_jitter=jitter)
            .sort_values(["rows", "_jitter"], ascending=[False, True])
            .index
            .tolist()
        )
        assigned_rows = np.zeros(3, dtype=float)
        assigned_positive = np.zeros(3, dtype=float)
        assignment: dict[Any, str] = {}
        for group in order:
            row = summary.loc[group]
            candidate_scores = []
            for split_index in range(3):
                rows = assigned_rows.copy()
                positives = assigned_positive.copy()
                rows[split_index] += float(row["rows"])
                positives[split_index] += float(row["positive"])
                score = float(np.square((rows - target_rows) / np.maximum(target_rows, 1)).sum())
                score += 0.25 * float(
                    np.square(
                        (positives - target_positive) / np.maximum(target_positive, 1)
                    ).sum()
                )
                candidate_scores.append(score)
            chosen = int(np.argmin(candidate_scores))
            assignment[group] = split_names[chosen]
            assigned_rows[chosen] += float(row["rows"])
            assigned_positive[chosen] += float(row["positive"])
        if any(name not in assignment.values() for name in split_names):
            continue
        final_score = float(
            np.square((assigned_rows - target_rows) / np.maximum(target_rows, 1)).sum()
            + 0.25 * np.square(
                (assigned_positive - target_positive) / np.maximum(target_positive, 1)
            ).sum()
        )
        if final_score < best_score:
            best_score, best_assignment = final_score, assignment

    if best_assignment is None:
        raise ValueError("Could not assign near-duplicate components to three non-empty splits.")
    group_values = sentence_df[group_column]
    splits = {
        name: np.flatnonzero(group_values.map(best_assignment).eq(name).to_numpy())
        for name in split_names
    }
    validate_splits(sentence_df, splits, group_columns=[group_column])
    return splits


def split_train_val_test(
    sentence_df: pd.DataFrame,
    data_cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    """Split sentence rows through their configured parent groups.

    Args:
        sentence_df: Sentence-level dataset.
        data_cfg: Split fractions and optional ``split_group_column``.
        seed: Random seed.

    Returns:
        Positional row indices for train, validation, and test sets.

    Raises:
        ValueError: If the grouping column is absent or a split is invalid.
    """
    group_column = ensure_split_group_column(sentence_df, data_cfg)

    if group_column is None:
        all_indices = np.arange(len(sentence_df), dtype=int)
        labels = sentence_df["label"].to_numpy(dtype=int)
        holdout_size = float(data_cfg["validation_size"]) + float(data_cfg["test_size"])
        train_indices, holdout_indices = train_test_split(
            all_indices,
            test_size=holdout_size,
            random_state=seed,
            stratify=_stratification_labels(labels),
        )
        test_fraction = float(data_cfg["test_size"]) / holdout_size
        val_indices, test_indices = train_test_split(
            holdout_indices,
            test_size=test_fraction,
            random_state=seed + 1,
            stratify=_stratification_labels(labels[holdout_indices]),
        )
        splits = {
            "train": np.sort(train_indices),
            "val": np.sort(val_indices),
            "test": np.sort(test_indices),
        }
        validate_splits(sentence_df, splits, group_columns=[])
        return splits

    if str(data_cfg.get("split_strategy", "group")) == "near_duplicate":
        return _size_aware_group_assignment(
            sentence_df,
            group_column,
            data_cfg,
            seed,
            attempts=int(data_cfg.get("near_duplicate_assignment_attempts", 500)),
        )

    group_labels = sentence_df.groupby(group_column, as_index=False)["label"].max()
    group_ids = group_labels[group_column].to_numpy()
    labels = group_labels["label"].to_numpy(dtype=int)
    holdout_size = float(data_cfg["validation_size"]) + float(data_cfg["test_size"])
    if not 0.0 < holdout_size < 1.0:
        raise ValueError("validation_size + test_size must be between 0 and 1.")

    if len(group_ids) < 3:
        raise ValueError(f"At least three atomic groups are required; found {len(group_ids)}.")
    try:
        train_ids, holdout_ids = train_test_split(
            group_ids,
            test_size=holdout_size,
            random_state=seed,
            stratify=_stratification_labels(labels),
        )
    except ValueError:
        train_ids, holdout_ids = train_test_split(
            group_ids, test_size=holdout_size, random_state=seed, stratify=None
        )
    holdout = group_labels[group_labels[group_column].isin(holdout_ids)]
    test_fraction = float(data_cfg["test_size"]) / holdout_size
    try:
        val_ids, test_ids = train_test_split(
            holdout[group_column].to_numpy(),
            test_size=test_fraction,
            random_state=seed + 1,
            stratify=_stratification_labels(holdout["label"].to_numpy(dtype=int)),
        )
    except ValueError:
        val_ids, test_ids = train_test_split(
            holdout[group_column].to_numpy(),
            test_size=test_fraction,
            random_state=seed + 1,
            stratify=None,
        )
    splits = {
        name: sentence_df.index[sentence_df[group_column].isin(ids)].to_numpy(dtype=int)
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    validate_splits(sentence_df, splits, group_columns=[group_column])
    return splits


def validate_splits(
    sentence_df: pd.DataFrame,
    splits: Mapping[str, Sequence[int]],
    group_columns: Sequence[str] = ("response_id",),
) -> dict[str, Any]:
    """Validate row coverage and report group leakage.

    Args:
        sentence_df: Full sentence dataset.
        splits: Named positional row indices.
        group_columns: Columns whose values should not cross partitions.

    Returns:
        JSON-serializable coverage and leakage diagnostics.

    Raises:
        ValueError: If rows are missing, duplicated, or out of range.
    """
    required = {"train", "val", "test"}
    if set(splits) != required:
        raise ValueError(f"Splits must be exactly {sorted(required)}.")
    arrays = {name: np.asarray(indices, dtype=int) for name, indices in splits.items()}
    combined = np.concatenate(list(arrays.values()))
    expected = np.arange(len(sentence_df), dtype=int)
    if len(combined) != len(expected) or not np.array_equal(np.sort(combined), expected):
        raise ValueError("Split indices must cover every sentence exactly once.")

    diagnostics: dict[str, Any] = {
        "rows": {name: int(len(indices)) for name, indices in arrays.items()},
        "group_overlap": {},
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for column in group_columns:
        if column not in sentence_df:
            continue
        values = {
            name: set(sentence_df.iloc[indices][column].dropna().astype(str))
            for name, indices in arrays.items()
        }
        diagnostics["group_overlap"][column] = {
            f"{left}_{right}": int(len(values[left] & values[right]))
            for left, right in pairs
        }
    return diagnostics
