"""Construct leakage-safe sentence and context inputs."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def build_model_inputs(
    sentence_df: pd.DataFrame,
    input_mode: str,
    markers: Dict[str, str] | None = None,
) -> List[str]:
    """Build one text input per target sentence.

    Context is looked up only inside the same response. Explicit target markers
    prevent a context model from silently turning into response-level
    classification.

    Args:
        sentence_df: Sentence data containing response and sentence identifiers.
        input_mode: One of ``sentence``, ``previous_target``, ``target_next``,
            ``window``, ``response``, ``question_target``, or
            ``question_window``.
        markers: Optional marker overrides.

    Returns:
        Texts in the same row order as ``sentence_df``.
    """
    supported = {
        "sentence",
        "previous_target",
        "target_next",
        "window",
        "response",
        "question_target",
        "question_window",
    }
    if input_mode not in supported:
        raise ValueError(f"Unsupported input_mode '{input_mode}'. Supported: {sorted(supported)}")
    marker_values: Dict[str, str] = {
        "previous": "[PREVIOUS]",
        "target": "[TARGET]",
        "next": "[NEXT]",
        "question": "[QUESTION]",
        "response": "[RESPONSE]",
    }
    marker_values.update(markers or {})

    by_response: Dict[Any, Dict[int, str]] = {}
    for row in sentence_df.itertuples():
        by_response.setdefault(row.response_id, {})[int(row.sentence_id)] = str(row.sentence)

    outputs: List[str] = []
    for row in sentence_df.itertuples():
        target = str(row.sentence)
        sentence_map = by_response[row.response_id]
        previous = sentence_map.get(int(row.sentence_id) - 1, "")
        following = sentence_map.get(int(row.sentence_id) + 1, "")
        question = str(getattr(row, "questionText", "") or "")
        response = str(getattr(row, "response", "") or "")

        target_text = f"{marker_values['target']} {target}"
        previous_text = f"{marker_values['previous']} {previous}" if previous else ""
        next_text = f"{marker_values['next']} {following}" if following else ""
        window = " ".join(part for part in (previous_text, target_text, next_text) if part)

        if input_mode == "sentence":
            text = target
        elif input_mode == "previous_target":
            text = " ".join(part for part in (previous_text, target_text) if part)
        elif input_mode == "target_next":
            text = " ".join(part for part in (target_text, next_text) if part)
        elif input_mode == "window":
            text = window
        elif input_mode == "response":
            # Keep the target first so right-side truncation cannot remove the
            # very sentence being classified.
            text = f"{target_text} {marker_values['response']} {response}"
        elif input_mode == "question_target":
            text = f"{marker_values['question']} {question} {target_text}"
        else:
            text = f"{marker_values['question']} {question} {window}"
        outputs.append(text.strip())
    return outputs


def token_length_summary(
    texts: List[str],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, float]:
    """Summarize unpadded token lengths and truncation risk.

    Args:
        texts: Model inputs.
        tokenizer: Matching Hugging Face tokenizer.
        max_length: Configured truncation length.

    Returns:
        Count, quantiles, maximum, and truncation proportion.
    """
    lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]
    series = pd.Series(lengths, dtype=float)
    return {
        "count": int(len(series)),
        "median": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "max_length": int(max_length),
        "truncated_count": int((series > max_length).sum()),
        "truncated_ratio": float((series > max_length).mean()),
    }
