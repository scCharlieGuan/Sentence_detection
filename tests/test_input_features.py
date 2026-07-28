"""Tests for target-marked context construction."""
import pandas as pd

from src.input_features import build_model_inputs


def test_window_context_never_crosses_response_boundary() -> None:
    """Previous and next sentences must belong to the target response."""
    frame = pd.DataFrame({
        "response_id": [1, 1, 2],
        "sentence_id": [0, 1, 0],
        "sentence": ["first", "second", "other"],
        "response": ["first second", "first second", "other"],
    })
    contexts = build_model_inputs(frame, "window")
    assert contexts[1] == "[PREVIOUS] first [TARGET] second"
    assert "second" not in contexts[2]
    assert contexts[2] == "[TARGET] other"
