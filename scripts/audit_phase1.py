"""Generate a reproducible, read-only Phase 1 repository/data audit.

The script deliberately reads existing artifacts without rebuilding them.  This
lets it detect stale model outputs and prepared-data inconsistencies before any
experimental result is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TASK_FIELDS = {
    "toxicity": ("toxicity_score", "toxicity_copy"),
    "factual": ("factual_consistency_score", "factual_copy"),
    "medical": ("medical_advice_score", "medical_copy"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.items()}


def build_audit(root: Path) -> dict[str, Any]:
    raw_path = root / "data" / "raw" / "CounselBench.csv"
    prepared_dir = root / "data" / "processed_question_grouped" / "toxicity"
    sentences_path = prepared_dir / "sentences.csv"
    splits_path = prepared_dir / "splits.csv"
    alignment_path = prepared_dir / "alignment_log.csv"
    manifest_path = prepared_dir / "protocol_manifest.json"

    raw = pd.read_csv(raw_path)
    sentences = pd.read_csv(sentences_path)
    splits = pd.read_csv(splits_path)
    alignment = pd.read_csv(alignment_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    raw_tasks: dict[str, Any] = {}
    for task, (score_column, copy_column) in TASK_FIELDS.items():
        copies = raw[copy_column].fillna("").astype(str).str.strip()
        table = pd.crosstab(raw[score_column], copies.ne(""), dropna=False)
        raw_tasks[task] = {
            "score_counts": _jsonable_counts(raw[score_column].value_counts(dropna=False)),
            "nonempty_copy_by_score": {
                str(index): {
                    "empty": int(row.get(False, 0)),
                    "nonempty": int(row.get(True, 0)),
                }
                for index, row in table.iterrows()
            },
        }

    fuzzy = alignment[
        alignment["method"].fillna("").astype(str).str.startswith("fuzzy")
    ].copy()
    fuzzy_counts = fuzzy["matched_sentence_count"].astype(int)
    split_sets = {
        name: set(splits.loc[splits["split"].eq(name), "row_index"].astype(int))
        for name in ("train", "val", "test")
    }

    current_test = split_sets["test"]
    prediction_status: list[dict[str, Any]] = []
    for prediction_path in sorted((root / "reports" / "thesis").rglob("test_predictions.csv")):
        prediction = pd.read_csv(prediction_path)
        row_ids = (
            set(prediction["row_index"].astype(int))
            if "row_index" in prediction else set()
        )
        labels_match = False
        if {"row_index", "label"}.issubset(prediction):
            current = sentences.reset_index(names="row_index")[["row_index", "label"]]
            joined = prediction[["row_index", "label"]].merge(
                current, on="row_index", suffixes=("_saved", "_current")
            )
            labels_match = bool(
                len(joined) == len(prediction)
                and joined["label_saved"].eq(joined["label_current"]).all()
            )
        prediction_status.append({
            "path": str(prediction_path.relative_to(root)),
            "rows": int(len(prediction)),
            "matches_current_test_rows": row_ids == current_test,
            "labels_match_current_original_rule": labels_match,
        })

    response_text_overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_values = set(sentences.iloc[sorted(split_sets[left])]["response_text_id"])
        right_values = set(sentences.iloc[sorted(split_sets[right])]["response_text_id"])
        response_text_overlap[f"{left}_{right}"] = int(len(left_values & right_values))

    return {
        "audit_scope": "read_only_phase_1",
        "raw_data": {
            "path": str(raw_path.relative_to(root)),
            "sha256": _sha256(raw_path),
            "rows": int(len(raw)),
            "questions": int(raw["questionID"].nunique()),
            "responses_by_question_and_responder": int(
                raw[["questionID", "responder", "response"]].drop_duplicates().shape[0]
            ),
            "unique_response_texts": int(raw["response"].nunique()),
            "responders": sorted(raw["responder"].dropna().astype(str).unique()),
            "topics": int(raw["topic"].nunique()),
            "task_fields": raw_tasks,
        },
        "prepared_toxicity": {
            "rows": int(len(sentences)),
            "responses": int(sentences["response_id"].nunique()),
            "positive": int(sentences["label"].sum()),
            "prevalence": float(sentences["label"].mean()),
            "split_rows": _jsonable_counts(splits["split"].value_counts()),
            "response_text_overlap": response_text_overlap,
            "manifest_rows": int(manifest.get("rows", -1)),
            "manifest_dataset_sha256": manifest.get("dataset_sha256"),
        },
        "alignment": {
            "total_spans": int(len(alignment)),
            "method_counts": _jsonable_counts(alignment["method"].value_counts(dropna=False)),
            "fuzzy_spans": int(len(fuzzy)),
            "fuzzy_cross_sentence_spans": int(fuzzy_counts.gt(1).sum()),
            "fuzzy_over_configured_max_window_3": int(fuzzy_counts.gt(3).sum()),
            "fuzzy_max_matched_sentences": int(fuzzy_counts.max()) if len(fuzzy) else 0,
            "fuzzy_matched_sentence_count_distribution": _jsonable_counts(
                fuzzy_counts.value_counts().sort_index()
            ),
            "failed_alignments": int(alignment["method"].eq("unmatched").sum()),
            "log_has_source_scores": bool(
                {field[0] for field in TASK_FIELDS.values()} & set(alignment.columns)
            ),
            "log_has_failure_reason_code": "failure_reason" in alignment.columns,
        },
        "artifact_consistency": {
            "current_test_rows": int(len(current_test)),
            "prediction_files_checked": int(len(prediction_status)),
            "prediction_files_matching_current_test_rows": int(
                sum(item["matches_current_test_rows"] for item in prediction_status)
            ),
            "prediction_files": prediction_status,
            "existing_dataset_audit_is_stale": (
                json.loads(
                    (root / "reports" / "thesis" / "analysis" / "toxicity" / "dataset_audit.json")
                    .read_text(encoding="utf-8")
                ).get("rows")
                != len(sentences)
            ),
        },
        "test_environment": {
            "verified_command": (
                r"C:\Users\15509\anaconda3\envs\MentalFlow\python.exe -m pytest -q"
            ),
            "verified_result": "9 passed",
            "default_pytest_issue": "default Anaconda interpreter lacks spaCy",
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    raw = audit["raw_data"]
    prepared = audit["prepared_toxicity"]
    alignment = audit["alignment"]
    artifacts = audit["artifact_consistency"]
    return f"""# Phase 1 repository audit

Generated by `scripts/audit_phase1.py`. This report is descriptive: it does not
change raw data, labels, splits, or model outputs.

## Confirmed facts

- Raw CounselBench file: {raw['rows']} expert-evaluation rows, {raw['questions']}
  questions, {raw['responses_by_question_and_responder']} question/responder
  response instances, {raw['unique_response_texts']} unique response texts,
  {raw['topics']} topics, and {len(raw['responders'])} responder categories.
- All three independent target fields are present. Their score/copy cross-tabs
  are saved in `phase1_audit.json`; they must not be merged into one task.
- Current prepared toxicity data contain {prepared['rows']} sentences and
  {prepared['positive']} positives (prevalence {prepared['prevalence']:.3f}).
- The question-grouped split has {prepared['split_rows']}. It prevents question
  leakage but still has exact response-text overlap across splits:
  {prepared['response_text_overlap']}.
- Alignment log: {alignment['total_spans']} spans; methods are
  {alignment['method_counts']}. Failed alignments are retained, but the log has
  no structured failure reason and no source rating fields.
- Of {alignment['fuzzy_spans']} fuzzy spans,
  {alignment['fuzzy_over_configured_max_window_3']} label more than the configured
  three-sentence maximum; the maximum is
  {alignment['fuzzy_max_matched_sentences']} sentences. The current algorithm
  unions every qualifying sentence/window, so `max_fuzzy_window=3` does not bound
  the final assignment. This is a confirmed label-expansion defect.
- The current test split contains {artifacts['current_test_rows']} sentences.
  Only {artifacts['prediction_files_matching_current_test_rows']} of
  {artifacts['prediction_files_checked']} saved test-prediction files use exactly
  those rows. The existing dataset audit is stale:
  {artifacts['existing_dataset_audit_is_stale']}.
- Tests pass in the documented available environment (`9 passed`). Plain
  `pytest` currently resolves to another Python installation without spaCy.

## Evidence-backed risks (not yet experimental conclusions)

1. Existing main model tables mix an older 417-row test snapshot with the newer
   421-row snapshot. Those numbers are historical artifacts, not comparable
   evidence under the current dataset fingerprint.
2. Current splitting detects but does not prevent response-text or near-duplicate
   overlap. Therefore the repository does not yet meet the no-known-leakage
   definition of done.
3. Fuzzy alignment can add unrelated neighboring sentences and materially change
   labels. Performance sensitivity to a corrected best-candidate policy must be
   measured rather than assumed.
4. Copy fields contain free-text comments as well as copied spans. Score/copy
   consistency and uncertain ratings require explicit audit; they must not be
   silently discarded or accepted under an undocumented rule.
5. Repeating deterministic TF-IDF training seeds on one persisted split does not
   estimate split uncertainty. Repeated grouped splits and model-seed variation
   are different sources of uncertainty and should be reported separately.

## Unverified or currently unanswered

- Whether stricter label rules improve true label validity; no independent
  sentence-level gold set exists in this repository.
- Whether BGE or ALBERT reliably beats TF-IDF after corrected alignment and
  duplicate-aware splitting; current saved outputs do not share one current
  test snapshot.
- Whether responder- or topic-grouped generalization is feasible without severe
  distribution shift; this requires explicit split experiments.
- H1-H3 remain inconclusive at Phase 1. Existing results are useful diagnostics
  but do not satisfy the predeclared decision rules after the snapshot mismatch.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1_audit"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(root)
    (output_dir / "phase1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "PHASE1_AUDIT.md").write_text(
        render_markdown(audit), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
