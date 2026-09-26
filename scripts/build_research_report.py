"""Assemble traceable thesis tables and cautious hypothesis decisions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reliability_study"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(name: str, payload: dict[str, Any], source: Path) -> dict[str, Any]:
    metrics = payload.get("test_metrics", payload)
    return {
        "model": name,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "macro_f1": metrics.get("macro_f1"),
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "accuracy_auxiliary": metrics.get("accuracy"),
        "positive_prevalence": metrics.get("prevalence"),
        "threshold": metrics.get("threshold"),
        "confusion_matrix": json.dumps(metrics.get("confusion_matrix")),
        "source_file": str(source.relative_to(ROOT)),
    }


def main() -> None:
    output_dir = RESULTS / "paper_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = RESULTS / "analysis" / "toxicity" / "baselines" / "baseline_results.csv"
    baselines = pd.read_csv(baseline_path)
    seed42 = baselines[baselines["seed"].eq(42)].drop_duplicates("experiment")

    main_rows: list[dict[str, Any]] = []
    for experiment, display in (
        ("class_prior_majority", "Class-prior / majority"),
        ("word_bigram", "TF-IDF word (1,2)-gram LR"),
        ("char", "TF-IDF char (3,5)-gram LR"),
    ):
        row = seed42.loc[seed42["experiment"].eq(experiment)].iloc[0]
        main_rows.append({
            "model": display,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "macro_f1": row.macro_f1,
            "pr_auc": row.pr_auc,
            "roc_auc": row.roc_auc,
            "accuracy_auxiliary": row.accuracy,
            "positive_prevalence": row.prevalence,
            "threshold": row.threshold,
            "confusion_matrix": "see prediction/CI artifact",
            "source_file": str(baseline_path.relative_to(ROOT)),
        })
    for directory, name in (
        ("sbert_bge_target", "BGE embedding + LR (target)"),
        ("sbert_bge_target_next", "BGE embedding + LR (target+next)"),
    ):
        path = RESULTS / "metrics" / "toxicity" / directory / "test_metrics.json"
        main_rows.append(_metric_row(name, _read_json(path), path))
    albert_path = (
        RESULTS / "weak_label" / "toxicity" / "runs" / "any_support"
        / "albert" / "test_metrics.json"
    )
    main_rows.append(_metric_row(
        "ALBERT (weighted CE, target)", _read_json(albert_path), albert_path
    ))
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(output_dir / "main_experiment_results.csv", index=False)

    frequency = seed42[
        seed42["experiment"].isin([
            "word_bigram_no_frequency_filter",
            "word_bigram",
            "word_bigram_high_frequency_filter",
        ])
    ][["experiment", "max_df", "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc"]]
    frequency.insert(0, "ablation_family", "tfidf_high_frequency_filter")
    context = main_table[main_table["model"].str.startswith("BGE")][
        ["model", "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc"]
    ].rename(columns={"model": "experiment"})
    context.insert(0, "max_df", pd.NA)
    context.insert(0, "ablation_family", "context_input")
    label_rules_path = RESULTS / "weak_label" / "toxicity" / "model_by_label_rule.csv"
    label_rules = pd.read_csv(label_rules_path)
    label_ablation = label_rules[
        ["label_rule", "model", "positive_rate", "test_positive_rate", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    ].rename(columns={"label_rule": "experiment"})
    label_ablation.insert(0, "max_df", pd.NA)
    label_ablation.insert(0, "ablation_family", "annotator_threshold")
    ablation = pd.concat([frequency, context, label_ablation], ignore_index=True, sort=False)
    ablation.to_csv(output_dir / "ablation_study.csv", index=False)

    split_path = RESULTS / "split_sensitivity" / "split_sensitivity_results.csv"
    split_results = pd.read_csv(split_path)
    split_results.to_csv(output_dir / "split_sensitivity_results.csv", index=False)
    split_summary = split_results.groupby("split_strategy").agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        pr_auc_min=("pr_auc", "min"),
        pr_auc_max=("pr_auc", "max"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        test_prevalence_mean=("test_prevalence", "mean"),
    ).reset_index()
    split_summary.to_csv(output_dir / "split_sensitivity_summary.csv", index=False)

    audit_path = RESULTS / "weak_label" / "toxicity" / "weak_label_audit.csv"
    audit = pd.read_csv(audit_path)
    sentence_rows = audit[audit["row_index"].notna()].drop_duplicates("row_index").copy()
    categories: list[pd.DataFrame] = []
    masks = {
        "tfidf_false_positive": sentence_rows["label"].eq(0) & sentence_rows["tfidf_prediction"].eq(1),
        "tfidf_false_negative": sentence_rows["label"].eq(1) & sentence_rows["tfidf_prediction"].eq(0),
        "fuzzy_aligned_positive": sentence_rows["flag_fuzzy_aligned_positive"].fillna(False),
        "single_annotator_positive": sentence_rows["flag_single_annotator_positive"].fillna(False),
        "multi_annotator_positive": sentence_rows["flag_multi_annotator_positive"].fillna(False),
        "context_dependent_proxy": sentence_rows["flag_context_dependent_proxy"].fillna(False),
        "potentially_noisy_label": sentence_rows["flag_potentially_noisy_label"].fillna(False),
    }
    for category, mask in masks.items():
        selected = sentence_rows.loc[mask].head(8).copy()
        selected.insert(0, "error_analysis_category", category)
        categories.append(selected)
    qualitative = pd.concat(categories, ignore_index=True, sort=False)
    qualitative["review_status"] = (
        "Requires manual review of span, context, category and annotator support; "
        "model disagreement alone is not evidence that the weak label is wrong."
    )
    preferred = [
        "error_analysis_category", "review_status", "questionID", "response_id",
        "responder", "topic", "sentence", "original_span", "alignment_method",
        "method", "positive_annotators", "total_annotators", "agreement_ratio",
        "label", "tfidf_probability", "tfidf_prediction", "bge_probability",
        "bge_prediction", "bge_context_probability", "bge_context_prediction",
        "audit_flags",
    ]
    qualitative[[column for column in preferred if column in qualitative]].to_csv(
        output_dir / "qualitative_error_analysis.csv", index=False, encoding="utf-8-sig"
    )

    comparison_bge = _read_json(
        RESULTS / "analysis" / "toxicity" / "model_comparison_sbert_bge_target_vs_baselines.json"
    )
    comparison_context = _read_json(
        RESULTS / "analysis" / "toxicity" / "model_comparison_sbert_bge_target_next_vs_sbert_bge_target.json"
    )
    comparison_albert_char = _read_json(
        RESULTS / "analysis" / "toxicity" / "model_comparison_albert_vs_baselines.json"
    )
    comparison_albert_bge = _read_json(
        RESULTS / "analysis" / "toxicity" / "model_comparison_albert_vs_sbert_bge_target.json"
    )
    paired = pd.DataFrame([
        {
            "comparison": "BGE target minus char TF-IDF",
            "metric": metric,
            **comparison_bge[f"paired_bootstrap_{metric}"],
        }
        for metric in ("pr_auc", "roc_auc")
    ] + [
        {
            "comparison": "BGE target+next minus BGE target",
            "metric": metric,
            **comparison_context[f"paired_bootstrap_{metric}"],
        }
        for metric in ("pr_auc", "roc_auc")
    ] + [
        {
            "comparison": "ALBERT minus char TF-IDF",
            "metric": metric,
            **comparison_albert_char[f"paired_bootstrap_{metric}"],
        }
        for metric in ("pr_auc", "roc_auc")
    ] + [
        {
            "comparison": "ALBERT minus BGE target",
            "metric": metric,
            **comparison_albert_bge[f"paired_bootstrap_{metric}"],
        }
        for metric in ("pr_auc", "roc_auc")
    ])
    paired.to_csv(output_dir / "paired_model_comparisons.csv", index=False)

    old = pd.read_csv(ROOT / "data" / "processed_question_grouped" / "toxicity" / "sentences.csv")
    new = pd.read_csv(RESULTS / "data" / "toxicity" / "sentences.csv")
    naive = split_results[split_results["split_strategy"].eq("naive_sentence")].set_index("split_seed")
    strict = split_results[
        split_results["split_strategy"].eq("near_duplicate_question_grouped")
    ].set_index("split_seed")
    split_differences = (naive["pr_auc"] - strict["pr_auc"]).to_dict()
    evidence = pd.DataFrame([
        {
            "hypothesis": "H1",
            "decision": "Partially supported",
            "supporting_evidence": (
                f"Legacy-union snapshot has {int(old.label.sum())}/{len(old)} positives; corrected "
                f"best-window snapshot has {int(new.label.sum())}/{len(new)}. Naive-minus-strict "
                f"TF-IDF PR-AUC differences by split seed: {split_differences}."
            ),
            "counterevidence_or_limitations": (
                "Only three split seeds; legacy and corrected snapshots also differ slightly in "
                "Unicode whitespace segmentation. Responder/topic splits change the target domain."
            ),
            "paper_conclusion": (
                "Observed performance is sensitive to label construction and split strategy; "
                "the direction is consistent here, but the magnitude is protocol- and sample-dependent."
            ),
        },
        {
            "hypothesis": "H2",
            "decision": "Inconclusive",
            "supporting_evidence": (
                "BGE-target and ALBERT exceed char TF-IDF PR-AUC by 0.026 and 0.028, "
                "respectively, on the strict test set."
            ),
            "counterevidence_or_limitations": (
                f"BGE-minus-char 95% CI is [{comparison_bge['paired_bootstrap_pr_auc']['ci_low']:.3f}, "
                f"{comparison_bge['paired_bootstrap_pr_auc']['ci_high']:.3f}], and ALBERT-minus-char "
                f"95% CI is [{comparison_albert_char['paired_bootstrap_pr_auc']['ci_low']:.3f}, "
                f"{comparison_albert_char['paired_bootstrap_pr_auc']['ci_high']:.3f}]; both cover zero. "
                "ALBERT and BGE are nearly tied, target+next does not improve PR-AUC, and each encoder ran once."
            ),
            "paper_conclusion": (
                "The current experiment does not establish a reliable semantic/contextual advantage over TF-IDF."
            ),
        },
        {
            "hypothesis": "H3",
            "decision": "Partially supported",
            "supporting_evidence": (
                "The mean naive-to-strict TF-IDF PR-AUC change is larger than the observed "
                "BGE-minus-char-TF-IDF point gain, and the latter CI covers zero. Annotator "
                "threshold changes prevalence from 0.164 to 0.043."
            ),
            "counterevidence_or_limitations": (
                "This is not a causal experiment, covers toxicity only, and each encoder was run with one seed."
            ),
            "paper_conclusion": (
                "Evidence suggests that supervision quality and dataset structure may be a larger "
                "bottleneck than the tested increase in representation capacity, but causality is not established."
            ),
        },
    ])
    evidence.to_csv(output_dir / "hypothesis_evidence_matrix.csv", index=False)

    manifest = _read_json(RESULTS / "data" / "toxicity" / "protocol_manifest.json")
    alignment = _read_json(RESULTS / "weak_label" / "toxicity" / "alignment_summary.json")
    report = f"""# Reliability study report (executed subset)

## Observed results

- Dataset fingerprint: `{manifest['dataset_sha256']}`; raw-data fingerprint:
  `{manifest['raw_data_sha256']}`.
- Corrected best-window labels: {len(new)} sentences, {int(new.label.sum())}
  positives ({new.label.mean():.3f}). Alignment: {alignment['total_spans']} spans,
  {alignment['unmatched_spans']} unmatched, and {alignment['fuzzy_spans']} fuzzy.
- Strict test set: 443 sentences, prevalence 0.163, with zero response-text,
  normalized-exact, or >=0.90 near-duplicate overlap in the generated diagnostic.
- Best lexical ranking result in the declared baseline table is char TF-IDF
  (PR-AUC 0.304). BGE target is 0.330, BGE target+next is 0.322,
  and corrected-label ALBERT is 0.331.
- BGE target minus char TF-IDF PR-AUC =
  {comparison_bge['paired_bootstrap_pr_auc']['difference_a_minus_b']:.3f}, 95% CI
  [{comparison_bge['paired_bootstrap_pr_auc']['ci_low']:.3f},
  {comparison_bge['paired_bootstrap_pr_auc']['ci_high']:.3f}].
- ALBERT minus char TF-IDF PR-AUC =
  {comparison_albert_char['paired_bootstrap_pr_auc']['difference_a_minus_b']:.3f},
  95% CI [{comparison_albert_char['paired_bootstrap_pr_auc']['ci_low']:.3f},
  {comparison_albert_char['paired_bootstrap_pr_auc']['ci_high']:.3f}].
- Under the stricter consensus-2 label rule, ALBERT has PR-AUC 0.139 and
  TF-IDF has 0.136; this is a label-policy sensitivity result, not a direct
  performance comparison with the any-support task because prevalence changes.
- Naive sentence-split TF-IDF mean PR-AUC is
  {naive.pr_auc.mean():.3f}; duplicate-aware question-grouped mean is
  {strict.pr_auc.mean():.3f} across split seeds 42-44.

## Interpretation

The experiment shows material sensitivity to fuzzy-label construction and split
policy. Neither BGE nor ALBERT shows a statistically reliable PR-AUC advantage
over char TF-IDF, ALBERT is effectively tied with BGE, and adding the next
sentence does not improve PR-AUC on this strict split. These observations are
consistent with a supervision/data-structure bottleneck, but do not prove one.

## Limitations and unrun work

- Only toxicity was rebuilt and executed. Factual and medical remain unrun.
- The target-aware cross-encoder remains unrun.
- BGE and ALBERT were each run once; the paired CI quantifies test-sample
  uncertainty, not encoder-training or random-seed uncertainty.
- Split sensitivity uses three split seeds. Responder/topic grouped conditions
  intentionally introduce unseen-domain categories.
- No independent sentence-level gold labels exist. The qualitative table is a
  review queue, not an adjudicated label-error table.

See the CSV/JSON files in this directory for the exact tables and evidence matrix.
"""
    (output_dir / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
