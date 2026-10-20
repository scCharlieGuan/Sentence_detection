# Reproducing the dissertation experiments

Run commands from the repository root in the environment described in the main [README](../README.md). Commands below use the final protocol by default. They describe how to rerun experiments, not checks that have already been executed during repository cleanup.

## Environment and snapshot identity

The recorded run used Python 3.11.15, pandas 3.0.1, NumPy 2.4.3, scikit-learn 1.9.0, spaCy 3.8.14 and `en_core_web_sm` 3.8.0. The saved ALBERT configuration records Transformers 5.9.0. These are provenance records from the dissertation, not a complete dependency lock.

| Object | SHA-256 |
| --- | --- |
| Raw CSV | `9d2d2201da6dd4de3e0ad986e101de894606984174a40053443f672a3d24a7ee` |
| Sentence dataset | `cbb3cabf8249d351cdd3c86bb867e9e2059320009c36ed666ae505bec6c10aba` |
| Recorded effective configuration | `a0767a9287f1054c8668bfd72f2d8d008381f7b51c9ee45d793c5183364a26dd` |

The dataset fingerprint covers the ordered protocol-critical columns, not raw CSV bytes, all derived labels or split membership. Preserve the manifest and split table together. A spaCy sentencizer fallback is available but does not reproduce the recorded parser-based segmentation.

## Output protection

The commands below target the existing thesis output directory. To run independently, copy `configs/reliability_protocol.yaml` to another file **inside `configs/`**, change its three output roots (`data.processed_dir`, `output.model_dir`, `output.report_dir`), and pass `--config configs/your_run.yaml` to each command. Config paths resolve relative to the parent of the configuration directory.

`--run-name` isolates ordinary training outputs, but does not isolate preparation, E2, E3 or baseline outputs. `--seed` changes model randomness without repartitioning an existing snapshot. E3 uses its own split-seed argument.

## E1: prepare and inspect alignment

```bash
python run_pipeline.py --mode prepare
python run_pipeline.py --mode diagnose
python run_pipeline.py --mode weak-label-audit
```

For an existing snapshot, skip `prepare` to retain the original sentence segmentation and split. Both historical `scripts/analyze_weak_labels.py` and `scripts/analyze_toxic_weak_labels.py` now delegate to the same audit CLI, whose default is the reliability protocol. Select `--config configs/thesis_protocol.yaml` explicitly to inspect older artifacts.

## E2: label-policy sensitivity

```bash
python run_pipeline.py --mode label-sensitivity --sensitivity-models tfidf,albert
```

This compares any-support and consensus-2 on the same split. The third policy, consensus-2-reliable, is included in prevalence/audit summaries, not the completed two-model comparison. The E2 TF–IDF model is fixed and balanced; it differs from E4. Completed sensitivity runs are reused unless `--force` is supplied.

## E3: split sensitivity

```bash
python -m scripts.run_split_sensitivity --config configs/reliability_protocol.yaml --seeds 42,43,44
```

This performs six strategies × three split seeds, fitting and selecting a word TF–IDF classifier for each partition. It is distinct from repeating models on one fixed split.

## E4: principal systems

```bash
python run_pipeline.py --mode baselines --seeds 42
python run_pipeline.py --mode train --model-name sbert_lr --run-name sbert_bge_target
python run_pipeline.py --mode train --model-name sbert_lr --input-mode target_next --run-name sbert_bge_target_next
```

The baseline suite includes the class-prior, word and character systems. E2 already produces the primary any-support weighted-CE ALBERT result consumed by the report builder. For a separate ordinary ALBERT run:

```bash
python run_pipeline.py --mode train --model-name albert --loss weighted_cross_entropy --run-name albert_weighted_ce
```

This separate run is not automatically substituted into the report builder's E2 ALBERT path.

## E5: paired comparison and review

```bash
python run_pipeline.py --mode compare --prediction-a results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv --prediction-b results/reliability_study/analysis/toxicity/baselines/char_seed_42_predictions.csv
python run_pipeline.py --mode audit --prediction tfidf=results/reliability_study/analysis/toxicity/baselines/word_bigram_seed_42_predictions.csv --prediction albert=results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv
```

The first command compares aligned prediction files. The second exports prediction disagreements. The full provenance-aware weak-label queue is produced by `weak-label-audit`. Queue categories can overlap and are not a representative prevalence sample.

## Locate the saved evidence

All paths below are relative to `results/reliability_study/`:

| Path | Evidence |
| --- | --- |
| `data/toxicity/` | Sentence table, alignment log, persisted splits and manifest |
| `weak_label/toxicity/` | Label-policy summaries, alignment diagnostics and audit queue |
| `weak_label/toxicity/runs/` | E2 TF–IDF/ALBERT predictions and metrics |
| `split_sensitivity/` | E3 run-level results and overlap audits |
| `analysis/toxicity/baselines/` | E4 lexical results and predictions |
| `metrics/toxicity/sbert_bge_target/` | Target-only BGE results |
| `metrics/toxicity/sbert_bge_target_next/` | BGE next-sentence results |
| `paper_outputs/` | Consolidated tables and qualitative review exports |
| `models/` | Local trained model artifacts |

`python -m scripts.build_research_report` assembles existing results; it does not train missing models. It currently reads the fixed `results/reliability_study/` layout. Use it only after all required artifacts are present. Read the manuscript for final claims; older development summaries may reflect earlier stages of the analysis.

## Saved-model inference

For a model trained through the ordinary training workflow, repeat its model and run name:

```bash
python run_pipeline.py --mode evaluate --model-name sbert_lr --run-name sbert_bge_target
python run_pipeline.py --mode predict --model-name sbert_lr --run-name sbert_bge_target --text "Your response text here."
```

Inference writes sentence probabilities, labels and offsets. E2 sensitivity artifacts use a different directory layout and are not selected by these ordinary training commands.
