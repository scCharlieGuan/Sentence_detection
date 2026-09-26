# Implementation and executed runs

## Scope and file changes

- `src/preprocessing.py`: Unicode whitespace normalization; explicit fuzzy
  selection policy; best contiguous candidate; correct fallback candidate;
  failure reasons, score margins, source score/reason traceability; visible
  spaCy fallback warning.
- `src/splitting.py`: sentence, grouped, and text-only near-duplicate strategies;
  duplicate connected components; size/prevalence-aware assignment for uneven
  components.
- `src/data_pipeline.py`: actual split-group persistence plus raw/config/dataset
  hashes, segmentation details, and package versions.
- `src/audit.py`: per-split responder/topic distributions and unknown-category
  diagnostics.
- `src/evaluate.py`: fixed-test bootstrap intervals in addition to paired model
  bootstrap and McNemar.
- `src/transformer_training.py`: ALBERT fixed-test bootstrap intervals plus
  dataset-fingerprint and split-provenance fields in saved metrics.
- `src/baselines.py`: class-prior baseline, high-frequency TF-IDF ablation,
  vocabulary/weight output, bootstrap CIs, and explicit model-seed caveat.
- `src/weak_label_analysis.py`: fuzzy score/failure/source-score summaries and
  exact/fuzzy/support/context/noise review flags.
- `configs/reliability_protocol.yaml`: isolated, corrected, duplicate-aware
  protocol.
- `scripts/audit_phase1.py`, `scripts/run_split_sensitivity.py`, and
  `scripts/build_research_report.py`: reproducible audit and paper-table entry
  points.
- `tests/test_preprocessing.py`, `tests/test_evaluate.py`: regression coverage.

## Executed commands

```powershell
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe scripts/audit_phase1.py
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode prepare
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode diagnose
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode weak-label-audit
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode baselines --seeds 42,43,44
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode train --model-name sbert_lr --run-name sbert_bge_target --input-mode sentence
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode train --model-name sbert_lr --run-name sbert_bge_target_next --input-mode target_next
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode label-sensitivity --sensitivity-models tfidf
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode label-sensitivity --sensitivity-models tfidf,albert
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode compare --prediction-a results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv --prediction-b results/reliability_study/analysis/toxicity/baselines/char_seed_42_predictions.csv
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode compare --prediction-a results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv --prediction-b results/reliability_study/metrics/toxicity/sbert_bge_target/test_predictions.csv
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode compare --prediction-a results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv --prediction-b results/reliability_study/weak_label/toxicity/runs/any_support/tfidf/test_predictions.csv
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe -m scripts.run_split_sensitivity --config configs/reliability_protocol.yaml --seeds 42,43,44
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe -m scripts.build_research_report
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe -m pytest -q
```

The ALBERT sensitivity runs use the corrected dataset fingerprint
`cbb3cabf8249d351cdd3c86bb867e9e2059320009c36ed666ae505bec6c10aba`
and the same 443-row fixed test split as the primary comparison.

## Explicitly unrun

- Target-aware cross-encoder.
- Factual-consistency and medical-advice task rebuilds/experiments.
- `consensus_2_reliable` model training.
- Independent human adjudication of the qualitative review queue.
- More than three split seeds and repeated trainable-encoder seeds.

The incompatible 417-row metric, model, and dependent analysis artifacts were
removed after row-count and path validation. They are not used in the paper
tables.
