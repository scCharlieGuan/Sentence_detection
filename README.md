# Sentence Error Detection for CounselBench

中文结构说明、论文 E1–E5 与代码对应关系及复现命令见
[docs/ARCHITECTURE_zh.md](docs/ARCHITECTURE_zh.md)。
CLI 默认使用 `configs/reliability_protocol.yaml`；旧配置必须通过 `--config` 显式选择。
`run_pipeline.py` 保留兼容入口，参数解析和流程实现分别位于 `src/cli.py`、`src/workflows.py`。

## Project overview

This project predicts sentence-level weak labels projected from expert spans in mental-health responses. The dissertation evaluates **toxicity only**; factual and medical tasks are configurable extensions requiring separate data and label audits:

- **Toxic sentence detection** using `toxicity_copy`
- **Factual error sentence detection** using `factual_copy`
- **Inappropriate medical advice detection** using `medical_copy`

The input is a complete response. The output is one row per sentence with a predicted label, a positive-class probability, and character offsets in the response. The dataset contains expert-copied spans rather than ready-made sentence labels, so the preprocessing stage aligns those spans to complete sentences and creates weak labels.

## Project structure

```text
sentence_error_detection_project/
├── configs/                         # general and thesis protocols
├── data/                            # raw data and generated prepared datasets
├── legacy/                          # preserved historical script; never imported
├── reports/                         # metrics, predictions, and concentrated analyses
├── scripts/                         # thin research workflow entry points
├── src/
│   ├── preprocessing.py             # cleaning, sentence splitting, span alignment, labels
│   ├── cli.py                       # argument parsing and validated overrides
│   ├── workflows.py                 # independent handlers for each CLI mode
│   ├── splitting.py                 # grouped splitting and leakage checks
│   ├── data_pipeline.py             # prepared-data persistence and split reuse
│   ├── input_features.py            # target/context model inputs
│   ├── models.py                    # model construction and persistence only
│   ├── classical_training.py        # TF-IDF and fixed-embedding training
│   ├── transformer_training.py      # ALBERT loop and checkpoint selection
│   ├── training.py                  # small public training dispatcher
│   ├── evaluate.py                  # metrics, thresholds, paired statistics
│   ├── baselines.py                 # declared strong TF-IDF baseline suite
│   ├── audit.py                     # dataset, duplicate, source, and error audits
│   ├── weak_labels.py               # non-destructive weak-label rules
│   ├── weak_label_analysis.py       # alignment/agreement/manual audit workflow
│   ├── experiments.py               # controlled label-rule sensitivity experiment
│   └── predict.py                   # saved-model inference
├── tests/
└── run_pipeline.py                  # CLI workflow dispatcher
```

## Data and weak labels

CounselBench-Eval contains repeated expert evaluations of the same response. For the selected task, the corresponding `*_copy` field contains text copied by an annotator as evidence of a problem. The pipeline:

1. groups annotations within one `questionID + responder` response instance;
2. retains a separate `response_text_id` for duplicate/leakage checks;
3. segments each response with spaCy;
4. tries exact character-span alignment first;
5. uses fuzzy sentence/window matching only when exact alignment fails;
6. counts unique annotators supporting each sentence;
7. derives independent label-rule columns without replacing `label_original`;
8. splits at the configured group level, never at sentence level.

`factual_copy` and `medical_copy` are expected by the requested multi-task design, but their exact names must be confirmed against the local CSV. The loader stops with a clear error when the selected target column is absent.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and an English spaCy model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The code falls back to `spacy.blank("en")` with a sentencizer when `en_core_web_sm` is unavailable, but the full model generally gives better sentence boundaries.

## Configuration

Edit `configs/config.yaml`.

### Switch task

```yaml
task:
  name: factual
  text_column: response
  target_column: factual_copy
```

Use `toxicity/toxicity_copy`, `factual/factual_copy`, or `medical/medical_copy`. Validation prevents mismatched task names and target columns.

### Switch model

Classical TF-IDF baseline:

```yaml
model:
  name: tfidf_lr
```

SBERT embeddings with logistic regression:

```yaml
model:
  name: sbert_lr
  sentence_transformer_name: BAAI/bge-base-en-v1.5
```

ALBERT fine-tuning:

```yaml
model:
  name: albert
  pretrained_model_name: albert-base-v2
  max_length: 256
```

ALBERT requires downloading Hugging Face weights on the first run unless the model is already cached.

## Run the pipeline

### Reliability-study protocol

The audited protocol writes only under `results/reliability_study/`, uses one
best fuzzy window rather than the legacy union policy, and makes text-only
near-duplicate components atomic split groups:

```powershell
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode prepare
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode diagnose
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode weak-label-audit
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe run_pipeline.py --config configs/reliability_protocol.yaml --mode baselines --seeds 42,43,44
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe -m scripts.run_split_sensitivity --config configs/reliability_protocol.yaml --seeds 42,43,44
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe -m scripts.build_research_report
```

Run the read-only repository/snapshot audit independently:

```powershell
C:\Users\15509\anaconda3\envs\MentalFlow\python.exe scripts/audit_phase1.py
```

`fuzzy_selection_policy: union` exists only to reproduce historical labels.
New experiments should use the explicit `best_window` policy and a fresh output
directory. Split seeds, model seeds, and fixed-test bootstrap uncertainty answer
different questions and are reported separately.

For corrected reliability experiments, use the fingerprinted, near-duplicate-
aware protocol. It keeps related response content in one partition and writes
to the isolated `results/reliability_study` directory:

```bash
python run_pipeline.py --config configs/reliability_protocol.yaml --mode prepare
python run_pipeline.py --config configs/reliability_protocol.yaml --mode diagnose
python run_pipeline.py --config configs/reliability_protocol.yaml --mode baselines --seeds 42,43,44
python run_pipeline.py --config configs/reliability_protocol.yaml --mode label-sensitivity --sensitivity-models tfidf,albert
```

Export paired prediction audits and statistical comparisons:

```bash
python run_pipeline.py --config configs/reliability_protocol.yaml --mode audit \
  --prediction tfidf=results/reliability_study/analysis/toxicity/baselines/char_seed_42_predictions.csv \
  --prediction albert=results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv

python run_pipeline.py --config configs/reliability_protocol.yaml --mode compare \
  --prediction-a results/reliability_study/weak_label/toxicity/runs/any_support/albert/test_predictions.csv \
  --prediction-b results/reliability_study/analysis/toxicity/baselines/char_seed_42_predictions.csv
```

## Weak-label quality workflow

Rebuild the prepared dataset and generate the concentrated diagnostic bundle:

```bash
python run_pipeline.py --config configs/reliability_protocol.yaml --mode prepare
python run_pipeline.py --config configs/reliability_protocol.yaml --mode weak-label-audit
```

Compare the same fixed TF-IDF and ALBERT weighted-CE protocols under the
`any_support` and `consensus_2` label rules:

```bash
python run_pipeline.py --config configs/reliability_protocol.yaml --mode label-sensitivity --sensitivity-models tfidf,albert
```

Use `--sensitivity-models tfidf` for a fast smoke run. Add `--force` only when
completed sensitivity runs should be retrained.

Use `--seed 43` (and 44–46) to write independent runs under a seed-specific
subdirectory without changing the persisted split. The thesis-ready summaries
are in `results/thesis_key_results.md` and `results/thesis_key_results_zh.md`.

Prepare weak labels and group-safe splits:

```bash
python run_pipeline.py --config configs/config.yaml --mode prepare
```

Train and evaluate on the held-out test set:

```bash
python run_pipeline.py --config configs/config.yaml --mode train
```

Print the saved test report:

```bash
python run_pipeline.py --config configs/config.yaml --mode evaluate
```

Predict one response:

```bash
python run_pipeline.py --config configs/config.yaml --mode predict --text "Your response text here."
```

Or read the response from a text file:

```bash
python run_pipeline.py --config configs/config.yaml --mode predict --input-file sample_response.txt
```

## Machine-learning workflow

```text
Read CSV
→ validate selected task fields
→ normalize response text
→ segment responses into sentences
→ align expert-copied spans
→ construct weak sentence labels
→ split by response ID
→ fit tokenizer/vectorizer only on training data
→ train model
→ tune threshold on validation data
→ evaluate once on test data
→ save model, metrics, and predictions
```

SBERT embeddings are generated by a frozen pretrained encoder and are not fitted on the dataset. TF-IDF is contained inside a scikit-learn pipeline, so its vocabulary is learned only from the training partition.

## Metrics

The test report includes accuracy, precision, recall, F1, macro-F1, ROC-AUC, Average Precision (AP), prevalence, AP lift, the selected threshold, and a confusion matrix. The legacy JSON key `pr_auc` contains `average_precision_score`, not trapezoidal PR area. AP lift is `AP / positive prevalence`.

The decision threshold is selected on validation F1 and then frozen before test evaluation. Test labels are not used for threshold selection.

## Outputs

Outputs are isolated by task and model. Weak-label analysis is concentrated in
one directory:

```text
models/<task>/<model>/
reports/metrics/<task>/<model>/test_metrics.json
reports/predictions/<task>/<model>_prediction.csv
data/processed/<task>/sentences.csv
data/processed/<task>/alignment_log.csv
data/processed/<task>/splits.csv
reports/thesis/weak_label/<task>/alignment_summary.json
reports/thesis/weak_label/<task>/annotator_agreement.csv
reports/thesis/weak_label/<task>/label_rule_summary.csv
reports/thesis/weak_label/<task>/weak_label_audit.csv
reports/thesis/weak_label/<task>/model_by_label_rule.csv
```

## Known limitations and next steps

- Weak labels inherit annotation and span-alignment noise.
- Label reliability still depends on sentence boundaries and fuzzy alignment; inspect the generated audit before treating weak labels as gold labels.
- The classical models use compact default parameters. A separate validation-only tuning module can be added without changing the test protocol.
- ALBERT training uses a straightforward PyTorch loop; mixed precision, gradient accumulation, and experiment tracking can be added for larger runs.
- Further work should include threshold stability analysis, repeated grouped splits, calibration checks, explainability, and manual error analysis.

## Reproducibility and safety

All random processes use the configured seed. Identical response text is kept in one split to prevent direct response leakage. Raw data and trained weights are excluded from Git. This is a research pipeline and is not a clinical diagnostic or treatment system.
