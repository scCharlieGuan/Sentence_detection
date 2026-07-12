# Sentence Error Detection for CounselBench

## Project overview

This project identifies problematic sentences inside a mental-health question-answering response. It supports three configurable binary tasks:

- **Toxic sentence detection** using `toxicity_copy`
- **Factual error sentence detection** using `factual_copy`
- **Inappropriate medical advice detection** using `medical_copy`

The input is a complete response. The output is one row per sentence with a predicted label, a positive-class probability, and character offsets in the response. The dataset contains expert-copied spans rather than ready-made sentence labels, so the preprocessing stage aligns those spans to complete sentences and creates weak labels.

## Project structure

```text
sentence_error_detection_project/
├── configs/config.yaml              # task, model, split, and output settings
├── data/
│   ├── raw/                         # local CounselBench CSV (not committed)
│   ├── processed/                   # generated labels and split assignments
│   └── README.md
├── legacy/toxicity_detection_v2.py # untouched uploaded source script
├── models/                          # task/model-specific checkpoints
├── notebooks/README.md              # notebook policy and source-file note
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── config.py                    # YAML loading and validation
│   ├── data_loader.py               # CSV loading and schema checks
│   ├── preprocessing.py             # sentence splitting, alignment, labels, splits
│   ├── dataset.py                   # PyTorch transformer dataset
│   ├── models.py                    # classical, SBERT, and ALBERT factories
│   ├── train.py                     # separate training backends
│   ├── evaluate.py                  # metrics and threshold selection
│   ├── predict.py                   # saved-model inference
│   └── utils.py                     # seeds, paths, logs, and JSON output
├── tests/
├── run_pipeline.py                  # command-line entry point
├── requirements.txt
└── .gitignore
```

## Data and weak labels

CounselBench-Eval contains repeated expert evaluations of the same response. For the selected task, the corresponding `*_copy` field contains text copied by an annotator as evidence of a problem. The pipeline:

1. groups identical response text under one `response_id`;
2. segments each response with spaCy;
3. tries exact character-span alignment first;
4. uses fuzzy sentence/window matching only when exact alignment fails;
5. marks a sentence positive when the configured minimum number of unique copied spans align to it;
6. splits at response level, not sentence level.

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

The test report includes accuracy, precision, recall, F1, macro-F1, ROC-AUC, PR-AUC, prevalence, PR-AUC lift, the selected threshold, and a confusion matrix. PR-AUC is especially useful for the expected class imbalance. PR-AUC lift is `PR-AUC / positive prevalence`; a value of 1 represents the random-ranking baseline.

The decision threshold is selected on validation F1 and then frozen before test evaluation. Test labels are not used for threshold selection.

## Outputs

Outputs are isolated by task and model:

```text
models/<task>/<model>/
reports/metrics/<task>/<model>/test_metrics.json
reports/predictions/<task>/<model>_prediction.csv
data/processed/<task>/sentences.csv
data/processed/<task>/alignment_log.csv
data/processed/<task>/splits.csv
```

## Known limitations and next steps

- Weak labels inherit annotation and span-alignment noise.
- The current agreement counter uses unique copied spans because the supplied script lacks a reliable annotator identifier. When an annotator ID is available, agreement should be counted by annotator rather than unique text.
- The classical models use compact default parameters. A separate validation-only tuning module can be added without changing the test protocol.
- ALBERT training uses a straightforward PyTorch loop; mixed precision, gradient accumulation, and experiment tracking can be added for larger runs.
- Further work should include threshold stability analysis, repeated grouped splits, calibration checks, explainability, and manual error analysis.

## Reproducibility and safety

All random processes use the configured seed. Identical response text is kept in one split to prevent direct response leakage. Raw data and trained weights are excluded from Git. This is a research pipeline and is not a clinical diagnostic or treatment system.
