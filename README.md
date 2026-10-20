# From Expert Spans to Sentence-Level Weak Supervision

Research code for Shucheng Guan's UCL MSc **Artificial Intelligence for Sustainable Development** dissertation: constructing and evaluating toxicity detectors for mental-health responses using CounselBench expert annotations.

The completed study covers **toxicity only**. A positive label means that expert-selected evidence was projected onto a sentence under a declared support policy; it is not independently adjudicated sentence-level ground truth. Factual consistency and medical-advice detection are extension interfaces, not completed experiments.

## Start here

- [Reproduction guide](docs/REPRODUCIBILITY.md): environment, E1–E5 commands and saved-result locations.
- [Code architecture and dissertation mapping](docs/ARCHITECTURE.md): module responsibilities and protocol details.
- [Historical development notes](docs/history/README.md): supporting records outside the current workflow.

The manuscript is maintained separately in the local sibling directory `../UCL Dissertaiton/`; it is not bundled in this code repository.

## Research workflow

```text
Expert toxicity spans + response text
  → normalise and segment responses
  → align whole-response markers, exact spans, then one best fuzzy window
  → aggregate distinct annotator votes within each response instance
  → derive any-support / consensus-2 / consensus-2-reliable labels
  → persist a near-duplicate-aware question-grouped partition
  → train and select models using training/validation data
  → evaluate fixed test predictions and paired uncertainty
```

The primary snapshot contains 2,907 sentences from 400 response instances, with 477 any-support positives. The fixed seed-42 split contains 2,035 / 429 / 443 training / validation / test sentences. The test set contains 72 positives.

On this fixed test set, AP is 0.304 for character TF–IDF, 0.330 for target-only BGE and 0.331 for ALBERT. Both paired intervals against character TF–IDF cross zero. These saved results do not establish a reliable advantage from greater model complexity.

## Repository layout

| Path | Purpose |
| --- | --- |
| `run_pipeline.py` | Supported CLI; compatibility imports for older callers |
| `src/` | Data preparation, splitting, model training, evaluation and inference |
| `configs/reliability_protocol.yaml` | Final dissertation protocol; CLI default |
| `configs/config.yaml`, `configs/thesis_protocol.yaml` | Earlier/alternative configurations; select explicitly |
| `scripts/` | Split-sensitivity study, result assembly and compatibility commands |
| `tests/` | Unit and regression checks |
| `data/raw/` | Local CounselBench CSV; excluded from Git |
| `results/reliability_study/` | Primary data snapshot, predictions, metrics and tables |
| `docs/` | Architecture, reproduction and development history |
| `notebooks/` | Exploratory analyses, outside the supported pipeline |
| `legacy/` | Historical monolithic implementation, outside the supported pipeline |
| `data/processed/`, `models/`, `reports/` | Earlier protocol outputs; not the primary thesis snapshot |

Stored predictions and derived tables may repeat information deliberately: they allow the reported metrics to be traced to individual test rows. Preserve the primary snapshot when starting new work.

## Installation and checks

Use Python 3.11 in a dedicated environment. From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_pipeline.py --help
python -m pytest -q
```

`requirements.txt` declares compatibility ranges, not an exact environment lock. The analysed snapshot used `en_core_web_sm` 3.8.0; parser/version changes can change sentence boundaries. See the [recorded environment and fingerprints](docs/REPRODUCIBILITY.md#environment-and-snapshot-identity).

## Running experiments

Place the source CSV at `data/raw/CounselBench.csv`, then follow the [reproduction guide](docs/REPRODUCIBILITY.md). The default configuration writes into `results/reliability_study/`; preparation and training can replace saved artifacts. For a new experiment, copy the YAML **within `configs/`** and change `data.processed_dir`, `output.model_dir` and `output.report_dir` before running it.

The primary lexical comparison uses `--mode baselines`; the fixed E2 lexical model uses `--mode label-sensitivity`. Generic `--mode train --model-name tfidf_lr` is a different tuning protocol. BGE requires pretrained encoder weights, and ALBERT training benefits from a CUDA GPU.

## Interpretation and reproducibility

Average Precision is stored under the historical JSON key `pr_auc`. Thresholds are selected by validation F1. The main bootstrap intervals resample sentences conditional on fixed predictions; they do not measure training-seed uncertainty or account fully for within-response dependence. The reliable-support policy is effectively exact-alignment-only in the recorded run. See the manuscript Appendices A–C for the full specification and limitations.

This is a research pipeline for inspecting projected expert support, not an autonomous clinical safety filter. Raw data and trained weights are not supplied by this repository; obtain the source data under its applicable access and licensing terms.
