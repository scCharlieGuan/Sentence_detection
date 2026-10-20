# Code Architecture and Dissertation Mapping

This project is organised around Chapters 3–6 and Appendices A–C in `UCL Dissertaiton/md files`. Its research objective is to predict sentence-level toxicity weak labels projected from expert-copied spans. Factual and medical tasks remain extension interfaces; they are not tasks validated by the completed dissertation experiments.

## 1. Layers and suggested reading order

| Layer | Files | Responsibilities |
| --- | --- | --- |
| Compatibility entry point | `run_pipeline.py` | Launch the CLI and retain the historical `prepare_data` and `load_prepared` imports |
| Command-line arguments | `src/cli.py` | Parse arguments, check mode-specific requirements and default to the dissertation reliability protocol |
| Configuration | `src/config.py` | Load YAML, validate settings, and copy and validate CLI overrides |
| Workflows | `src/workflows.py` | Provide one handler per mode and consistent logging and output paths |
| Data processing | `src/data_loader.py`, `src/preprocessing.py` | Read and clean raw data, segment sentences, align spans and count annotator support |
| Label construction | `src/weak_labels.py` | Derive alternative support policies while preserving `label_original` |
| Splitting and persistence | `src/splitting.py`, `src/data_pipeline.py` | Construct grouped partitions, save/load snapshots, and check row identities and fingerprints |
| Inputs and models | `src/input_features.py`, `src/models.py`, `src/dataset.py` | Construct sentence/context inputs and model objects |
| Training | `src/training.py`, `src/classical_training.py`, `src/transformer_training.py` | Dispatch training and implement separate classical-model and ALBERT backends |
| Experiments | `src/baselines.py`, `src/experiments.py` | Run lexical baselines and label-policy sensitivity experiments |
| Evaluation and auditing | `src/evaluate.py`, `src/audit.py`, `src/weak_label_analysis.py` | Compute metrics and bootstrap estimates, and audit data and predictions |
| Inference | `src/predict.py` | Load saved models and thresholds and produce sentence predictions |

The suggested reading order is: entry point → CLI → workflows → data/training modules. Research scripts can call the underlying modules directly without going through the CLI. `legacy/`, `notebooks/` and historical artifacts are outside the main execution path.

## 2. Mapping the dissertation to experiments

| Dissertation section | Implementation and execution entry point |
| --- | --- |
| Chapter 3, Appendix A: weak-label construction | `preprocessing.py`, `weak_labels.py`; `--mode prepare` |
| E1: alignment diagnostics | `weak_label_analysis.py`; `--mode weak-label-audit` |
| E2: label-policy sensitivity | `experiments.py`; `--mode label-sensitivity` |
| E3: six split strategies, seeds 42–44 | `scripts/run_split_sensitivity.py` |
| E4: lexical models, frozen BGE and ALBERT | `baselines.py` and the two training backends |
| E5: paired differences and manual review queue | `evaluate.py`, `audit.py`; `--mode compare`, `--mode audit` |
| Chapter 5, Appendix C: result assembly | `scripts/build_research_report.py` and saved result snapshots |
| Chapter 6 | Interpretation boundaries and future extensions, rather than additional training targets |

## 3. Protocol and compatibility

- The default configuration is `configs/reliability_protocol.yaml`. `config.yaml` and `thesis_protocol.yaml` are alternative/historical configurations and must be selected explicitly when needed.
- Exact matching takes priority. Fuzzy matching uses `best_window`, with a primary threshold of 0.78, a fallback threshold of 0.68 and a maximum contiguous window of three sentences.
- Response instances are defined by questionID + responder + normalised response text. Support is aggregated across distinct annotators; repeated evidence from the same annotator does not create additional votes.
- The primary split uses connected components of questionID groups linked by near-duplicate text, with a character-similarity threshold of 0.90, a minimum fuzzy-match text length of 30 characters and 500 allocation attempts.
- Dataset fingerprinting follows the fields and serialisation specified in Appendix A. Loading checks the fingerprint and row identities in the split table. Newly generated manifests additionally store `split_file_sha256`; older manifests without this field remain readable.
- The dataset fingerprint does not cover all derived labels or establish label correctness. Older snapshots lack a split hash, so their manifests alone cannot detect every otherwise valid reassignment of rows to partitions.
- The JSON key `pr_auc` contains Average Precision (AP). Its historical name is retained for compatibility with saved reports.
- Thresholds are selected by validation F1; test data must not be used for parameter selection. BGE's inner cross-validation remains grouped by questionID, preserving the protocol and limitations disclosed in Appendix B.
- `--mode train --model-name tfidf_lr` uses the general classical-model training workflow. Use `--mode baselines` for the principal E4 lexical models and `--mode label-sensitivity` for the fixed E2 model. These three workflows use different tuning protocols.
- The default loss in the general ALBERT configuration does not represent the dissertation's weighted cross-entropy condition. Specify `--loss weighted_cross_entropy` when training that condition directly. The label-sensitivity workflow sets this loss automatically.

## 4. Reproduction and historical records

Complete commands, environment versions, snapshot fingerprints and output locations are maintained in the [reproduction guide](REPRODUCIBILITY.md) to prevent command instructions from diverging across documents.

`scripts/analyze_weak_labels.py` and `scripts/analyze_toxic_weak_labels.py` are compatibility entry points. Both reuse the main CLI's `weak-label-audit` mode and default to the reliability protocol. Historical analyses require an explicitly selected older configuration. Both the generic script and the toxicity alias support direct execution and `python -m scripts.<name>`.

Development records are collected in [history/](history/README.md). Their older paths and preliminary conclusions are retained as historical context, not current execution instructions. The dissertation text, dataset snapshots and predictions remain the evidence for checking the final research conclusions.
