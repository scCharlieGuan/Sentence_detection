# Refactoring report

## Source analysis

The uploaded source is a single Python script rather than a Jupyter notebook. It implements:

- CSV loading from a hard-coded relative path;
- spaCy sentence segmentation;
- exact and fuzzy alignment of `toxicity_copy` spans;
- weak sentence-label construction;
- response-level train/validation/test splitting;
- TF-IDF + logistic regression, calibrated linear SVM, and SBERT-based logistic regression, LightGBM, and ExtraTrees models;
- validation threshold tuning, sentence metrics, response retrieval metrics, test evaluation, joblib persistence, and single-response inference.

The original file has been preserved unchanged in `legacy/toxicity_detection_v2.py`.

## Main risks found and changes

1. **Single-task hard-coding**: `toxicity_copy` and toxicity-specific output names were embedded throughout the script. The refactor centralizes task name and target column in YAML and validates their pairing.
2. **Monolithic responsibilities**: configuration, label generation, training, evaluation, persistence, and inference shared one file. They are now separated under `src/`.
3. **Hard-coded paths and parameters**: moved to `configs/config.yaml` and resolved relative to the project root.
4. **Potential response leakage**: the source factorized normalized response text, which is directionally correct. The refactor makes this explicit and persists split assignments. Identical response text cannot appear in different splits.
5. **Ambiguous annotator agreement**: duplicate copied spans could inflate counts in the original grouping. The refactor counts unique copied span strings. This is still an approximation; true annotator IDs should be used when available.
6. **Model-selection inconsistency**: the source selected hyperparameters by PR-AUC lift while selecting thresholds by validation F1. The refactor currently presents a single configured model and selects only the threshold on validation F1. It avoids claiming that compact defaults are fully tuned.
7. **Full-dataset SBERT encoding**: this is not target leakage because the frozen encoder is not fitted on labels or corpus statistics, but it can be memory-heavy. The refactor retains frozen encoding for SBERT models.
8. **Model persistence gaps**: the source saved classifiers but relied on external knowledge of thresholds and encoder names. The refactor saves `inference_config.json` and the encoder/model metadata under task-specific directories.
9. **GPU handling**: the original code did not include a transformer training device path. ALBERT now selects CUDA when available and otherwise runs on CPU.
10. **No explicit CLI**: replaced with `run_pipeline.py` modes for prepare, train, evaluate, and predict.

## Validation performed

Static compilation succeeded, the default YAML configuration loaded successfully, and all three lightweight pytest checks passed. Full training was not run because the CounselBench CSV was not supplied and pretrained ALBERT/SBERT weights may require network access and substantial compute.
