# Reliability study report (executed subset)

## Observed results

- Dataset fingerprint: `cbb3cabf8249d351cdd3c86bb867e9e2059320009c36ed666ae505bec6c10aba`; raw-data fingerprint:
  `9d2d2201da6dd4de3e0ad986e101de894606984174a40053443f672a3d24a7ee`.
- Corrected best-window labels: 2907 sentences, 477
  positives (0.164). Alignment: 405 spans,
  44 unmatched, and 100 fuzzy.
- Strict test set: 443 sentences, prevalence 0.163, with zero response-text,
  normalized-exact, or >=0.90 near-duplicate overlap in the generated diagnostic.
- Best lexical ranking result in the declared baseline table is char TF-IDF
  (PR-AUC 0.304). BGE target is 0.330, BGE target+next is 0.322,
  and corrected-label ALBERT is 0.331.
- BGE target minus char TF-IDF PR-AUC =
  0.026, 95% CI
  [-0.070,
  0.106].
- ALBERT minus char TF-IDF PR-AUC =
  0.028,
  95% CI [-0.066,
  0.119].
- Under the stricter consensus-2 label rule, ALBERT has PR-AUC 0.139 and
  TF-IDF has 0.136; this is a label-policy sensitivity result, not a direct
  performance comparison with the any-support task because prevalence changes.
- Naive sentence-split TF-IDF mean PR-AUC is
  0.343; duplicate-aware question-grouped mean is
  0.248 across split seeds 42-44.

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
