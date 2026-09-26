# Thesis Key Results — Corrected Protocol

## 1. Executive summary

This report contains only results that match the corrected toxicity dataset and its fixed evaluation split. The dataset contains 2,907 sentences from 400 responses, with 477 positive sentences under the primary any-support rule. Its fingerprint is cbb3cabf8249d351cdd3c86bb867e9e2059320009c36ed666ae505bec6c10aba. The strict test set contains 443 sentences, including 72 positives.

The main conclusions are:

- Span-to-sentence weak supervision is operationally feasible but not equivalent to independently adjudicated sentence labels. Of 405 expert spans, 361 (89.14%) were aligned; 100 (24.69%) required fuzzy matching and 44 (10.86%) remained unmatched.
- The labels are sensitive to annotator support. Of the 477 any-support positives, 353 (74.00%) were supported by only one annotator. Requiring two annotators reduces the positive set to 124 sentences (4.27% of all sentences).
- Evaluation design materially changes the measured result. Across three split seeds, naive sentence splitting produced mean TF-IDF PR-AUC 0.343, compared with 0.248 for near-duplicate-aware question grouping.
- On the fixed 443-sentence test set, ALBERT obtained the highest PR-AUC point estimate (0.331), closely followed by target-only BGE (0.330) and character TF-IDF (0.304). Neither ALBERT nor BGE showed a statistically reliable PR-AUC advantage over character TF-IDF.
- Adding the next sentence to BGE raised recall from 0.681 to 0.903 but reduced precision, F1, macro-F1, and PR-AUC. It changed the operating point rather than improving ranking quality.
- Under the stricter consensus-2 label rule, ALBERT and TF-IDF obtained PR-AUC 0.139 and 0.136, respectively. These values describe a different, rarer target and should not be ranked directly against the any-support results.

Accordingly, H1 is partially supported, H2 is inconclusive, and H3 is partially supported. The evidence suggests that supervision policy and dataset structure are at least as influential as the tested representation changes, but it does not establish a causal bottleneck.

## 2. Experimental scope and protocol

The completed reliability study covers toxicity only. Factual-consistency and medical-advice experiments were not run. All primary model results below use:

- corrected best-window span alignment;
- the any-support label rule unless otherwise stated;
- a fixed near-duplicate-aware split with 2,035 training, 429 validation, and 443 test sentences;
- zero overlap across train, validation, and test for near-duplicate group, response ID, and response-text ID;
- validation data for checkpoint selection and threshold selection; and
- the untouched test set for final evaluation.

PR-AUC is the primary ranking metric because the test prevalence is 16.25%. Positive-class F1 and macro-F1 are also reported. Accuracy is treated as auxiliary because a majority classifier already achieves 0.837 accuracy while detecting no positives.

ALBERT was fine-tuned with weighted cross-entropy. Its checkpoint was selected by validation PR-AUC and its decision threshold by validation F1. BGE uses frozen sentence embeddings with logistic regression. Each encoder configuration was run once, so the reported bootstrap intervals quantify uncertainty from the fixed test sample, not variation across training seeds.

## 3. Label construction and annotation evidence

### 3.1 Span alignment

| Alignment outcome | Spans | Percentage | Interpretation |
| --- | ---: | ---: | --- |
| Exact match | 252 | 62.22% | Directly located in the response |
| Fuzzy match | 100 | 24.69% | Assigned using the best contiguous sentence window |
| Whole response | 9 | 2.22% | Applied across the response and therefore highly context-dependent |
| Unmatched | 44 | 10.86% | Below the fallback similarity threshold |
| Matched in total | 361 | 89.14% | Exact, fuzzy, or whole-response alignment |

The 89.14% match rate demonstrates technical feasibility. It does not by itself establish label validity: fuzzy matching changes the spatial precision of the annotation, whole-response labels do not supply sentence-specific boundaries, and unmatched spans remove intended positive evidence.

### 3.2 Annotator support and label rules

| Label rule | Positive sentences | Overall rate | Test positives | Test rate |
| --- | ---: | ---: | ---: | ---: |
| Any support: at least 1 annotator | 477 | 16.41% | 72 | 16.25% |
| Consensus 2: at least 2 annotators | 124 | 4.27% | 26 | 5.87% |
| Consensus 2 reliable: at least 2 reliable supports | 78 | 2.68% | 12 | 2.71% |

Among the 477 primary positives, 353 have one annotator vote, 87 have two votes, and 37 have at least three votes. Thus 74.00% of primary positives depend on the most permissive part of the any-support policy. The stricter rules may improve specificity, but they also change the target definition and sharply reduce the number of positives; lower raw PR-AUC or F1 under those rules is not evidence that the labels are worse.

## 4. RQ1: effects of label construction and splitting

### 4.1 Split sensitivity

The same TF-IDF evaluation procedure was repeated with three split seeds.

| Split strategy | Mean PR-AUC ± SD | Mean F1 ± SD | Mean test prevalence | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Naive sentence | 0.343 ± 0.055 | 0.335 ± 0.061 | 16.48% | Optimistic because related content can cross partitions |
| Response grouped | 0.333 ± 0.064 | 0.300 ± 0.019 | 17.26% | Keeps responses intact |
| Question grouped | 0.315 ± 0.019 | 0.343 ± 0.059 | 14.02% | Tests generalisation across questions |
| Near-duplicate-aware question grouped | 0.248 ± 0.034 | 0.298 ± 0.019 | 16.18% | Primary strict condition |
| Topic grouped | 0.328 ± 0.055 | 0.303 ± 0.068 | 13.62% | Changes the target domain |
| Responder grouped | 0.271 ± 0.167 | 0.386 ± 0.214 | 25.45% | Highly unstable due to domain and prevalence shift |

The mean PR-AUC decrease from naive splitting to the strict split is 0.095. The seed-specific differences are 0.108, 0.167, and 0.011, so the direction is consistent but the magnitude is unstable. Responder- and topic-grouped results should be interpreted as domain-shift stress tests rather than cleaner estimates of the same target distribution.

### 4.2 Label-rule sensitivity with TF-IDF and ALBERT

| Label rule | Model | Test positives | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Any support | Fixed TF-IDF sensitivity model | 72 | 0.195 | 0.333 | 0.246 | 0.247 | 0.601 |
| Any support | ALBERT | 72 | 0.352 | 0.347 | 0.350 | 0.331 | 0.664 |
| Consensus 2 | Fixed TF-IDF sensitivity model | 26 | 0.111 | 0.192 | 0.141 | 0.136 | 0.684 |
| Consensus 2 | ALBERT | 26 | 0.178 | 0.308 | 0.225 | 0.139 | 0.754 |

Within each rule, the rows share the same test sentences and labels. Across rules, however, prevalence and target semantics change. ALBERT clearly improves F1 over the fixed sensitivity TF-IDF configuration under both rules. Its PR-AUC advantage is substantial under any support (+0.084; paired 95% CI [0.009, 0.170]) but very small under consensus 2 (+0.003). The any-support sensitivity TF-IDF is a deliberately fixed configuration and is not the stronger character TF-IDF baseline used in the main comparison.

### Direct answer to RQ1

Label construction and split strategy both materially affect the observed task. Most primary positives rely on one annotator, stricter support thresholds sharply reduce prevalence, and naive splitting yields higher average PR-AUC than duplicate-aware grouping. Reported performance is therefore protocol-dependent and should always be accompanied by the label rule, split policy, and dataset fingerprint.

## 5. RQ2: model comparison on the corrected test set

### 5.1 Main results

| Model | Input | Precision | Recall | Positive F1 | Macro-F1 | PR-AUC | PR-AUC 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Class-prior / majority | None | 0.000 | 0.000 | 0.000 | 0.456 | 0.163 | Reference prevalence |
| TF-IDF word (1,2)-gram LR | Target sentence | 0.265 | 0.361 | 0.306 | 0.571 | 0.267 | — |
| TF-IDF character (3,5)-gram LR | Target sentence | 0.270 | 0.417 | 0.328 | 0.576 | 0.304 | [0.221, 0.417] |
| BGE embedding + LR | Target sentence | 0.251 | 0.681 | 0.367 | 0.547 | 0.330 | [0.238, 0.428] |
| BGE embedding + LR | Target + next sentence | 0.199 | 0.903 | 0.326 | 0.387 | 0.322 | [0.239, 0.432] |
| ALBERT, weighted CE | Target sentence | 0.352 | 0.347 | 0.350 | 0.612 | 0.331 | [0.234, 0.453] |

ALBERT has the highest PR-AUC, macro-F1, and precision point estimates. BGE target-only has the highest positive F1 because it detects more positives, while character TF-IDF remains a competitive and simpler baseline. The wide and strongly overlapping confidence intervals show that the apparent ranking is uncertain.

### 5.2 Paired comparisons

| Comparison, A minus B | PR-AUC difference | Paired 95% CI | Bootstrap p-value | Conclusion |
| --- | ---: | --- | ---: | --- |
| BGE target − character TF-IDF | +0.026 | [-0.070, 0.106] | 0.676 | No reliable advantage |
| ALBERT − character TF-IDF | +0.028 | [-0.066, 0.119] | 0.609 | No reliable advantage |
| ALBERT − BGE target | +0.001 | [-0.079, 0.090] | 0.919 | Effectively tied |
| BGE target+next − BGE target | -0.008 | [-0.071, 0.062] | 0.914 | Added context did not improve ranking |

These tests are paired over the same 443 test rows. They support a cautious claim: semantic models have modest positive point differences over character TF-IDF, but the present sample does not establish that those differences are reliable. ALBERT and BGE target-only are indistinguishable on PR-AUC.

Thresholded predictions tell a complementary but operating-point-dependent story. ALBERT predicts fewer positives than BGE, giving higher precision and macro-F1 but lower recall. BGE target+next reaches 90.3% recall by predicting 83.1% of the test set as positive, which explains its low precision and macro-F1. It should not be described as a better model solely because of recall.

### Direct answer to RQ2

The corrected experiment does not establish that semantic or contextual models reliably outperform lexical baselines. ALBERT and target-only BGE show small PR-AUC point gains over character TF-IDF, but both paired intervals cross zero. Limited next-sentence context changes the precision–recall trade-off without improving PR-AUC.

## 6. Ablation findings

| Ablation | Change | Observed effect | Interpretation |
| --- | --- | --- | --- |
| TF-IDF document-frequency filter | max_df 1.00, 0.95, 0.80 | Identical PR-AUC and F1 | No effect within the tested range; not a general stop-word result |
| Word n-grams | Unigram → (1,2)-gram | PR-AUC +0.004; F1 +0.011 | Small descriptive gain |
| Lexical granularity | Word bigrams → character (3,5)-grams | PR-AUC +0.037; F1 +0.022 | Character features are the strongest tested lexical representation |
| BGE context | Target → target+next | PR-AUC -0.008; F1 -0.041; recall +0.222 | Context shifts the threshold behaviour but does not improve ranking |
| Annotator threshold | Any support → consensus 2 | Test prevalence 16.25% → 5.87% | Produces a smaller, semantically stricter target |
| Split strategy | Naive → near-duplicate aware | Mean PR-AUC -0.095 | Leakage control materially lowers the estimate |

## 7. TF-IDF feature analysis

The largest positive word-feature coefficients include will, you, the, is, we, really, because, her, you will, you need, with her, but unable, and unable to. Large negative coefficients include can, and, especially, support, client, professional, safety, alone, when you, can help, and not alone.

Many of these are function words, stylistic phrases, broad topics, or repeated response templates rather than direct toxicity markers. Together with the responder- and topic-specific prevalence differences, this pattern is consistent with lexical shortcut learning. It does not prove leakage or causation: coefficients can reflect correlated topics, response styles, label policy, or source-specific templates.

## 8. Error analysis

The qualitative review table contains 56 rows: eight examples each for TF-IDF false positives, false negatives, fuzzy-aligned positives, single-annotator positives, multi-annotator positives, context-dependent proxies, and potentially noisy labels. Categories overlap, so 56 is not the number of unique errors or an error prevalence estimate.

The main observed patterns are:

- False positives often contain generic reassurance or directive language that may resemble repeated positive-label templates.
- False negatives include directive or assumptive statements, some of which have only one annotator vote and some of which are recovered by semantic models.
- Single-annotator cases dominate the positive set and are intrinsically ambiguous: low support may reflect disagreement, rare valid concerns, or projection noise.
- Fuzzy cases include both harmless formatting differences and genuinely uncertain span boundaries; they should not be treated as one homogeneous noise class.
- Whole-response or multi-sentence spans can assign a contextual judgement to sentences that look benign in isolation.

The review queue has not been independently adjudicated. A model–label disagreement is therefore an error against the weak target, not proof that either the model or the label is clinically correct.

## 9. Hypothesis assessment

| Hypothesis | Decision | Evidence | Why the decision remains qualified |
| --- | --- | --- | --- |
| H1: label construction and splitting affect results | Partially supported | Positive prevalence changes from 16.41% to 4.27% under consensus 2; naive versus strict mean PR-AUC differs by 0.095 | Only three split seeds; label rules change target semantics and prevalence |
| H2: semantic/contextual models outperform lexical baselines | Inconclusive | BGE and ALBERT exceed character TF-IDF by 0.026 and 0.028 PR-AUC | Both paired CIs include zero; ALBERT and BGE ran once; target+next is not better |
| H3: supervision and structure are a larger bottleneck than model capacity | Partially supported | Split effect is larger than the tested representation point effects; annotator policy sharply changes the target | The comparisons are observational and cannot identify a causal bottleneck |

## 10. Thesis-ready answers

### Main research question

Expert span annotations can be transformed into a usable sentence-level weak-supervision signal, but the resulting labels should not be presented as an independently validated gold standard. The transformation aligns most spans and supports above-chance modelling, yet its output is sensitive to fuzzy matching, whole-response projection, and annotator-support policy.

### RQ1

Dataset construction and split design materially affect measured performance. Duplicate-aware grouping lowers average TF-IDF PR-AUC relative to naive splitting, and stricter annotator thresholds substantially reduce positive prevalence. These changes must be treated as changes to the evaluation protocol and, in the case of label thresholds, to the target itself.

### RQ2

ALBERT and frozen BGE embeddings provide small PR-AUC point gains over character TF-IDF on the corrected test set, but paired uncertainty intervals include zero. ALBERT is more precise, target-only BGE is more recall-oriented, and adding the next sentence yields very high recall at the cost of many false positives. The evidence does not support a general claim that semantic or contextual models are superior.

## 11. Thesis-ready results paragraph

Under the corrected best-window alignment protocol, 361 of 405 expert toxicity spans (89.14%) were mapped to sentences, producing 477 positive sentences among 2,907 instances (16.41%). However, 100 spans (24.69%) required fuzzy alignment, 44 (10.86%) remained unmatched, and 353 of 477 positive sentences (74.00%) were supported by only one annotator. Across three split seeds, naive sentence splitting yielded higher mean TF-IDF PR-AUC than near-duplicate-aware question grouping (0.343 versus 0.248). On the fixed 443-sentence strict test set, ALBERT, target-only BGE, and character TF-IDF achieved PR-AUC values of 0.331, 0.330, and 0.304, respectively. The paired ALBERT–TF-IDF difference was 0.028 (95% CI [-0.066, 0.119]), and the paired BGE–TF-IDF difference was 0.026 (95% CI [-0.070, 0.106]); neither established a reliable advantage. Adding the next sentence to BGE reduced PR-AUC by 0.008 while increasing recall from 0.681 to 0.903 and lowering precision from 0.251 to 0.199. Under the consensus-2 label rule, test prevalence fell to 5.87%, with ALBERT and TF-IDF PR-AUC of 0.139 and 0.136.

## 12. Discussion-ready interpretation

The most defensible interpretation is not that model architecture is irrelevant, but that the present model differences are smaller than the uncertainty created by supervision and evaluation choices. Character TF-IDF remains competitive, suggesting that lexical and template cues carry substantial signal. BGE improves recall, and ALBERT improves precision and macro-F1, but neither shows a reliable PR-AUC gain over the lexical baseline. The failure of target+next input to improve ranking also shows that merely adding text is not the same as modelling context effectively.

For this mental-health evaluation setting, false positives and false negatives have different practical costs, and threshold choice should be tied to a stated use case. None of the current results validates deployment for clinical triage, automated safety enforcement, or quality assurance. Independent blinded sentence-level adjudication is required before error rates can be interpreted as clinical validity.

## 13. Recommended conclusion

The study demonstrates a reproducible way to project expert span annotations into sentence-level weak labels and shows that the resulting signal is learnable. It also shows that conclusions depend strongly on the annotation rule and split design. On the corrected, duplicate-aware test set, ALBERT and BGE have small but uncertain PR-AUC advantages over character TF-IDF, while simple next-sentence context does not improve ranking. The results therefore support cautious use of weak supervision for research prototyping, not a claim of gold-standard reliability or model readiness.

## 14. Limitations

- Only toxicity was rebuilt and evaluated.
- There is no independently adjudicated sentence-level gold set.
- Seventy-four per cent of primary positive sentences have one annotator vote.
- Fuzzy and whole-response alignment can introduce boundary and context ambiguity.
- Split sensitivity uses three seeds only.
- ALBERT and BGE each have one training run, so encoder-seed uncertainty is unknown.
- The fixed test set has 72 primary positives and only 26 consensus-2 positives, producing wide intervals.
- Thresholds optimise validation F1 and may not match a clinical cost function.
- The context experiment adds only the next sentence; it is not a target-aware cross-encoder.
- Factual consistency, medical advice, consensus-2-reliable modelling, and independent error adjudication remain unrun.

## 15. Evidence provenance

| Claim or table | Primary artifact |
| --- | --- |
| Dataset fingerprint and split integrity | results/reliability_study/data/toxicity/protocol_manifest.json |
| Alignment counts | results/reliability_study/weak_label/toxicity/alignment_summary.json |
| Label prevalence by rule and split | results/reliability_study/weak_label/toxicity/label_rule_summary.csv |
| Annotator-support distribution | results/reliability_study/weak_label/toxicity/annotator_agreement.csv |
| Main model metrics | results/reliability_study/paper_outputs/main_experiment_results.csv |
| ALBERT label-rule comparison | results/reliability_study/weak_label/toxicity/model_by_label_rule.csv |
| Split sensitivity | results/reliability_study/paper_outputs/split_sensitivity_results.csv |
| Paired bootstrap comparisons | results/reliability_study/paper_outputs/paired_model_comparisons.csv |
| Ablations | results/reliability_study/paper_outputs/ablation_study.csv |
| Qualitative review queue | results/reliability_study/paper_outputs/qualitative_error_analysis.csv |
| Hypothesis decisions | results/reliability_study/paper_outputs/hypothesis_evidence_matrix.csv |

## 16. Verification status

- All primary prediction files contain the same 443 test rows and labels.
- Current ALBERT metric files record the corrected dataset fingerprint and near-duplicate split provenance.
- Bootstrap intervals were generated from 1,000 fixed-test resamples; paired differences use 2,000 paired resamples.
- The main tables include only artifacts from the corrected, fingerprinted protocol.
- Numerical statements in this report are traceable to the artifacts listed above.
