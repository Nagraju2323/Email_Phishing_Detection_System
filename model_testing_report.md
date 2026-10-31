# Phishing Email Model Verification Report

**Audit date:** 2026-10-31  
**Artifacts tested:** `logistic_regression_model.pkl`, `tfidf_vectorizer.pkl`  
**Dataset used:** `cleaned_dataset.csv`  
**Model or existing project files changed:** No. This report is new.

## Executive summary

The saved model and vectorizer load and produce the same predictions as direct `predict_proba` evaluation. The original 80:20 stratified split is reproducible, and the saved TF-IDF vocabulary exactly matches a vectorizer fit on reconstructed training text. There are no exact normalized email overlaps between train and test.

A near-duplicate scan did find **781 of 16,416 test rows (4.76%)** with at least one highly similar training email. This means the reported random-split accuracy of 98.68% may be optimistic for unrelated, future emails. All detected near-duplicate pairs had matching labels. Retest with a near-duplicate-grouped split and an independent, newer dataset before treating the score as production generalization performance.

## Split and leakage checks

- Reconstructed the split with `test_size=0.20`, `random_state=42`, and stratification, matching `train_model.py`.
- Train: **65,661** rows (31,386 label 0; 34,275 label 1). Test: **16,416** rows (7,847 label 0; 8,569 label 1).
- Exact normalized text overlap across the split: **0**.
- A newly fitted TF-IDF vectorizer using training text has exactly the same vocabulary mapping as the saved vectorizer (**200,000 features**). This supports the intended train-only vectorizer fit.
- Near-duplicate scan: word-trigram Jaccard similarity ≥0.90, with SimHash used to identify candidate pairs. **781 test rows (4.76%)** had one or more qualifying train counterparts; **213,651 qualifying pairs** were detected, all with the same label. The high pair count reflects repeated variants sharing common message text; it is a pair count, not a count of unique test emails.
- These are highly similar message variants, not exact duplicates. The existing exact-row deduplication therefore did not prevent them crossing the random split.

## Recomputed holdout evaluation

| Metric | Result |
|---|---:|
| Accuracy | 98.68% |
| Phishing precision | 98.42% |
| Phishing recall | 99.05% |
| Phishing F1 | 98.74% |
| ROC-AUC | 0.9987 |

Confusion matrix (rows = actual; columns = predicted; label order 0=legitimate, 1=phishing):

|  | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Actual legitimate | 7,711 | 136 |
| Actual phishing | 81 | 8,488 |

The classification report rounds both classes' precision, recall, and F1 to 0.98–0.99. The positive class for the reported phishing precision and recall is label 1.

### False positives (136)

False positives are legitimate-labeled test messages classified as phishing. The highest phishing-probability examples include:

- A travel/resource message listing vacation links (phishing probability 99.32%).
- A message welcoming a user to an online Citibank account and linking to the bank’s sign-in page (97.23%).
- An SPSS account registration message asking the user to verify an email address (96.56%).
- A sales-lead marketing message (95.76%).

This suggests the model can flag legitimate account onboarding, link-heavy notices, and marketing content. A deployed detector may inconvenience users if it blocks such messages outright.

### False negatives (81)

False negatives are phishing-labeled test messages classified as legitimate. Examples with the lowest phishing probabilities include:

- A forwarded technical discussion about click fraud (4.54%).
- A job/education solicitation with an unusual degree offer (9.01%).
- A power-trading event invitation (9.24%).
- A long forwarded message with names, addresses, and discussion text (9.51%).
- A movie lookalike/audition solicitation (15.79%).

These examples are less overtly credential-focused and resemble ordinary discussion, invitations, or solicitations. They may also reflect limitations or ambiguity in the dataset labels. Review them against the intended threat definition before interpreting every error as a model failure.

## New-message pattern probes

Each message below is newly composed for this audit and was checked for exact normalized-text absence from the cleaned corpus. The outputs are model predictions, not independently adjudicated truth labels.

| Pattern | Prediction | Confidence | Phishing probability |
|---|---|---:|---:|
| Legitimate meeting note | Legitimate Email | 96.37% | 3.63% |
| Legitimate invoice notice | Legitimate Email | 77.39% | 22.61% |
| Fake login/password reset | Phishing Email | 51.95% | 51.95% |
| OTP verification scam | Phishing Email | 89.96% | 89.96% |
| Banking/financial credential request | Phishing Email | 95.17% | 95.17% |
| Suspicious URL and macro attachment | Phishing Email | 60.73% | 60.73% |
| Urgent account suspension | Phishing Email | 82.63% | 82.63% |

The fake reset and malicious attachment probes are close to the decision boundary. They receive the expected phishing label, but confidence is modest; those patterns need broader testing and should not be treated as reliably detected based on one synthetic example each.

## Artifact and confidence checks

- Both Joblib artifacts loaded successfully and accepted the reconstructed test features.
- Model classes are `[0, 1]`; prediction code interprets 0 as legitimate and 1 as phishing.
- For the fresh probes, class probabilities summed to 1.0. The reported `confidence` matched the probability of the predicted class, and `phishing_probability` matched the class-1 probability.
- A separate call through `predict_email()` matched the saved model/vectorizer calculation for the banking probe.
- Probability arithmetic is correct, but this does **not** establish probability calibration. A score such as 90% should not be interpreted as a verified 90% real-world likelihood without a calibration study on representative data.

## Issues and recommended next steps

1. **Near-duplicate leakage:** make connected email threads or near-duplicate clusters indivisible during splitting, then retrain and report metrics on a group-held-out test set. Keep a final independent test set untouched during model selection.
2. **Dataset representativeness and label review:** manually review the false positives and false negatives, especially legitimate account verification messages and phishing-labeled discussion/invitation messages. Confirm label definitions and remove or correct mislabeled rows.
3. **Boundary behavior:** assess performance by pattern on a larger, adjudicated set. Consider a review/abstain band for low-confidence messages instead of forcing an automatic decision; select thresholds based on false-positive and false-negative costs.
4. **Deployment robustness:** test current phishing examples and benign lookalikes from a temporally separate source. Include URL-domain features and attachment metadata only if those inputs are available and safely handled at inference time.

## Reproduction

The split and metrics were recomputed without retraining or overwriting the saved model. To repeat the existing training pipeline, run `python train_model.py`; note that this command regenerates the cleaned CSV and model artifacts.

