"""Validate the selected saved Logistic Regression model on external mail.

This script never fits a model or vectorizer. It reads the official test
partition of the external Hugging Face corpus, removes training overlaps, and
writes a new Markdown report.
"""
from datetime import date
import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_cleaning import TARGET_COLUMN, TEXT_COLUMN
from group_split import build_similarity_groups
from preprocessing import normalize_text

ROOT = Path(__file__).resolve().parent
EXTERNAL_TEST_PATH = ROOT / "external_validation_data" / "test-00000-of-00001.parquet"
TRAINING_DATA_PATH = ROOT / "cleaned_dataset.csv"
SPLIT_MANIFEST_PATH = ROOT / "grouped_split_manifest.csv"
MODEL_PATH = ROOT / "comparison_logistic_regression_model.pkl"
VECTORIZER_PATH = ROOT / "comparison_logistic_regression_tfidf.pkl"
COMPARISON_RESULTS_PATH = ROOT / "algorithm_comparison_results.csv"
REPORT_PATH = ROOT / "external_validation_report.md"
EXTERNAL_TARGET_COLUMN = "target"


def canonical_for_overlap(text):
    """Normalize punctuation/spacing differences for duplicate analysis only."""
    return " ".join(re.findall(r"\w+", "" if text is None else str(text).casefold()))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate_slices(texts, labels, predictions, probabilities):
    rules = {
        "URL or web link": r"https?\s*:|\bwww\s*\.",
        "Attachment or file mention": r"attach|filename|\.pdf|\.doc|\.xls|\.zip",
        "Forwarded or header-like text": r"forwarded|original message|subject\s*:|from\s*:|sent\s*:",
        "Credential or account wording": r"password|log\s*in|verify|account|bank|otp|passcode|credential",
        "HTML/MIME marker": r"mime-version|content-type|multipart|content-transfer-encoding|<html|<body|base64",
    }
    rows = []
    text_series = pd.Series(texts)
    for title, expression in rules.items():
        mask = text_series.str.contains(expression, regex=True, case=False, na=False).to_numpy()
        count = int(mask.sum())
        if not count:
            rows.append({"name": title, "count": 0, "accuracy": None, "recall": None, "precision": None})
            continue
        truth = labels[mask]
        guess = predictions[mask]
        rows.append({
            "name": title,
            "count": count,
            "accuracy": accuracy_score(truth, guess),
            "recall": recall_score(truth, guess, zero_division=0),
            "precision": precision_score(truth, guess, zero_division=0),
            "phishing_count": int(np.count_nonzero(truth == 1)),
        })
    return rows


def _error_examples(texts, truth, predicted, phishing_probability, kind, limit=8):
    if kind == "False positives":
        indices = np.flatnonzero((truth == 0) & (predicted == 1))
        indices = indices[np.argsort(phishing_probability[indices])[::-1]]
    else:
        indices = np.flatnonzero((truth == 1) & (predicted == 0))
        indices = indices[np.argsort(phishing_probability[indices])]
    if not len(indices):
        return f"**{kind}: 0**\n\n- None in the retained external test rows."
    lines = [f"**{kind}: {len(indices):,}**"]
    for index in indices[:limit]:
        excerpt = " ".join(str(texts[index]).split())[:320].replace("|", "\\|")
        lines.append(f"- Phishing probability {phishing_probability[index]:.2%}: `{excerpt}`")
    return "\n".join(lines)


def _report(data, counts, metrics, matrix, class_report, slice_rows,
            false_positives, false_negatives, source_hash, baseline):
    slice_table = ["| External pattern/format slice | Rows | Phishing rows | Accuracy | Phishing recall | Phishing precision |", "|---|---:|---:|---:|---:|---:|"]
    for row in slice_rows:
        if row["accuracy"] is None:
            slice_table.append(f"| {row['name']} | 0 | 0 | N/A | N/A | N/A |")
        else:
            slice_table.append(
                f"| {row['name']} | {row['count']:,} | {row['phishing_count']:,} | "
                f"{row['accuracy']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} |"
            )

    report = f"""# External Validation Report

**Evaluation date:** {date.today().isoformat()}  
**Selected model:** `comparison_logistic_regression_model.pkl`  
**Saved TF-IDF vectorizer:** `comparison_logistic_regression_tfidf.pkl`  
**Model/vectorizer fitting during this evaluation:** None

## External dataset

The evaluation used the official **test** partition of the Hugging Face dataset [`jmmrcp/phishing-IBM`](https://huggingface.co/datasets/jmmrcp/phishing-IBM), file `test-00000-of-00001.parquet`. The hosted dataset viewer lists 16,672 records across a 15,004-row train partition and a 1,668-row test partition, with a text field (`features`) and integer target (`target`). The test file SHA-256 is `{source_hash}`. The dataset viewer examples show ordinary business email with target 0 and suspicious unsolicited offers with target 1; the external task mapping used here is **0 = legitimate/safe, 1 = phishing**. A 2026 cross-corpus study reports that 30.99% of this repository's full corpus matched its own Dataset A under a 200-character prefix rule. That is not an overlap estimate for this project's training data, but it shows public corpus reuse is possible ([study](https://www.mdpi.com/2079-9202/15/14/3051)).

Only the repository's official test partition was scored. Its train partition was not used to train or tune this model. The external rows were never added to the project training data. The source's feature is already preprocessed text; it is not a raw MIME or `.eml` archive. Target 1 also includes spam-like promotions/scams, so these results reflect the source's broad phishing label and are not a pure credential-phishing benchmark. The hosted README is empty and the dataset card does not clearly state a license; check the source terms before redistributing the files.

## Preprocessing and overlap control

- Raw official test rows: **{counts['raw_rows']:,}**; target counts: **{counts['raw_label_counts']}**. Missing text/labels: {counts['missing_rows']:,}; invalid labels: {counts['invalid_labels']:,}; empty text rows: {counts['empty_rows']:,}.
- Labels were verified to be binary and mapped as 0=legitimate, 1=phishing. Exact duplicate analysis used case-folded alphanumeric word tokens, which removes superficial punctuation-spacing differences. Conflicting-label duplicate texts: {counts['conflicting_duplicate_texts']:,}; duplicate rows removed within the external test set: {counts['duplicate_rows_removed']:,}.
- Training comparisons used only rows marked `train` in `grouped_split_manifest.csv` (65,661 project training emails). Canonical exact matches to those rows: **{counts['exact_training_matches']:,}**, removed before scoring.
- Near-duplicate search used the same project similarity method on canonical word tokens: word-trigram Jaccard ≥0.90 with 32-bit CRC32 SimHash Hamming distance ≤3. This found **{counts['near_training_pairs']:,}** near-match links covering **{counts['near_training_rows']:,}** external rows, which were excluded. Conflicting-label pairs in the similarity graph: {counts['near_label_conflict_pairs']:,}.
- Retained external evaluation set: **{counts['retained_rows']:,}** rows, with labels `{counts['retained_label_counts']}`. The saved vectorizer was loaded and used with `transform()` only; it was never refit. Predictions were generated by the selected saved Logistic Regression model.

This is a conservative overlap screen, not proof that all remaining messages have independent provenance. The external set contains old Enron-style threads and spam examples; shared source collections or lightly edited messages can remain below the chosen similarity threshold.

## Results

| Metric | Previous grouped holdout | External test after overlap removal |
|---|---:|---:|
| Accuracy | {baseline['accuracy']:.4f} | {metrics['accuracy']:.4f} |
| Phishing precision | {baseline['precision']:.4f} | {metrics['precision']:.4f} |
| Phishing recall | {baseline['recall']:.4f} | {metrics['recall']:.4f} |
| Phishing F1 | {baseline['f1']:.4f} | {metrics['f1']:.4f} |
| ROC-AUC | {baseline['roc_auc']:.4f} | {metrics['roc_auc']:.4f} |

The comparison is descriptive: the grouped holdout and external test differ in source, filtering, and label composition. The external score is higher on this retained sample, but the small error count, old/preprocessed corpus, and possible source-family overlap prevent interpreting that as proof of stronger real-world performance.

Confusion matrix (rows=actual, columns=predicted; labels 0=legitimate, 1=phishing):

|  | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Actual legitimate | {matrix[0,0]:,} | {matrix[0,1]:,} |
| Actual phishing | {matrix[1,0]:,} | {matrix[1,1]:,} |

```text
{class_report.rstrip()}
```

## Pattern and format checks

The following slices use labels and text markers already present in the external test partition; they are not synthetic samples. Counts may overlap, and a zero-count slice means this preprocessed corpus does not permit that format to be evaluated.

{chr(10).join(slice_table)}

The corpus contains body/text strings with some forwarded headers, links, and file references, but no separate subject field, raw MIME structure, HTML source, or attachment payload. Therefore, the HTML/MIME slice only measures explicit textual markers and cannot establish resilience to malformed MIME, hidden HTML links, or malicious attachments.

## Error analysis

### False positives

{false_positives}

The retained false positive is a two-word message (`gymnastics pictures`) with little context. Very short messages can be difficult to classify from content alone; review the source label and surrounding email context before using the model to quarantine messages automatically.

### False negatives

{false_negatives}

The retained false negatives include a mortgage/payment unsubscribe message, a quote-heavy message with a vague call to action, a short reply-like mortgage phrase, and a short grant offer. These lack overt credential theft language and several are very short; the broad target-1 label may include unsolicited spam/scams that do not fit a narrow phishing definition. The model sees normalized text only and cannot inspect sender authentication, URL reputation, or attachment contents.

## Limitations and recommendations

1. Treat this as one external-corpus check, not a production guarantee. The remaining **{counts['retained_rows']:,}** records are from a single historical corpus and may share source families with the project data despite exact and high-similarity filtering.
2. Confirm the source's `target=1` policy distinguishes phishing from general spam/scams. Relabel or report these categories separately if the deployment objective is credential theft or malicious-link phishing specifically.
3. Validate on a recent, organization-separated and human-adjudicated email corpus with original subject, headers, URLs, and attachment metadata. Keep sender/domain/thread grouping intact across datasets.
4. Inspect the false negatives and false positives, then measure calibration and operational thresholds on a separate validation set. Do not tune thresholds on this external test report.

## Reproducibility and source links

Run `python external_validation.py` to repeat the analysis. The script reads the external test Parquet, cleaned training data, manifest, and saved artifacts; it writes only this new report. The original dataset and all existing models/vectorizers remain unchanged.

- [Hugging Face dataset card and viewer](https://huggingface.co/datasets/jmmrcp/phishing-IBM)
- [Related cross-corpus study discussing this dataset](https://www.mdpi.com/2079-9202/15/14/3051)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_external_validation():
    for path in (EXTERNAL_TEST_PATH, TRAINING_DATA_PATH, SPLIT_MANIFEST_PATH, MODEL_PATH, VECTORIZER_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required validation input is missing: {path}")

    external = pd.read_parquet(EXTERNAL_TEST_PATH)
    required = {"features", "target"}
    if not required.issubset(external.columns):
        raise ValueError(f"External test file must contain columns {sorted(required)}")
    raw_rows = len(external)
    missing_rows = int(external[["features", "target"]].isna().any(axis=1).sum())
    external = external.dropna(subset=["features", "target"]).copy()
    external["target"] = pd.to_numeric(external["target"], errors="coerce")
    external = external.dropna(subset=["target"])
    invalid = sorted(set(external["target"].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"External target must be 0/1; found {invalid}")
    external["target"] = external["target"].astype(int)
    external["model_text"] = external["features"].map(normalize_text)
    external["canonical"] = external["features"].map(canonical_for_overlap)
    empty_rows = int((external["canonical"] == "").sum())

    conflict_counts = external.groupby("canonical")["target"].nunique()
    conflicting_texts = set(conflict_counts[conflict_counts > 1].index)
    conflicting_duplicate_texts = len(conflicting_texts)
    duplicate_rows_removed = int(external["canonical"].duplicated(keep="first").sum())
    external = external.loc[
        ~external["canonical"].duplicated(keep="first")
        & ~external["canonical"].isin(conflicting_texts)
        & (external["canonical"] != "")
    ].copy().reset_index(drop=True)

    training = pd.read_csv(TRAINING_DATA_PATH, usecols=[TEXT_COLUMN, TARGET_COLUMN])
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    if len(training) != len(manifest) or not np.array_equal(
        manifest["row_index"].to_numpy(), np.arange(len(training))
    ):
        raise ValueError("Training data and grouped manifest are not aligned.")
    if not np.array_equal(training[TARGET_COLUMN].to_numpy(), manifest[TARGET_COLUMN].to_numpy()):
        raise ValueError("Training labels differ from the saved grouped split manifest.")
    training = training.loc[manifest["split"].eq("train")].reset_index(drop=True)
    training_canonical = training[TEXT_COLUMN].map(canonical_for_overlap).tolist()
    training_labels = training[TARGET_COLUMN].astype(int).to_numpy()
    training_canonical_set = set(training_canonical)

    exact_match_mask = external["canonical"].isin(training_canonical_set)
    exact_training_matches = int(exact_match_mask.sum())
    external = external.loc[~exact_match_mask].copy().reset_index(drop=True)
    if external.empty:
        raise ValueError("No external records remain after exact-overlap filtering.")

    combined_text = training_canonical + external["canonical"].tolist()
    combined_labels = np.concatenate([training_labels, external["target"].to_numpy()])
    split_membership = np.concatenate([
        np.zeros(len(training_canonical), dtype=np.int8),
        np.ones(len(external), dtype=np.int8),
    ])
    groups, similarity_stats = build_similarity_groups(
        combined_text, labels=combined_labels, prior_split=split_membership
    )
    near_indices = set()
    for members in similarity_stats["groups"]:
        contains_train = any(index < len(training_canonical) for index in members)
        contains_external = any(index >= len(training_canonical) for index in members)
        if contains_train and contains_external:
            near_indices.update(
                index - len(training_canonical)
                for index in members if index >= len(training_canonical)
            )
    near_training_rows = len(near_indices)
    near_training_pairs = similarity_stats["cross_split_pairs"]
    external["overlap_index"] = np.arange(len(external))
    external = external.loc[~external["overlap_index"].isin(near_indices)].copy().reset_index(drop=True)
    if external[EXTERNAL_TARGET_COLUMN].nunique() < 2:
        raise ValueError("Both labels must remain after overlap filtering to calculate ROC-AUC.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Unexpected saved-model classes: {model.classes_}")
    text_to_score = external["model_text"].tolist()
    features = vectorizer.transform(text_to_score)  # use saved vocabulary; never fit here
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    truth = external[EXTERNAL_TARGET_COLUMN].to_numpy()
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision_score(truth, predictions, zero_division=0),
        "recall": recall_score(truth, predictions, zero_division=0),
        "f1": f1_score(truth, predictions, zero_division=0),
        "roc_auc": roc_auc_score(truth, probabilities),
    }
    class_report = classification_report(
        truth, predictions, labels=[0, 1],
        target_names=["Legitimate Email", "Phishing Email"], zero_division=0,
    )
    slices = _evaluate_slices(text_to_score, truth, predictions, probabilities)
    errors_fp = _error_examples(text_to_score, truth, predictions, probabilities, "False positives")
    errors_fn = _error_examples(text_to_score, truth, predictions, probabilities, "False negatives")

    prior_results = pd.read_csv(COMPARISON_RESULTS_PATH)
    baseline_row = prior_results.loc[prior_results["name"].eq("Logistic Regression")].iloc[0]
    baseline = {name: float(baseline_row[name]) for name in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    counts = {
        "raw_rows": raw_rows,
        "raw_label_counts": pd.Series(
            pd.read_parquet(EXTERNAL_TEST_PATH)["target"].astype(int)
        ).value_counts().sort_index().to_dict(),
        "missing_rows": missing_rows,
        "invalid_labels": len(invalid),
        "empty_rows": empty_rows,
        "conflicting_duplicate_texts": conflicting_duplicate_texts,
        "duplicate_rows_removed": duplicate_rows_removed,
        "exact_training_matches": exact_training_matches,
        "near_training_pairs": near_training_pairs,
        "near_training_rows": near_training_rows,
        "near_label_conflict_pairs": similarity_stats["label_conflict_pairs"],
        "retained_rows": len(external),
        "retained_label_counts": external[EXTERNAL_TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }
    _report(
        external, counts, metrics, matrix, class_report, slices,
        errors_fp, errors_fn, _sha256(EXTERNAL_TEST_PATH), baseline,
    )
    print(f"External test rows: {raw_rows:,}; exact training overlaps removed: {exact_training_matches:,}")
    print(f"Near-overlap external rows removed: {near_training_rows:,}; retained: {len(external):,}")
    print("Metrics:", {key: round(value, 6) for key, value in metrics.items()})
    print("Confusion matrix (actual rows, predicted columns; labels 0, 1):")
    print(matrix)
    print(f"Report saved to: {REPORT_PATH}")
    return metrics


if __name__ == "__main__":
    run_external_validation()
