"""Train/evaluate a separate Logistic Regression model with grouped splitting."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data_cleaning import DEFAULT_DATASET, TARGET_COLUMN, TEXT_COLUMN
from evaluate_model import evaluate_model
from group_split import build_similarity_groups, choose_stratified_group_split
from preprocessing import create_vectorizer, normalize_text

ROOT = Path(__file__).resolve().parent
CLEANED_DATASET = ROOT / "cleaned_dataset.csv"
GROUPED_MODEL_PATH = ROOT / "logistic_regression_grouped_model.pkl"
GROUPED_VECTORIZER_PATH = ROOT / "tfidf_vectorizer_grouped.pkl"
SPLIT_MANIFEST_PATH = ROOT / "grouped_split_manifest.csv"
REPORT_PATH = ROOT / "generalization_evaluation_report.md"
RANDOM_STATE = 42


def _evaluate(model, vectorizer, texts, labels):
    features = vectorizer.transform(texts)
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities) if len(set(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]),
        "classification_report": classification_report(
            labels, predicted, labels=[0, 1],
            target_names=["Legitimate Email", "Phishing Email"], zero_division=0,
        ),
        "predicted": predicted,
        "probabilities": probabilities,
    }


def _comparison_rows(original, grouped):
    rows = []
    for title, key in (
        ("Accuracy", "accuracy"),
        ("Phishing precision", "precision"),
        ("Phishing recall", "recall"),
        ("Phishing F1", "f1"),
        ("ROC-AUC", "roc_auc"),
    ):
        left = "N/A" if original[key] is None else f"{original[key]:.4f}"
        right = "N/A" if grouped[key] is None else f"{grouped[key]:.4f}"
        rows.append(f"| {title} | {left} | {right} |")
    return "\n".join(rows)


def _confusion_table(matrix):
    return (
        "|  | Predicted legitimate | Predicted phishing |\n"
        "|---|---:|---:|\n"
        f"| Actual legitimate | {matrix[0, 0]:,} | {matrix[0, 1]:,} |\n"
        f"| Actual phishing | {matrix[1, 0]:,} | {matrix[1, 1]:,} |\n"
    )


def _format_errors(texts, labels, result, kind, limit=6):
    predicted = result["predicted"]
    probabilities = result["probabilities"]
    labels = np.asarray(labels)
    if kind == "false positives":
        indices = np.flatnonzero((labels == 0) & (predicted == 1))
        indices = indices[np.argsort(probabilities[indices])[::-1]]
    else:
        indices = np.flatnonzero((labels == 1) & (predicted == 0))
        indices = indices[np.argsort(probabilities[indices])]
    lines = [f"**{kind.title()}: {len(indices):,}**"]
    if not len(indices):
        return "\n".join(lines + ["- None in this test split."])
    for index in indices[:limit]:
        excerpt = " ".join(str(texts[index]).split())[:320].replace("|", "\\|")
        lines.append(f"- Phishing probability {probabilities[index]:.2%}: `{excerpt}`")
    return "\n".join(lines)


def _write_report(data, groups, split_stats, train_indices, test_indices,
                  original_result, grouped_result, x_test, y_test):
    grouped_labels = data[TARGET_COLUMN].to_numpy()
    sizes = [(len(members), members) for members in split_stats["groups"]]
    sizes.sort(reverse=True, key=lambda item: item[0])
    family_lines = []
    for size, members in sizes[:6]:
        text = " ".join(str(data.iloc[members[0]][TEXT_COLUMN]).split())[:220]
        family_lines.append(f"- {size:,} messages, label {int(grouped_labels[members[0]])}: `{text}`")

    group_sizes = np.bincount(groups)
    train_labels = grouped_labels[train_indices]
    test_labels = grouped_labels[test_indices]
    train_counts = pd.Series(train_labels).value_counts().sort_index().to_dict()
    test_counts = pd.Series(test_labels).value_counts().sort_index().to_dict()
    original_cm = original_result["confusion_matrix"]
    grouped_cm = grouped_result["confusion_matrix"]
    original_test_texts = original_result["test_texts"]
    original_test_labels = original_result["test_labels"]
    report = f"""# Group-Based Generalization Evaluation

**Run date:** 2026-10-31  
**Source dataset:** `cleaned_dataset.csv` (original CSV preserved)  
**Label mapping:** 0 = legitimate, 1 = phishing  
**Random seed:** {RANDOM_STATE}

## Summary

The original random-split model and the new near-duplicate-grouped model were evaluated on their respective holdout splits. The original saved model was not modified. The new model and TF-IDF vectorizer were saved separately. Grouping kept identical and highly similar messages together before selecting the grouped holdout.

The grouped split contains {len(train_indices):,} training rows and {len(test_indices):,} test rows ({len(test_indices)/len(data):.2%} test). Training class counts are {train_counts}; test class counts are {test_counts}. The grouped split contains {len(set(groups)):,} indivisible groups, with {int(group_sizes.max()):,} rows in its largest group.

## Near-duplicate analysis

- Similarity used normalized lowercase text split into word trigrams. A deterministic 32-bit CRC32 SimHash, indexed as four 8-bit bands, found candidates. Candidates were verified with word-trigram Jaccard similarity ≥ {0.90:.2f} and SimHash Hamming distance ≤ 3 bits.
- All verified qualifying pairs were joined as graph edges; connected components (including transitive variants) became indivisible groups. This deliberately keeps a template family together even when a chain of small edits connects its messages.
- Across the cleaned dataset, the scan found **{split_stats['qualifying_pairs']:,} qualifying pairs**, forming **{split_stats['group_count']:,} groups**; {split_stats['singleton_groups']:,} groups were singletons. No verified pair had conflicting labels ({split_stats['label_conflict_pairs']:,} conflicting-label pairs).
- Rechecking the previous random split with this full-dataset grouping rule found **{split_stats['cross_split_test_rows']:,} test rows** connected to training messages by **{split_stats['cross_split_pairs']:,} qualifying pairs**. The earlier audit reported 781 under its prior candidate-scan implementation; the current count is based on the explicit deterministic grouping rule above.
- The three largest components contain {sizes[0][0]:,}, {sizes[1][0]:,}, and {sizes[2][0]:,} messages. Their representative texts and nearby rows are repeated “Daily Top 10” newsletter-style messages with CNN video/story links. Other repeated groups include bulk software offers, stock promotions, grant solicitations, and adult-content spam. These are template families, not confirmed threat-actor campaigns.
- Largest detected families (representative text excerpt):
{chr(10).join(family_lines)}
- The dataset contains message text and a binary label, but no sender, message ID, received date, or original URL-domain metadata. Attribution to a specific campaign or actor cannot be established from these data.

## Data leakage prevention

1. The existing `cleaned_dataset.csv` was read as input; the source CSV and existing artifacts were not overwritten.
2. Exact duplicates had already been removed during cleaning. Similarity grouping was performed before splitting using text only and no fitted TF-IDF vocabulary.
3. A seeded `StratifiedGroupKFold` generated candidate folds. The selected fold minimized deviation from the 20% target test size and the full dataset's two-class proportions. A runtime check confirmed no group ID occurs in both partitions.
4. TF-IDF was fitted only on grouped training text. The grouped test data were transformed using that fitted vectorizer.

## Performance comparison

Original saved-model results are recomputed on the original 80:20 stratified random split. Grouped results are from the new model on the group-held-out split. Metrics are not directly equivalent because the test sets differ. Group-based accuracy is 0.13 percentage points higher on this selected fold; this is not evidence that the new model is intrinsically better. It shows that performance did not fall on this particular grouped holdout, and should be confirmed with repeated grouped folds or a source-separated test set.

| Metric | Original random split | Group-based split |
|---|---:|---:|
{_comparison_rows(original_result, grouped_result)}
### Confusion matrices

Original random split (rows = actual; columns = predicted; order 0=legitimate, 1=phishing):

{_confusion_table(original_cm)}
Group-based split (same order):

{_confusion_table(grouped_cm)}

### Classification reports

Original random split:

```text
{original_result['classification_report'].rstrip()}
```

Group-based split:

```text
{grouped_result['classification_report'].rstrip()}
```

## Error analysis

### Original random-split errors

{_format_errors(original_test_texts, original_test_labels, original_result, 'false positives')}\n\n{_format_errors(original_test_texts, original_test_labels, original_result, 'false negatives')}

### Group-based test errors

{_format_errors(x_test, y_test, grouped_result, 'false positives')}\n\n{_format_errors(x_test, y_test, grouped_result, 'false negatives')}

Compare the excerpts with the dataset's intended label definition. In the earlier random split, false positives included legitimate account-registration and link-heavy notices; false negatives included discussion, invitation, and solicitation-style messages whose wording is less overtly credential-focused. The grouped examples below show the corresponding generalization errors when related templates are held out together.

## Recommendations

1. Use the group-held-out metrics as a template-disjoint evaluation for this dataset, but do not claim a model improvement from the small score increase. The connected-component split does not simulate time-based deployment by itself.
2. Review the largest template groups and error excerpts for label quality. The near-duplicate components are highly imbalanced in size, and several large label-1 clusters are newsletter or spam templates rather than confirmed phishing campaigns.
3. Keep a separate temporal or source-separated test set for final evaluation. Add sender, URL-domain, attachment, and message-thread metadata only when available at inference and represented safely in training.
4. Evaluate precision/recall tradeoffs and probability calibration on adjudicated real-world mail before choosing operational blocking thresholds.

## Artifacts

- Grouped model: `logistic_regression_grouped_model.pkl`
- Grouped TF-IDF vectorizer: `tfidf_vectorizer_grouped.pkl`
- Row/group/split manifest: `grouped_split_manifest.csv`
- Existing model files remain unchanged.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def train_grouped_pipeline(dataset_path=CLEANED_DATASET):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {dataset_path}. Run data_cleaning.py first."
        )
    data = pd.read_csv(dataset_path)
    if not {TEXT_COLUMN, TARGET_COLUMN}.issubset(data.columns):
        raise ValueError(f"Dataset must contain {TEXT_COLUMN!r} and {TARGET_COLUMN!r}.")

    texts = data[TEXT_COLUMN].map(normalize_text)
    labels = data[TARGET_COLUMN].astype(int).to_numpy()

    # Recreate the original random split solely to quantify its near-duplicate overlap.
    old_train, old_test = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=RANDOM_STATE,
        stratify=labels,
    )
    old_membership = np.zeros(len(data), dtype=np.int8)
    old_membership[old_test] = 1
    groups, group_stats = build_similarity_groups(
        texts.tolist(), labels=labels, prior_split=old_membership
    )
    train_indices, test_indices = choose_stratified_group_split(labels, groups)

    # Load and re-evaluate the original artifacts on their exact original split.
    original_model = joblib.load(ROOT / "logistic_regression_model.pkl")
    original_vectorizer = joblib.load(ROOT / "tfidf_vectorizer.pkl")
    original_result = _evaluate(
        original_model, original_vectorizer, texts.iloc[old_test].tolist(), labels[old_test]
    )
    original_result["test_texts"] = texts.iloc[old_test].tolist()
    original_result["test_labels"] = labels[old_test]

    x_train, x_test = texts.iloc[train_indices].tolist(), texts.iloc[test_indices].tolist()
    y_train, y_test = labels[train_indices], labels[test_indices]
    vectorizer = create_vectorizer()
    train_features = vectorizer.fit_transform(x_train)
    model = LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE)
    model.fit(train_features, y_train)
    grouped_result = _evaluate(model, vectorizer, x_test, y_test)

    joblib.dump(model, GROUPED_MODEL_PATH)
    joblib.dump(vectorizer, GROUPED_VECTORIZER_PATH)
    manifest = pd.DataFrame({
        "row_index": np.arange(len(data)),
        "group_id": groups,
        "split": np.where(np.isin(np.arange(len(data)), test_indices), "test", "train"),
        "label": labels,
    })
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)
    _write_report(
        data, groups, group_stats, train_indices, test_indices,
        original_result, grouped_result, x_test, y_test,
    )

    print(f"Verified similar pairs: {group_stats['qualifying_pairs']:,}")
    print(f"Near-duplicate groups: {group_stats['group_count']:,}; largest: {group_stats['largest_group']:,}")
    print(f"Prior split test rows with near-duplicate train counterpart: {group_stats['cross_split_test_rows']:,}")
    print(f"Grouped split: train={len(train_indices):,}, test={len(test_indices):,}")
    print("Grouped holdout metrics:")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name}: {grouped_result[name]:.4f}" if grouped_result[name] is not None else f"{name}: N/A")
    print("Grouped confusion matrix (rows=actual, columns=predicted; labels 0, 1):")
    print(grouped_result["confusion_matrix"])
    print(grouped_result["classification_report"])
    print(f"New model saved to: {GROUPED_MODEL_PATH}")
    print(f"New vectorizer saved to: {GROUPED_VECTORIZER_PATH}")
    print(f"Split manifest saved to: {SPLIT_MANIFEST_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    return model, vectorizer, grouped_result


if __name__ == "__main__":
    train_grouped_pipeline()
