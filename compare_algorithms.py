"""Fair grouped-holdout comparison of Logistic Regression, Multinomial NB, RF."""
import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import StratifiedGroupKFold

from common_evaluation import (
    RANDOM_STATE,
    create_comparison_charts,
    save_separate_artifacts,
    tune_fit_evaluate,
)
from data_cleaning import TARGET_COLUMN, TEXT_COLUMN
from logistic_regression_comparison import build_model as build_logistic
from multinomial_nb_comparison import build_model as build_nb
from preprocessing import normalize_text
from random_forest_comparison import build_model as build_forest

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cleaned_dataset.csv"
MANIFEST_PATH = ROOT / "grouped_split_manifest.csv"
REPORT_PATH = ROOT / "algorithm_comparison_report.md"
RESULTS_PATH = ROOT / "algorithm_comparison_results.csv"


def _write_report(data, manifest, results, train_indices, test_indices):
    rows = []
    for result in results:
        rows.append(
            "| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {roc_auc:.4f} | {training_seconds:.2f} | "
            "{prediction_ms_per_email:.3f} |".format(**result)
        )
    matrix_sections = []
    for result in results:
        matrix = result["confusion_matrix"]
        matrix_sections.append(
            f"### {result['name']}\n\n"
            "Rows are actual classes; columns are predicted classes (0=legitimate, 1=phishing).\n\n"
            "|  | Predicted legitimate | Predicted phishing |\n"
            "|---|---:|---:|\n"
            f"| Actual legitimate | {matrix[0,0]:,} | {matrix[0,1]:,} |\n"
            f"| Actual phishing | {matrix[1,0]:,} | {matrix[1,1]:,} |\n\n"
            "```text\n" + result["classification_report"].rstrip() + "\n```"
        )
    params = "\n".join(
        f"- **{r['name']}** (best group-CV F1 {r['cv_best_f1']:.4f}): "
        f"`{json.dumps(r['best_params'], sort_keys=True, default=str)}`"
        for r in results
    )
    best = max(results, key=lambda r: (r["f1"], r["recall"], r["precision"]))
    grouped = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    train_counts = grouped.loc["train"].to_dict()
    test_counts = grouped.loc["test"].to_dict()
    group_count = manifest["group_id"].nunique()
    group_test_fraction = len(test_indices) / len(data)
    report = f"""# Comparative Algorithm Evaluation: Email Phishing Detection

**Run date:** {date.today().isoformat()}  
**Dataset:** `cleaned_dataset.csv` (original source CSV was not modified)  
**Rows:** {len(data):,}; features: `{TEXT_COLUMN}`; target: `{TARGET_COLUMN}` (0=legitimate, 1=phishing)  
**Grouped split seed:** {RANDOM_STATE}; **group-aware CV folds:** 3

## Dataset and experimental methodology

- Reused `grouped_split_manifest.csv` for every model. Training rows: {len(train_indices):,}; holdout rows: {len(test_indices):,} ({group_test_fraction:.2%}). Training labels: `{train_counts}`; holdout labels: `{test_counts}`.
- The manifest contains {group_count:,} similarity groups. A validation check confirmed zero group IDs overlap between train and test. All three algorithms received exactly the same text and labels in each partition.
- Text normalization lowercases and collapses whitespace without removing URL/security wording. Each pipeline uses word-level TF-IDF with unigrams and bigrams, no stop-word removal, and a 200,000-feature cap.
- Hyperparameters were selected by 3-fold `StratifiedGroupKFold` on training records only (seed {RANDOM_STATE}), optimizing phishing F1. Each CV fold had disjoint group IDs, and TF-IDF lived inside the pipeline, so it was fit only on that fold's training text. The grouped holdout was used only for final evaluation.
- After tuning, the best configuration was refit on the complete grouped training partition. Reported training time is this final TF-IDF plus model fit; tuning time is also saved in `algorithm_comparison_results.csv`. Prediction time includes TF-IDF transform and `predict_proba` for the holdout, reported per email.
- All models use default unweighted classes because training classes are close in size. The Naive Bayes smoothing value and model-specific parameters were tuned using CV, not the holdout.

## Model configurations

{params}

Search grids were intentionally bounded for a reproducible comparison. Logistic Regression searched `C` in {{0.25, 1, 4}}; Multinomial Naive Bayes searched `alpha` in {{0.1, 0.5, 1}}; Random Forest used 75 trees and searched `max_depth` in {{30, None}} and `min_samples_leaf` in {{1, 3}}, with `max_features='sqrt'`. Search selection used training-fold F1 only.

## Performance comparison

| Algorithm | Accuracy | Precision | Recall | F1 | ROC-AUC | Final fit (s) | Prediction (ms/email) |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

**Tuning time (seconds):** {"; ".join(f"{r['name']}: {r['tuning_seconds']:.2f}" for r in results)}.  
**Selected by holdout F1 for recommendation:** {best['name']} (F1 {best['f1']:.4f}, recall {best['recall']:.4f}, ROC-AUC {best['roc_auc']:.4f}). This is a comparison among the tested candidates, not an independent estimate beyond this one grouped holdout.

### Accuracy

![Accuracy comparison](comparison_accuracy.svg)

### Precision, recall, and F1

![Precision, recall, and F1 comparison](comparison_precision_recall_f1.svg)

### Confusion matrices

![Confusion matrices for the three algorithms](comparison_confusion_matrices.svg)

{chr(10).join(matrix_sections)}

### ROC curves

![ROC curves for the three algorithms](comparison_roc_curves.svg)

## Advantages and limitations

- **Logistic Regression:** suited to sparse high-dimensional TF-IDF and provides a fast, interpretable linear baseline. It can miss nonlinear feature interactions and its probabilities still require calibration checks.
- **Multinomial Naive Bayes:** simple and usually fast for word-count/TF-IDF features; useful as a lightweight baseline. Its conditional-independence assumption is unrealistic for language, and probabilities can be poorly calibrated.
- **Random Forest:** can model nonlinear feature interactions and offers feature-importance tools. On sparse text with many mostly-zero features it can be slower, larger, and less effective than linear models; its probabilities may need calibration.

Measured fit and prediction times above should guide operational tradeoffs for this dataset and machine, rather than general algorithm claims.

## Recommendation

The tested candidate with the highest holdout F1 is **{best['name']}**. The evidence is the grouped-holdout F1 of {best['f1']:.4f}, phishing recall of {best['recall']:.4f}, ROC-AUC of {best['roc_auc']:.4f}, and measured timing in the table. Before deployment, confirm that its false-positive and false-negative costs are acceptable, review its error examples, and validate on a separate temporal or source-separated dataset. The tuned model comparison must not be interpreted as proof of future performance from one holdout alone.

## Saved artifacts

Each model and its fitted vectorizer are separate Joblib files:

{chr(10).join(f"- `{r['model_path']}` and `{r['vectorizer_path']}`" for r in results)}

Charts are generated as SVG without a plotting-library dependency. Detailed machine-readable results, including best parameters and tuning times, are in `algorithm_comparison_results.csv`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_comparison():
    if not DATA_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Expected cleaned_dataset.csv and grouped_split_manifest.csv in the project directory."
        )
    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    if len(data) != len(manifest):
        raise ValueError("Split manifest row count does not match the cleaned dataset.")
    if not np.array_equal(manifest["row_index"].to_numpy(), np.arange(len(data))):
        raise ValueError("Manifest row_index is not aligned with the cleaned dataset.")
    labels = data[TARGET_COLUMN].astype(int).to_numpy()
    if not np.array_equal(labels, manifest[TARGET_COLUMN].to_numpy()):
        raise ValueError("Manifest labels do not match the cleaned dataset.")

    train_indices = np.flatnonzero(manifest["split"].to_numpy() == "train")
    test_indices = np.flatnonzero(manifest["split"].to_numpy() == "test")
    groups = manifest["group_id"].to_numpy()
    if set(groups[train_indices]).intersection(groups[test_indices]):
        raise ValueError("The saved split manifest contains group leakage.")

    texts = data[TEXT_COLUMN].map(normalize_text).to_numpy()
    x_train, x_test = texts[train_indices], texts[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    train_groups = groups[train_indices]

    # Check every inner-CV split too, before launching expensive searches.
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for fold_train, fold_valid in cv.split(x_train, y_train, train_groups):
        if set(train_groups[fold_train]).intersection(train_groups[fold_valid]):
            raise ValueError("A group crosses an inner CV fold.")

    configurations = [
        ("Logistic Regression", build_logistic()),
        ("Multinomial Naive Bayes", build_nb()),
        ("Random Forest", build_forest()),
    ]
    results = []
    # Temporary joblib cache reuses the exact same fold-specific TF-IDF fit
    # across hyperparameter candidates and algorithms; it is deleted at exit.
    with TemporaryDirectory(prefix="phishing_tfidf_cv_") as cache_directory:
        cache_memory = Memory(location=cache_directory, verbose=0)
        for name, (estimator, grid) in configurations:
            result = tune_fit_evaluate(
                name, estimator, grid, x_train, y_train, train_groups, x_test, y_test,
                cache_memory=cache_memory,
            )
            save_separate_artifacts(result, ROOT)
            print(
                f"{name}: accuracy={result['accuracy']:.4f}, precision={result['precision']:.4f}, "
                f"recall={result['recall']:.4f}, F1={result['f1']:.4f}, "
                f"ROC-AUC={result['roc_auc']:.4f}, final_fit={result['training_seconds']:.2f}s, "
                f"prediction={result['prediction_ms_per_email']:.3f}ms/email",
                flush=True,
            )
            results.append(result)

    create_comparison_charts(results, ROOT)
    result_rows = []
    for result in results:
        result_rows.append({
            key: result[key]
            for key in (
                "name", "accuracy", "precision", "recall", "f1", "roc_auc",
                "cv_best_f1", "tuning_seconds", "training_seconds",
                "prediction_seconds", "prediction_ms_per_email", "best_params",
                "model_path", "vectorizer_path",
            )
        })
    results_frame = pd.DataFrame(result_rows)
    results_frame["best_params"] = results_frame["best_params"].map(
        lambda value: json.dumps(value, sort_keys=True, default=str)
    )
    results_frame.to_csv(RESULTS_PATH, index=False)
    _write_report(data, manifest, results, train_indices, test_indices)
    print(f"Comparison report: {REPORT_PATH}")
    print(f"Machine-readable results: {RESULTS_PATH}")
    print("SVG charts: comparison_accuracy.svg, comparison_precision_recall_f1.svg, comparison_confusion_matrices.svg, comparison_roc_curves.svg")
    return results


if __name__ == "__main__":
    run_comparison()
