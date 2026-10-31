"""Shared leakage-safe model tuning, evaluation, timing, and SVG charts."""
from html import escape
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MAX_FEATURES = 200_000


def make_pipeline(estimator, cache_memory=None):
    """Build a fresh TF-IDF + estimator pipeline; CV fits TF-IDF per fold."""
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,  # input strings are normalized before splitting
        max_features=MAX_FEATURES,
        dtype=np.float64,
    )
    return Pipeline(
        [("tfidf", vectorizer), ("classifier", estimator)], memory=cache_memory
    )


def tune_fit_evaluate(name, estimator, param_grid, x_train, y_train,
                      train_groups, x_test, y_test, cache_memory=None):
    """Tune with group-aware CV, refit only on train data, then test once."""
    pipeline = make_pipeline(estimator, cache_memory=cache_memory)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        refit=False,
        n_jobs=1,
        pre_dispatch=1,
        return_train_score=False,
        error_score="raise",
        verbose=1,
    )
    print(f"\nTuning {name} with {cv.get_n_splits()} group-aware folds...")
    tuning_start = perf_counter()
    search.fit(x_train, y_train, groups=train_groups)
    tuning_seconds = perf_counter() - tuning_start

    classifier_params = {
        key.removeprefix("classifier__"): value
        for key, value in search.best_params_.items()
    }
    final_estimator = clone(estimator).set_params(**classifier_params)
    final_pipeline = make_pipeline(final_estimator, cache_memory=cache_memory)
    training_start = perf_counter()
    final_pipeline.fit(x_train, y_train)
    training_seconds = perf_counter() - training_start

    prediction_start = perf_counter()
    probabilities = final_pipeline.predict_proba(x_test)[:, 1]
    prediction_seconds = perf_counter() - prediction_start
    predictions = (probabilities >= 0.5).astype(np.int8)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=1)
    result = {
        "name": name,
        "best_params": search.best_params_,
        "cv_best_f1": float(search.best_score_),
        "tuning_seconds": tuning_seconds,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_ms_per_email": prediction_seconds * 1000 / len(x_test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": matrix,
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Legitimate Email", "Phishing Email"], zero_division=0,
        ),
        "fpr": fpr,
        "tpr": tpr,
        "pipeline": final_pipeline,
        "predictions": predictions,
        "probabilities": probabilities,
    }
    return result


def save_separate_artifacts(result, root):
    """Save classifier and fitted vectorizer as separate Joblib artifacts."""
    slug = {
        "Logistic Regression": "logistic_regression",
        "Multinomial Naive Bayes": "multinomial_nb",
        "Random Forest": "random_forest",
    }[result["name"]]
    pipeline = result["pipeline"]
    model_path = Path(root) / f"comparison_{slug}_model.pkl"
    vectorizer_path = Path(root) / f"comparison_{slug}_tfidf.pkl"
    joblib.dump(pipeline.named_steps["classifier"], model_path)
    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)
    result["model_path"] = model_path.name
    result["vectorizer_path"] = vectorizer_path.name


def _svg_header(title, width=900, height=520):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="bold">{escape(title)}</text>',
    ]


def _save_bar_chart(results, metrics, title, path):
    width, height = 900, 520
    left, top, plot_w, plot_h = 85, 70, 760, 350
    bottom = top + plot_h
    colors = ["#2563eb", "#16a34a", "#ea580c"]
    svg = _svg_header(title, width, height)
    for tick in np.linspace(0, 1, 6):
        y = bottom - tick * plot_h
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#d1d5db"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.1f}</text>')
    group_w = plot_w / len(metrics)
    bar_w = min(42, group_w / (len(results) + 1))
    for metric_index, metric in enumerate(metrics):
        center = left + group_w * (metric_index + 0.5)
        for model_index, result in enumerate(results):
            value = result[metric]
            x = center + (model_index - (len(results)-1)/2) * (bar_w + 8) - bar_w/2
            y = bottom - value * plot_h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{value*plot_h:.1f}" fill="{colors[model_index]}"/>')
            svg.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3f}</text>')
        svg.append(f'<text x="{center:.1f}" y="{bottom+28}" text-anchor="middle" font-family="Arial" font-size="13">{escape(metric.title())}</text>')
    legend_y = 480
    legend_width = 270
    start_x = (width - legend_width * len(results)) / 2
    for i, result in enumerate(results):
        x = start_x + i * legend_width
        svg.append(f'<rect x="{x:.1f}" y="{legend_y-12}" width="14" height="14" fill="{colors[i]}"/>')
        svg.append(f'<text x="{x+20:.1f}" y="{legend_y}" font-family="Arial" font-size="12">{escape(result["name"])}</text>')
    svg.append('</svg>')
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def _save_confusion_matrices(results, path):
    width, height = 1000, 390
    svg = _svg_header("Confusion Matrices (actual rows, predicted columns)", width, height)
    panel_w = width / 3
    for index, result in enumerate(results):
        matrix = result["confusion_matrix"]
        maximum = max(1, int(matrix.max()))
        origin_x = index * panel_w + 85
        origin_y = 105
        cell = 115
        svg.append(f'<text x="{index*panel_w+panel_w/2:.1f}" y="70" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">{escape(result["name"])}</text>')
        svg.append(f'<text x="{origin_x+cell:.1f}" y="{origin_y-12}" text-anchor="middle" font-family="Arial" font-size="11">Predicted</text>')
        svg.append(f'<text x="{origin_x-20}" y="{origin_y+cell}" text-anchor="middle" font-family="Arial" font-size="11" transform="rotate(-90 {origin_x-20} {origin_y+cell})">Actual</text>')
        for row in range(2):
            for column in range(2):
                value = int(matrix[row, column])
                intensity = 0.12 + 0.70 * value / maximum
                fill = f"rgb({int(219-80*intensity)},{int(234-35*intensity)},{int(254-5*intensity)})"
                x, y = origin_x + column * cell, origin_y + row * cell
                svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="3"/>')
                svg.append(f'<text x="{x+cell/2}" y="{y+cell/2+6}" text-anchor="middle" font-family="Arial" font-size="18">{value:,}</text>')
                svg.append(f'<text x="{x+cell/2}" y="{origin_y+2*cell+22}" text-anchor="middle" font-family="Arial" font-size="11">{["Legit", "Phish"][column]}</text>')
            svg.append(f'<text x="{origin_x-8}" y="{origin_y+row*cell+cell/2+4}" text-anchor="end" font-family="Arial" font-size="11">{["Legit", "Phish"][row]}</text>')
    svg.append('</svg>')
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def _save_roc_curves(results, path):
    width, height = 820, 620
    left, top, plot_w, plot_h = 90, 70, 600, 470
    colors = ["#2563eb", "#16a34a", "#ea580c"]
    svg = _svg_header("ROC Curves (group-held-out test set)", width, height)
    for tick in np.linspace(0, 1, 6):
        x = left + tick * plot_w
        y = top + plot_h - tick * plot_h
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#d1d5db"/>')
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#d1d5db"/>')
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.1f}</text>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.1f}</text>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#6b7280" stroke-dasharray="6,6"/>')
    for index, result in enumerate(results):
        fpr, tpr = result["fpr"], result["tpr"]
        stride = max(1, len(fpr)//1500)
        points = []
        for xval, yval in zip(fpr[::stride], tpr[::stride]):
            x = left + xval * plot_w
            y = top + plot_h - yval * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        if points[-1] != f"{left+plot_w:.1f},{top:.1f}":
            points.append(f"{left+plot_w:.1f},{top:.1f}")
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[index]}" stroke-width="2.5"/>')
        legend_y = 95 + index * 26
        svg.append(f'<line x1="720" y1="{legend_y}" x2="750" y2="{legend_y}" stroke="{colors[index]}" stroke-width="3"/>')
        svg.append(f'<text x="758" y="{legend_y+4}" font-family="Arial" font-size="12">{escape(result["name"])} (AUC {result["roc_auc"]:.4f})</text>')
    svg.append(f'<text x="{left+plot_w/2}" y="590" text-anchor="middle" font-family="Arial" font-size="14">False Positive Rate</text>')
    svg.append(f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 25 {top+plot_h/2})">True Positive Rate</text>')
    svg.append('</svg>')
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def create_comparison_charts(results, root):
    root = Path(root)
    _save_bar_chart(results, ["accuracy"], "Accuracy by Algorithm", root / "comparison_accuracy.svg")
    _save_bar_chart(results, ["precision", "recall", "f1"], "Precision, Recall, and F1", root / "comparison_precision_recall_f1.svg")
    _save_confusion_matrices(results, root / "comparison_confusion_matrices.svg")
    _save_roc_curves(results, root / "comparison_roc_curves.svg")
