"""Held-out evaluation helpers for the phishing classifier."""
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_model(model, vectorizer, test_texts, y_test):
    """Print standard metrics and return them as a dictionary."""
    features = vectorizer.transform(test_texts)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Legitimate Email", "Phishing Email"], zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_test, probabilities) if len(set(y_test)) == 2 else None,
    }
    print("\nHoldout evaluation (positive class: phishing / label 1)")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"{metric.title()}: {results[metric]:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted; labels 0, 1):")
    print(results["confusion_matrix"])
    print("Classification report:")
    print(results["classification_report"])
    if results["roc_auc"] is not None:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
    return results
