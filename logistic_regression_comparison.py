"""Estimator and training-only search space for Logistic Regression."""
from sklearn.linear_model import LogisticRegression


def build_model():
    estimator = LogisticRegression(
        max_iter=2000, class_weight=None, solver="liblinear", random_state=42
    )
    grid = {"classifier__C": [0.25, 1.0, 4.0]}
    return estimator, grid
