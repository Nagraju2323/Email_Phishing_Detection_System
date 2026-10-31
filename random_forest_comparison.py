"""Estimator and bounded search space for Random Forest on sparse TF-IDF."""
from sklearn.ensemble import RandomForestClassifier


def build_model():
    estimator = RandomForestClassifier(
        n_estimators=75,
        max_features="sqrt",
        class_weight=None,
        n_jobs=1,
        random_state=42,
    )
    grid = {
        "classifier__max_depth": [30, None],
        "classifier__min_samples_leaf": [1, 3],
    }
    return estimator, grid
