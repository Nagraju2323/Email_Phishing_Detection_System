"""Estimator and training-only search space for Multinomial Naive Bayes."""
from sklearn.naive_bayes import MultinomialNB


def build_model():
    estimator = MultinomialNB()
    grid = {"classifier__alpha": [0.1, 0.5, 1.0]}
    return estimator, grid
