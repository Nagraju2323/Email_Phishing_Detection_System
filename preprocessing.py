"""Text normalization and train-only TF-IDF feature extraction."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text):
    """Normalize case and spacing while retaining URLs and security terms."""
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def create_vectorizer():
    """Create word-level TF-IDF features without discarding stop words."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,  # normalize_text handles lowercasing consistently
        max_features=200_000,
        dtype="float32",
    )
