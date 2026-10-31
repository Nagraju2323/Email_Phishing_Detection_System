"""Train, evaluate, and save the Logistic Regression phishing pipeline."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data_cleaning import clean_dataset, DEFAULT_DATASET, TEXT_COLUMN, TARGET_COLUMN
from evaluate_model import evaluate_model
from preprocessing import create_vectorizer, normalize_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "logistic_regression_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.pkl"
RANDOM_STATE = 42


def train_pipeline(input_path=DEFAULT_DATASET):
    """Clean, split, fit TF-IDF on train text only, train, evaluate, and save."""
    cleaned_path = Path(input_path).resolve().with_name("cleaned_dataset.csv")
    data = clean_dataset(input_path=input_path, output_path=cleaned_path)
    texts = data[TEXT_COLUMN].map(normalize_text)
    labels = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels,
    )

    vectorizer = create_vectorizer()
    train_features = vectorizer.fit_transform(x_train)
    test_texts = x_test
    model = LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE)
    model.fit(train_features, y_train)

    evaluate_model(model, vectorizer, test_texts, y_test)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"TF-IDF vectorizer saved to: {VECTORIZER_PATH}")
    return model, vectorizer


if __name__ == "__main__":
    train_pipeline()
    # Exercise predictions from the persisted artifacts after training.
    from predict import predict_email
    examples = {
        "Ordinary legitimate email": "Hi Alex, our project meeting is confirmed for Tuesday at 10 AM. The agenda is attached.",
        "Suspicious message": "URGENT: Your account will be closed today. Verify your identity immediately to avoid suspension.",
        "Suspicious link": "Your mailbox is full. Sign in now at http://secure-account.example.invalid/verify to restore access.",
        "Password, OTP, or banking request": "We detected a problem. Reply with your password, OTP, and bank account details to unlock your account.",
    }
    for title, message in examples.items():
        result = predict_email(message)
        print(f"{title}: {result['prediction']} (confidence {result['confidence']:.2%}; phishing probability {result['phishing_probability']:.2%})")
