"""Load the saved artifacts and classify new email text."""
from pathlib import Path
import joblib
from preprocessing import normalize_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "logistic_regression_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.pkl"


def predict_email(email_text, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    """Return a human-readable class and probability for the predicted class."""
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("Email text must be a non-empty string.")
    model_path, vectorizer_path = Path(model_path), Path(vectorizer_path)
    if not model_path.is_file() or not vectorizer_path.is_file():
        raise FileNotFoundError("Model files are missing. Run train_model.py first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    probabilities = model.predict_proba(vectorizer.transform([normalize_text(email_text)]))[0]
    label = int(probabilities.argmax())
    return {
        "prediction": "Phishing Email" if label == 1 else "Legitimate Email",
        "confidence": float(probabilities[label]),
        "phishing_probability": float(probabilities[1]),
    }


if __name__ == "__main__":
    examples = {
        "Ordinary legitimate email": "Hi Alex, our project meeting is confirmed for Tuesday at 10 AM. The agenda is attached.",
        "Suspicious message": "URGENT: Your account will be closed today. Verify your identity immediately to avoid suspension.",
        "Suspicious link": "Your mailbox is full. Sign in now at http://secure-account.example.invalid/verify to restore access.",
        "Password or banking request": "We detected a problem. Reply with your password, OTP, and bank account details to unlock your account.",
    }
    for title, message in examples.items():
        result = predict_email(message)
        print(f"{title}: {result['prediction']} (confidence {result['confidence']:.2%}; phishing probability {result['phishing_probability']:.2%})")
