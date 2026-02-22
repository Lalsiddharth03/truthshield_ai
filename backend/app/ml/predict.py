import joblib
import os


# Dynamically locate artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_PATH, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_PATH, "vectorizer.pkl")


# Load model once at startup (important for performance)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def get_confidence_level(confidence: float) -> str:
    if confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.65:
        return "MEDIUM"
    else:
        return "LOW"


def predict_text(text: str):
    """
    Predict whether a news article is fake or real.
    Returns structured professional response.
    """

    if not text or len(text.strip()) < 20:
        return {
            "error": "Input text is too short. Please provide a longer news article."
        }

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]

    fake_prob = float(probabilities[0])
    real_prob = float(probabilities[1])

    confidence = max(fake_prob, real_prob)

    result = {
        "prediction": "REAL" if prediction == 1 else "FAKE",
        "confidence": round(confidence, 4),
        "confidence_level": get_confidence_level(confidence),
        "probabilities": {
            "fake": round(fake_prob, 4),
            "real": round(real_prob, 4)
        }
    }

    return result