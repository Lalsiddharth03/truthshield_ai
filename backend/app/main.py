from fastapi import FastAPI
from pydantic import BaseModel
import logging

from app.ml.predict import predict_text

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# FastAPI App Initialization
# -------------------------------
app = FastAPI(
    title="TruthShield AI",
    description="Real-Time Fake News Detection API",
    version="1.0"
)

# -------------------------------
# Request & Response Models
# -------------------------------

class NewsRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    confidence_level: str
    probabilities: dict


# -------------------------------
# Routes
# -------------------------------

@app.get("/")
def home():
    return {"message": "TruthShield AI is running 🚀"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_news(request: NewsRequest):
    logger.info("Received prediction request")

    result = predict_text(request.text)

    if "error" in result:
        logger.warning("Prediction failed due to invalid input")
        return result

    logger.info(f"Prediction result: {result['prediction']} | Confidence: {result['confidence']}")

    return result