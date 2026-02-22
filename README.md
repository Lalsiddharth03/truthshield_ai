# TruthShield AI

A real-time fake news detection API powered by machine learning. TruthShield AI uses logistic regression with TF-IDF vectorization to classify news articles as real or fake with confidence scores.

## Features

- FastAPI-based REST API for news classification
- Machine learning model trained on balanced dataset
- Confidence scoring with HIGH/MEDIUM/LOW levels
- Health check and prediction endpoints

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   └── ml/
│   │       ├── prepare_dataset.py   # Dataset preprocessing
│   │       ├── train.py             # Model training
│   │       └── predict.py           # Prediction logic
│   └── artifacts/               # Trained model files
└── data/
    └── fake_or_real_news.csv    # Raw dataset
```

## Setup

1. Install dependencies:
```bash
pip install fastapi pandas scikit-learn joblib uvicorn
```

2. Prepare the dataset:
```bash
python backend/app/ml/prepare_dataset.py
```

3. Train the model:
```bash
python backend/app/ml/train.py
```

4. Run the API:
```bash
uvicorn backend.app.main:app --reload
```

## API Usage

**Predict news authenticity:**
```bash
POST /predict
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 0.9234,
  "confidence_level": "HIGH",
  "probabilities": {
    "fake": 0.0766,
    "real": 0.9234
  }
}
```
