import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Dynamically get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "news.csv")


def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df


def train_model(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


def save_artifacts(model, vectorizer):
    ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(ARTIFACT_PATH, exist_ok=True)

    joblib.dump(model, os.path.join(ARTIFACT_PATH, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(ARTIFACT_PATH, "vectorizer.pkl"))

    print("\nModel and vectorizer saved successfully.")


if __name__ == "__main__":
    print("Reading dataset from:", DATA_PATH)
    df = load_data(DATA_PATH)
    model, vectorizer = train_model(df)
    save_artifacts(model, vectorizer)