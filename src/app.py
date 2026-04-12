from __future__ import annotations

import re

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]


class DirectPredictRequest(BaseModel):
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: float


class PredictResponse(BaseModel):
    response: str


app = FastAPI(title="Diabetes Risk Interface", version="1.0.0")


class _SimpleModel:
    def predict_proba(self, sample: pd.DataFrame):
        # Deterministic baseline score for demonstration and testing.
        g = float(sample["Glucose"].iloc[0])
        bp = float(sample["BloodPressure"].iloc[0])
        bmi = float(sample["BMI"].iloc[0])
        age = float(sample["Age"].iloc[0])

        raw = 0.015 * (g - 100) + 0.01 * (bp - 70) + \
            0.05 * (bmi - 25) + 0.02 * (age - 30)
        prob = 1.0 / (1.0 + pow(2.718281828, -raw))
        prob = max(0.0, min(1.0, prob))
        return [[1.0 - prob, prob]]


def load_best_model() -> _SimpleModel:
    return _SimpleModel()


def parse_natural_language_input(query: str) -> dict[str, object]:
    normalized = query.lower()
    patterns = {
        "Glucose": r"(?:glucose)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
        "BloodPressure": r"(?:blood\s*pressure|bp)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
        "BMI": r"(?:bmi)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
        "Age": r"(?:age)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)",
    }

    features: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, normalized)
        if match:
            features[key] = float(match.group(1))

    missing = [feature for feature in MODEL_FEATURES if feature not in features]
    return {
        "features": features,
        "missing_features": missing,
        "is_complete": len(missing) == 0,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/form", response_model=PredictResponse)
def predict_direct(req: DirectPredictRequest):
    features = {
        "Glucose": req.Glucose,
        "BloodPressure": req.BloodPressure,
        "BMI": req.BMI,
        "Age": req.Age,
    }

    if features["Age"] <= 0 or features["BMI"] <= 0:
        raise HTTPException(
            status_code=422, detail="Age and BMI must be positive values.")

    sample = pd.DataFrame([features])[MODEL_FEATURES]
    model = load_best_model()
    prob = float(model.predict_proba(sample)[0][1])
    pred = int(prob >= 0.5)

    label = "higher diabetes risk" if pred == 1 else "lower diabetes risk"
    return {
        "response": f"Prediction: {label} (probability={prob:.2f}).",
    }
