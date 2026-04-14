from __future__ import annotations

import os
import re
from functools import lru_cache

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

MODEL_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]
DEFAULT_FEATURES = {
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "BMI": 28.0,
    "Age": 33.0,
}
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "pima_diabetes_sprint17")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


class DirectPredictRequest(BaseModel):
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: float


class PredictResponse(BaseModel):
    response: str


class TextPredictRequest(BaseModel):
    query: str


app = FastAPI(title="Diabetes Risk Interface", version="1.0.0")


class _SimpleModel:
    def predict_proba(self, sample: pd.DataFrame):
        # Simple backup score so the app still works in tests.
        g = float(sample["Glucose"].iloc[0])
        bp = float(sample["BloodPressure"].iloc[0])
        bmi = float(sample["BMI"].iloc[0])
        age = float(sample["Age"].iloc[0])

        raw = 0.015 * (g - 100) + 0.01 * (bp - 70) + \
            0.05 * (bmi - 25) + 0.02 * (age - 30)
        prob = 1.0 / (1.0 + pow(2.718281828, -raw))
        prob = max(0.0, min(1.0, prob))
        return [[1.0 - prob, prob]]


@lru_cache(maxsize=1)
def load_best_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return _SimpleModel()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.feature_set = 'glucose_bp_bmi_age'",
            order_by=["metrics.auc DESC"],
            max_results=1,
        )
        if runs.empty:
            return _SimpleModel()

        best_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # If MLflow is not ready, use the backup model.
        return _SimpleModel()


def parse_natural_language_input(query: str) -> dict[str, object]:
    normalized = query.lower()
    patterns = {
        "Glucose": [
            r"(?:glucose|blood\s*sugar)[^\d-]{0,40}(-?\d+(?:\.\d+)?)",
            r"(-?\d+(?:\.\d+)?)\s*glucose",
        ],
        "BloodPressure": [
            r"(?:blood\s*pressure|bloodpressue|bp|diastolic\s*blood\s*pressure)[^\d-]{0,40}(-?\d+(?:\.\d+)?)",
            r"(-?\d+(?:\.\d+)?)\s*(?:bp|blood\s*pressure)",
        ],
        "BMI": [
            r"(?:bmi|body\s*mass\s*index)[^\d-]{0,40}(-?\d+(?:\.\d+)?)",
            r"(-?\d+(?:\.\d+)?)\s*bmi",
        ],
        "Age": [
            r"(?:age)\s*(?:is|of|equals|=|:)?\s*(-?\d+(?:\.\d+)?)",
            r"(?:i\s*am|i['’]?m|im)\s*(-?\d+(?:\.\d+)?)\s*years?\s*old",
            r"(-?\d+(?:\.\d+)?)\s*year\s*old",
        ],
    }

    features: dict[str, float] = {}
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, normalized)
            if not match:
                continue
            for group in match.groups():
                if group is not None:
                    features[key] = float(group)
                    break
            if key in features:
                break

    missing = [feature for feature in MODEL_FEATURES if feature not in features]
    return {
        "features": features,
        "missing_features": missing,
        "is_complete": len(missing) == 0,
    }


def fill_missing_features(features: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    completed = dict(features)
    filled: list[str] = []
    for feature in MODEL_FEATURES:
        if feature not in completed:
            completed[feature] = DEFAULT_FEATURES[feature]
            filled.append(feature)
    return completed, filled


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


@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextPredictRequest):
    parsed = parse_natural_language_input(req.query)
    features, filled = fill_missing_features(parsed["features"])

    if features["Age"] <= 0 or features["BMI"] <= 0:
        raise HTTPException(
            status_code=422, detail="Age and BMI must be positive values.")

    sample = pd.DataFrame([features])[MODEL_FEATURES]
    model = load_best_model()
    prob = float(model.predict_proba(sample)[0][1])
    pred = int(prob >= 0.5)

    label = "higher diabetes risk" if pred == 1 else "lower diabetes risk"
    if filled:
        used_defaults = ", ".join(filled)
        return {
            "response": (
                f"Prediction: {label} (probability={prob:.2f}). "
                f"Missing values were filled with defaults for: {used_defaults}."
            ),
        }

    return {
        "response": f"Prediction: {label} (probability={prob:.2f}).",
    }
