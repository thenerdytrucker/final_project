import os
import re
from functools import lru_cache

import kagglehub
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mlflow.tracking import MlflowClient
from openai import OpenAI
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = [
    "Glucose",
    "BloodPressure",
    "BMI",
    "Age",
]

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pima_diabetes_sprint17")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI",
                         "sqlite:////app/data/mlflow.db")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL = os.getenv(
    "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")
NEBIUS_MODEL = os.getenv(
    "NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
AUTO_BOOTSTRAP_RUNS = os.getenv(
    "AUTO_BOOTSTRAP_RUNS", "true").lower() == "true"


class DirectPredictRequest(BaseModel):
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: float


class PredictResponse(BaseModel):
    status: str
    best_run_id: str
    prediction_class: int
    prediction_probability: float
    response: str
    limitations: list[str]


app = FastAPI(title="Pima Diabetes LLM Interface", version="1.0.0")


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI | None:
    if not NEBIUS_API_KEY:
        return None
    return OpenAI(api_key=NEBIUS_API_KEY, base_url=NEBIUS_BASE_URL)


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

    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in features]
    return {
        "features": features,
        "missing_features": missing_features,
        "is_complete": len(missing_features) == 0,
    }


def _metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def bootstrap_experiment_runs() -> None:
    data_path = kagglehub.dataset_download(
        "uciml/pima-indians-diabetes-database")
    df = pd.read_csv(os.path.join(data_path, "diabetes.csv"))

    X = df[MODEL_FEATURES]
    y = df["Outcome"]

    train_features, test_features, train_target, test_target = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    run_configs: dict[str, Pipeline] = {
        "logreg_default": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "logreg_balanced_c03": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1200, C=0.3,
             class_weight="balanced", random_state=42)),
        ]),
        "rf_200_default": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "rf_400_depth8": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=400,
             max_depth=8, min_samples_leaf=2, random_state=42)),
        ]),
        "gb_200_lr01_depth4": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200,
             learning_rate=0.1, max_depth=4, random_state=42)),
        ]),
    }

    for run_name, model_pipeline in run_configs.items():
        with mlflow.start_run(run_name=run_name):
            model_pipeline.fit(train_features, train_target)
            preds = model_pipeline.predict(test_features)
            probs = model_pipeline.predict_proba(test_features)[:, 1]
            mlflow.log_metrics(_metrics(test_target, preds, probs))
            mlflow.log_params(
                {
                    "run_name": run_name,
                    "feature_count": len(MODEL_FEATURES),
                    "feature_set": "glucose_bp_bmi_age",
                    "data_source": "kagglehub:uciml/pima-indians-diabetes-database",
                    "split_random_state": 42,
                }
            )
            mlflow.sklearn.log_model(
                model_pipeline,
                name="model",
                serialization_format="cloudpickle",
                pip_requirements=["scikit-learn", "cloudpickle"],
            )


@lru_cache(maxsize=1)
def load_best_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        if not AUTO_BOOTSTRAP_RUNS:
            raise RuntimeError(
                "MLflow experiment not found. Run training first or set AUTO_BOOTSTRAP_RUNS=true.")
        bootstrap_experiment_runs()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.feature_set = 'glucose_bp_bmi_age'",
        order_by=["metrics.auc DESC"],
        max_results=50,
    )

    if runs.empty:
        if not AUTO_BOOTSTRAP_RUNS:
            raise RuntimeError(
                "No runs found in experiment. Run training first or set AUTO_BOOTSTRAP_RUNS=true.")
        bootstrap_experiment_runs()
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.feature_set = 'glucose_bp_bmi_age'",
            order_by=["metrics.auc DESC"],
            max_results=50,
        )

    best_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model, best_run_id


def generate_prediction_response(prediction: int, probability: float, features: dict[str, float]) -> str:
    client = get_llm_client()
    if client is None:
        label = "higher diabetes risk" if prediction == 1 else "lower diabetes risk"
        return (
            f"Prediction: {label} (probability={probability:.2f}). "
            "This is a screening estimate, not a diagnosis. Use clinical evaluation for medical decisions."
        )

    prompt = (
        "Create a concise user-facing explanation for a diabetes-risk model prediction. "
        "Include: result, plain-language meaning, and caution that this is not medical advice. "
        f"Prediction class: {prediction} (1 means higher risk). Probability: {probability:.4f}. Features: {features}."
    )

    try:
        completion = client.chat.completions.create(
            model=NEBIUS_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a careful healthcare AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        label = "higher diabetes risk" if prediction == 1 else "lower diabetes risk"
        return (
            f"Prediction: {label} (probability={probability:.2f}). "
            "This is a screening estimate, not a diagnosis. Use clinical evaluation for medical decisions."
        )


@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html>
<head><title>Pima Diabetes LLM API</title></head>
<body>
<h2>Pima Diabetes LLM API is running.</h2>
<ul>
  <li><a href="/docs">http://localhost:8000/docs</a> &mdash; Swagger UI</li>
  <li><a href="/health">http://localhost:8000/health</a> &mdash; Health check</li>
  <li><a href="/predict">http://localhost:8000/predict</a> &mdash; Diabetes prediction form</li>
</ul>
</body>
</html>"""


@app.get("/health")
def health_check():
    return {"status": "ok", "tracking_uri": TRACKING_URI, "experiment": EXPERIMENT_NAME}


PREDICT_FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Diabetes Onset Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1 { color: #333; }
    p.desc { color: #555; margin-bottom: 24px; }
    label { display: block; margin-top: 12px; font-weight: bold; color: #444; }
    .hint { font-size: 0.8em; color: #888; font-weight: normal; }
    input[type=number] { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    button { margin-top: 20px; padding: 10px 28px; background: #0066cc; color: white; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
    button:hover { background: #0052a3; }
    #result { margin-top: 28px; padding: 16px; border-radius: 6px; display: none; }
    #result.ok { background: #e8f5e9; border: 1px solid #66bb6a; }
    #result.risk { background: #fff3e0; border: 1px solid #ffa726; }
    #result.error { background: #fdecea; border: 1px solid #ef5350; }
    .prob-bar-wrap { background: #ddd; border-radius: 4px; height: 12px; margin: 8px 0; }
    .prob-bar { height: 12px; border-radius: 4px; transition: width 0.5s; }
    a.back { display: inline-block; margin-top: 16px; color: #0066cc; text-decoration: none; }
  </style>
</head>
<body>
  <h1>Diabetes Onset Predictor</h1>
  <p class="desc">Predict the onset of diabetes based on medical and demographic data such as glucose levels, BMI, and age.</p>

  <form id="predictForm">
    <label>Glucose <span class="hint">(plasma glucose mg/dL, e.g. 120)</span></label>
    <input type="number" name="Glucose" min="0" max="300" step="1" value="120" required>

    <label>Blood Pressure <span class="hint">(diastolic mm Hg, e.g. 70)</span></label>
    <input type="number" name="BloodPressure" min="0" max="150" step="1" value="70" required>

    <label>BMI <span class="hint">(body mass index kg/m&sup2;, e.g. 28.5)</span></label>
    <input type="number" name="BMI" min="0" max="70" step="0.1" value="28.5" required>

    <label>Age <span class="hint">(years, e.g. 33)</span></label>
    <input type="number" name="Age" min="1" max="120" step="1" value="33" required>

    <button type="submit">Predict</button>
  </form>

  <div id="result"></div>
  <a class="back" href="/">&larr; Back to home</a>

  <script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = {};
      fd.forEach((v, k) => { body[k] = parseFloat(v); });
      const resDiv = document.getElementById('result');
      resDiv.style.display = 'none';
      try {
        const resp = await fetch('/predict/form', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
                const data = await resp.json();
                if (!resp.ok) {
                    throw new Error(data.detail || 'Prediction failed');
                }
        const prob = data.prediction_probability ?? 0;
        const pct = Math.round(prob * 100);
        const isRisk = data.prediction_class === 1;
        const barColor = isRisk ? '#ffa726' : '#66bb6a';
        resDiv.className = isRisk ? 'risk' : 'ok';
        resDiv.innerHTML = `
          <strong>Result: ${isRisk ? '&#9888;&#65039; Higher diabetes risk' : '&#9989; Lower diabetes risk'}</strong>
          <div class="prob-bar-wrap"><div class="prob-bar" style="width:${pct}%;background:${barColor}"></div></div>
          <p>Risk probability: <strong>${pct}%</strong></p>
          <p>${data.response}</p>
          <p style="font-size:0.8em;color:#888">${(data.limitations || []).join(' ')}</p>`;
        resDiv.style.display = 'block';
            } catch(err) {
        resDiv.className = 'error';
                resDiv.innerHTML = `<strong>Error:</strong> ${err.message}`;
        resDiv.style.display = 'block';
      }
    });
  </script>
</body>
</html>"""


@app.get("/predict", response_class=HTMLResponse)
def predict_form_page():
    return PREDICT_FORM_HTML


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

    model, best_run_id = load_best_model()
    sample = pd.DataFrame([features])[MODEL_FEATURES]

    pred = int(model.predict(sample)[0])
    prob = float(model.predict_proba(sample)[0][1]) if hasattr(
        model, "predict_proba") else float(pred)

    response_text = generate_prediction_response(pred, prob, features)

    return {
        "status": "ok",
        "best_run_id": best_run_id,
        "prediction_class": pred,
        "prediction_probability": round(prob, 4),
        "response": response_text,
        "limitations": [
            "Model is trained on a dataset and may not generalize to all populations.",
            "This output is not medical advice and should not replace clinical diagnosis.",
        ],
    }
