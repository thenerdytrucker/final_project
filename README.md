# Diabetes Risk Prediction + LLM Interface

## Project Description
This project predicts the onset risk of diabetes from clinical features and provides a lightweight API/UI interface for interaction. It is designed for learners, data practitioners, and early-stage prototyping teams who need a reproducible machine learning workflow with experiment tracking and an application layer.

The problem it solves is turning raw health-style tabular features into a usable risk estimate, while keeping training, evaluation, configuration, and app integration organized in one codebase.

## Repository Structure
```
your-project/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   └── config.yaml
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_interface.py
├── notebooks/
│   └── exploration.ipynb
└── data/
    └── .gitkeep
```

## Setup Instructions
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create your environment file from template:
   ```bash
   cp .env.example .env
   ```
4. Update `.env` with your API keys if you plan to use external LLM services.
5. Data:
   - The training flow uses the Pima Indians Diabetes dataset via `kagglehub`.
   - Training settings and selected feature columns are defined in `configs/config.yaml`.

## Usage Instructions
### Train Model with Config
```bash
python -m src.train --config configs/config.yaml
```

### Run API
```bash
uvicorn src.app:app --reload
```
Then open:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

### Run with Docker
```bash
docker build -t diabetes-risk-app .
docker run -p 8000:8000 diabetes-risk-app
```

### Run Tests
```bash
pytest tests/ -v
```

## Architecture Overview
- `src/preprocess.py`: data cleaning, categorical encoding, and numeric scaling.
- `src/train.py`: config-driven training entry point that reads hyperparameters from YAML, splits data, trains pipeline, logs to MLflow.
- `src/evaluate.py`: evaluation utilities (accuracy, precision, recall, F1, AUC).
- `src/app.py`: interface layer exposing prediction endpoint and natural-language parsing helper.

The ML pipeline produces probability outputs; the app converts them into user-facing responses.

## Results Summary
Current reference model tests enforce minimum quality thresholds:
- Accuracy >= 0.90
- AUC >= 0.95

In local test runs, these thresholds are met consistently.

## Reflection
### What I learned
- How to separate ML training and app logic into testable modules.
- How config-driven workflows improve reproducibility and iteration speed.

### What was challenging
- Keeping interface tests deterministic while avoiding external API dependencies.
- Balancing project simplicity with realistic production structure.

### What I would improve with more time
- Add model registry + version promotion rules.
- Add richer calibration analysis and fairness checks.
- Add CI pipeline for lint, tests, and Docker build verification.
