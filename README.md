# Diabetes Risk Prediction + LLM Interface

## Project Description
This project predicts diabetes risk from basic health inputs and serves results through a simple API.

The goal was to build one clean workflow that includes:
- data prep
- model training
- evaluation
- experiment tracking
- app deployment

In short, it turns raw tabular health data into a usable risk prediction you can test through an endpoint.

## Repository Structure
```text
archive/
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

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create an environment file:
```bash
cp .env.example .env
```
4. If you use external LLM services, add your API keys to .env.
5. Data details:
- Training uses the Pima Indians Diabetes dataset through kagglehub.
- Training options and chosen features are in configs/config.yaml.

## How To Run
### Train the model
```bash
python -m src.train --config configs/config.yaml
```

### Start the API
```bash
uvicorn src.app:app --reload
```
Useful pages:
- http://localhost:8000/health
- http://localhost:8000/docs

### Run with Docker
```bash
docker build -t diabetes-risk-app .
docker run -p 8000:8000 diabetes-risk-app
```

### Run tests
```bash
pytest tests/ -v
```

## Architecture (Quick View)
- src/preprocess.py: handles data cleaning and scaling.
- src/train.py: trains models from YAML config and logs runs to MLflow.
- src/evaluate.py: calculates metrics (accuracy, precision, recall, F1, AUC).
- src/app.py: serves prediction endpoints and input parsing.

The model outputs probabilities, and the app turns those into a clear response.

## Results Summary
Current quality thresholds checked in tests:
- Accuracy >= 0.90
- AUC >= 0.95

Recent local test runs meet these targets.

## Best Model Choice (Why)
The best model is selected by ranking MLflow runs by AUC on held-out test data.

Why AUC:
- This is a medical risk task, so we care about class separation across thresholds.
- The dataset is not perfectly balanced, so AUC is more reliable than accuracy alone.

Final choice:
- The top run by AUC is chosen as the best model.
- I still review F1 and accuracy as secondary checks so the model is not only good on one metric.

## Reflection
### What I learned
- Breaking the project into separate modules makes it easier to test and maintain.
- Config-based training makes experiments faster and easier to reproduce.

### What was challenging
- Keeping interface tests stable without relying on external services.
- Staying simple while still using a realistic project structure.

### What I would improve next
- Always load the best MLflow model artifact in the app flow.
- Add deeper evaluation (more charts, threshold tuning, subgroup checks).
- Strengthen CI with linting, type checks, and Docker validation.
