from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import kagglehub
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(config: dict) -> Pipeline:
    lr_cfg = config["training"]["logistic_regression"]
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=int(lr_cfg.get("max_iter", 2000)),
                C=float(lr_cfg.get("C", 1.0)),
                class_weight=lr_cfg.get("class_weight", None),
                random_state=int(config["data"].get("random_state", 42)),
            ),
        ),
    ])


def build_pipeline(model_type: str, params: dict[str, Any], random_state: int) -> Pipeline:
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=int(params.get("max_iter", 2000)),
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", None),
            random_state=random_state,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model),
    ])


def _default_run_configs() -> list[dict[str, Any]]:
    return [
        {
            "run_name": "logreg_default",
            "model_type": "logistic_regression",
            "params": {"max_iter": 2000, "C": 1.0, "class_weight": None},
        },
        {
            "run_name": "logreg_balanced_c03",
            "model_type": "logistic_regression",
            "params": {"max_iter": 2000, "C": 0.3, "class_weight": "balanced"},
        },
        {
            "run_name": "rf_200_default",
            "model_type": "random_forest",
            "params": {"n_estimators": 200, "min_samples_leaf": 1},
        },
        {
            "run_name": "rf_400_depth8",
            "model_type": "random_forest",
            "params": {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 2},
        },
        {
            "run_name": "gb_200_lr01_depth3",
            "model_type": "gradient_boosting",
            "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        },
    ]


def compare_experiment_runs(experiment_id: str) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="params.feature_set = 'glucose_bp_bmi_age'",
        order_by=["metrics.auc DESC"],
        max_results=100,
    )
    columns = [
        "run_id",
        "tags.mlflow.runName",
        "params.model_type",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.auc",
    ]
    available = [c for c in columns if c in runs.columns]
    return runs[available]


def load_training_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    source = config["data"].get("source", "kagglehub_pima_diabetes")
    if source != "kagglehub_pima_diabetes":
        raise ValueError(
            "Only 'kagglehub_pima_diabetes' is supported in this project.")

    dataset_path = kagglehub.dataset_download(
        "uciml/pima-indians-diabetes-database")
    df = pd.read_csv(os.path.join(dataset_path, "diabetes.csv"))
    feature_columns = config["features"].get(
        "numeric", ["Glucose", "BloodPressure", "BMI", "Age"])
    return df[feature_columns], df["Outcome"]


def train_with_config(config_path: str) -> dict[str, float]:
    config = load_config(config_path)

    tracking_uri = config["experiment"].get(
        "tracking_uri", "sqlite:///mlflow.db")
    experiment_name = config["experiment"].get(
        "name", "pima_diabetes_sprint17")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    x, y = load_training_data(config)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(config["data"].get("test_size", 0.2)),
        random_state=int(config["data"].get("random_state", 42)),
        stratify=y,
    )

    run_configs = config.get("training", {}).get(
        "run_configs", _default_run_configs())
    random_state = int(config["data"].get("random_state", 42))

    for run_cfg in run_configs:
        run_name = run_cfg["run_name"]
        model_type = run_cfg["model_type"]
        params = run_cfg.get("params", {})

        model = build_pipeline(model_type, params, random_state)
        with mlflow.start_run(run_name=run_name):
            model.fit(x_train, y_train)
            metrics = evaluate_model(model, x_test, y_test)
            mlflow.log_metrics(metrics)
            flat_params = {f"model_{k}": v for k, v in params.items()}
            mlflow.log_params({
                "model_type": model_type,
                "feature_set": "glucose_bp_bmi_age",
                "features": ",".join(config["features"].get("numeric", [])),
                "test_size": config["data"].get("test_size", 0.2),
                "random_state": random_state,
                **flat_params,
            })
            mlflow.sklearn.log_model(model, name="model")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError("Experiment not found after logging runs.")

    ranked_runs = compare_experiment_runs(experiment.experiment_id)
    if ranked_runs.empty:
        raise RuntimeError("No runs found for comparison.")

    best_row = ranked_runs.iloc[0]
    return {
        "best_accuracy": float(best_row.get("metrics.accuracy", 0.0)),
        "best_precision": float(best_row.get("metrics.precision", 0.0)),
        "best_recall": float(best_row.get("metrics.recall", 0.0)),
        "best_f1": float(best_row.get("metrics.f1", 0.0)),
        "best_auc": float(best_row.get("metrics.auc", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train model with YAML config.")
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "config.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    metrics = train_with_config(args.config)
    print("Training complete. Best run metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
