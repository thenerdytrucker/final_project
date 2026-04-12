from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
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


def load_training_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    source = config["data"].get("source", "sklearn_breast_cancer")
    if source != "sklearn_breast_cancer":
        raise ValueError(
            "Only 'sklearn_breast_cancer' is supported in this template.")

    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


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

    model = build_model_from_config(config)

    with mlflow.start_run(run_name="config_driven_logreg"):
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "model_type": "logistic_regression",
            "features": ",".join(config["features"].get("numeric", [])),
            "test_size": config["data"].get("test_size", 0.2),
            "random_state": config["data"].get("random_state", 42),
        })
        mlflow.sklearn.log_model(model, name="model")

    return metrics


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
    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
