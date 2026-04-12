import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate_model
from src.train import build_model_from_config


def _test_config() -> dict:
    return {
        "data": {"random_state": 42},
        "training": {
            "logistic_regression": {
                "max_iter": 2000,
                "C": 1.0,
                "class_weight": None,
            }
        },
    }


def _train_reference():
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model_from_config(_test_config())
    model.fit(x_train, y_train)
    return model, x_test, y_test


def test_model_predictions_have_correct_type_and_shape() -> None:
    model, x_test, _ = _train_reference()
    predictions = model.predict(x_test)

    assert predictions.shape[0] == x_test.shape[0]
    assert np.issubdtype(predictions.dtype, np.integer)


def test_model_meets_minimum_performance_threshold() -> None:
    model, x_test, y_test = _train_reference()
    metrics = evaluate_model(model, x_test, y_test)

    assert metrics["accuracy"] >= 0.90
    assert metrics["auc"] >= 0.95
