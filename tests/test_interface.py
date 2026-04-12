from fastapi.testclient import TestClient

from src import app as api_app


def test_natural_language_parser_extracts_features() -> None:
    query = "glucose 148, blood pressure 72, bmi 33.6, age 50"
    parsed = api_app.parse_natural_language_input(query)

    assert parsed["is_complete"] is True
    assert parsed["features"] == {
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "BMI": 33.6,
        "Age": 50.0,
    }


def test_natural_language_parser_handles_incomplete_or_invalid_input() -> None:
    query = "glucose is high and bmi is unknown"
    parsed = api_app.parse_natural_language_input(query)

    assert parsed["is_complete"] is False
    assert parsed["features"] == {}
    assert set(parsed["missing_features"]) == {
        "Glucose", "BloodPressure", "BMI", "Age"}


def test_predict_form_rejects_invalid_inputs() -> None:
    client = TestClient(api_app.app)
    response = client.post(
        "/predict/form",
        json={"Glucose": 120, "BloodPressure": 70, "BMI": 28.5, "Age": 0},
    )

    assert response.status_code == 422
    assert "Age and BMI must be positive values" in response.text


def test_predict_form_returns_diabetes_probability() -> None:
    client = TestClient(api_app.app)

    response = client.post(
        "/predict/form",
        json={"Glucose": 148, "BloodPressure": 72, "BMI": 33.6, "Age": 50},
    )

    payload = response.json()
    assert response.status_code == 200
    assert set(payload.keys()) == {"response"}
    assert "Prediction:" in payload["response"]
