import pytest
from fastapi.testclient import TestClient

from src import app as api_app


def test_natural_language_parser_extracts_features() -> None:
    query = "I am a 50 year old male with a bmi of 33 and my bloodpressue is 77"
    parsed = api_app.parse_natural_language_input(query)

    assert parsed["is_complete"] is False
    assert parsed["features"] == {
        "BloodPressure": 77.0,
        "BMI": 33.0,
        "Age": 50.0,
    }


@pytest.mark.parametrize(
    "query,feature,expected_value",
    [
        ("My glucose is 145.", "Glucose", 145.0),
        ("Glucose: 118.", "Glucose", 118.0),
        ("My blood sugar is 160.", "Glucose", 160.0),
        ("I have a glucose level of 132.", "Glucose", 132.0),
        ("Glucose equals 109.", "Glucose", 109.0),
        ("My glucose reading today was 175.", "Glucose", 175.0),
        ("My blood pressure is 78.", "BloodPressure", 78.0),
        ("BloodPressure: 72.", "BloodPressure", 72.0),
        ("BP is 84.", "BloodPressure", 84.0),
        ("My bp today is 90.", "BloodPressure", 90.0),
        ("Blood pressure equals 76.", "BloodPressure", 76.0),
        ("My diastolic blood pressure is 80.", "BloodPressure", 80.0),
        ("My BMI is 33.", "BMI", 33.0),
        ("BMI: 27.5.", "BMI", 27.5),
        ("I have a bmi of 31.", "BMI", 31.0),
        ("My body mass index is 29.4.", "BMI", 29.4),
        ("bmi equals 35.", "BMI", 35.0),
        ("BMI is around 26.", "BMI", 26.0),
        ("My age is 50.", "Age", 50.0),
        ("Age: 41.", "Age", 41.0),
        ("I am 63 years old.", "Age", 63.0),
        ("I'm 29 years old.", "Age", 29.0),
        ("Age equals 37.", "Age", 37.0),
        ("I am a 55 year old male.", "Age", 55.0),
    ],
)
def test_natural_language_parser_supports_multiple_phrasings(
    query: str,
    feature: str,
    expected_value: float,
) -> None:
    parsed = api_app.parse_natural_language_input(query)
    assert feature in parsed["features"]
    assert parsed["features"][feature] == expected_value


def test_natural_language_parser_extracts_all_features_from_combined_text() -> None:
    query = "I am 50 years old, my glucose is 148, blood pressure is 77, and bmi is 33."
    parsed = api_app.parse_natural_language_input(query)

    assert parsed["is_complete"] is True
    assert parsed["features"] == {
        "Glucose": 148.0,
        "BloodPressure": 77.0,
        "BMI": 33.0,
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


def test_predict_text_handles_incomplete_input_gracefully() -> None:
    client = TestClient(api_app.app)

    response = client.post(
        "/predict/text",
        json={
            "query": "I am a 50 year old male with a bmi of 33 and my bloodpressue is 77"
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert set(payload.keys()) == {"response"}
    assert "Prediction:" in payload["response"]
    assert "probability=" in payload["response"]
    assert "Missing values were filled with defaults" in payload["response"]
