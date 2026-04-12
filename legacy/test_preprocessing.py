import numpy as np
import pandas as pd
import pandas.testing as pdt

from ml_pipeline_utils import preprocess_dataframe


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Glucose": [110.0, np.nan, 150.0, 95.0],
            "BMI": [28.4, 31.2, np.nan, 22.5],
            "Age": [33.0, 47.0, 52.0, np.nan],
            "Gender": ["F", "M", None, "F"],
        }
    )


def test_missing_values_are_handled() -> None:
    df = _sample_df()
    processed, _ = preprocess_dataframe(
        df,
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )

    assert not processed.isna().any().any()


def test_categoricals_are_encoded() -> None:
    df = _sample_df()
    processed, _ = preprocess_dataframe(
        df,
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )

    encoded_columns = [
        col for col in processed.columns if col.startswith("Gender_")]
    assert set(encoded_columns) >= {"Gender_F", "Gender_M"}


def test_numerics_are_scaled_to_expected_ranges() -> None:
    df = _sample_df()
    processed, _ = preprocess_dataframe(
        df,
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )

    numeric_processed = processed[["Glucose", "BMI", "Age"]]
    means = numeric_processed.mean().abs()
    stds = numeric_processed.std(ddof=0)

    assert (means < 1e-9).all()
    assert ((stds - 1.0).abs() < 1e-9).all()


def test_original_dataframe_is_not_modified() -> None:
    original = _sample_df()
    original_snapshot = original.copy(deep=True)

    preprocess_dataframe(
        original,
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )

    pdt.assert_frame_equal(original, original_snapshot)
