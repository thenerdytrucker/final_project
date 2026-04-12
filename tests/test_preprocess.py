import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.preprocess import preprocess_dataframe


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
    processed, _ = preprocess_dataframe(
        _sample_df(),
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )
    assert not processed.isna().any().any()


def test_categoricals_are_encoded() -> None:
    processed, _ = preprocess_dataframe(
        _sample_df(),
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )
    encoded_columns = [c for c in processed.columns if c.startswith("Gender_")]
    assert set(encoded_columns) >= {"Gender_F", "Gender_M"}


def test_numerics_are_scaled_to_expected_ranges() -> None:
    processed, _ = preprocess_dataframe(
        _sample_df(),
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )
    numerics = processed[["Glucose", "BMI", "Age"]]
    assert (numerics.mean().abs() < 1e-9).all()
    assert ((numerics.std(ddof=0) - 1.0).abs() < 1e-9).all()


def test_original_dataframe_is_not_modified() -> None:
    original = _sample_df()
    snapshot = original.copy(deep=True)
    preprocess_dataframe(
        original,
        numeric_cols=["Glucose", "BMI", "Age"],
        categorical_cols=["Gender"],
    )
    pdt.assert_frame_equal(original, snapshot)
