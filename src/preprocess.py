from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    out = df.copy(deep=True)

    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())

    for col in categorical_cols:
        mode = out[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        out[col] = out[col].fillna(fill_value)

    return out


def encode_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    if not categorical_cols:
        return df.copy(deep=True)
    return pd.get_dummies(df, columns=categorical_cols, drop_first=False)


def scale_numerics(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    out = df.copy(deep=True)
    scaler = StandardScaler()
    out[numeric_cols] = scaler.fit_transform(out[numeric_cols].astype(float))
    return out, scaler


def preprocess_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    cleaned = handle_missing_values(df, numeric_cols, categorical_cols)
    encoded = encode_categoricals(cleaned, categorical_cols)
    scaled, scaler = scale_numerics(encoded, numeric_cols)
    return scaled, scaler
