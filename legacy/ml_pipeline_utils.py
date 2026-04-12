import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def preprocess_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    working_df = df.copy(deep=True)

    for col in numeric_cols:
        working_df[col] = working_df[col].fillna(working_df[col].median())

    for col in categorical_cols:
        mode_values = working_df[col].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        working_df[col] = working_df[col].fillna(fill_value)

    scaled = StandardScaler().fit_transform(
        working_df[numeric_cols].astype(float))
    scaled_numeric = pd.DataFrame(
        scaled,
        columns=numeric_cols,
        index=working_df.index,
    )

    if categorical_cols:
        encoded_cats = pd.get_dummies(
            working_df[categorical_cols], drop_first=False)
        processed = pd.concat([scaled_numeric, encoded_cats], axis=1)
    else:
        processed = scaled_numeric

    scaler = StandardScaler()
    scaler.fit(working_df[numeric_cols].astype(float))
    return processed, scaler


def train_reference_model(random_state: int = 42) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])
    model.fit(x_train, y_train)
    return model, x_test, y_test


def evaluate_reference_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "auc": float(roc_auc_score(y_test, probabilities)),
    }
