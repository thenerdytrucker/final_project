import mlflow.sklearn
import mlflow
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import kagglehub
import numpy as np
import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# Dataset: Pima Indians Diabetes Database (UCI / Kaggle)
# Goal:    Predict onset of diabetes from medical and
#          demographic features (glucose, BMI, age, etc.)
# -------------------------------------------------------


# ----------------------------
# 1) Download from Kaggle
# ----------------------------
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)

data_path = Path(path) / "diabetes.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"diabetes.csv not found in {path}. "
        "Files available: " + str(list(Path(path).iterdir()))
    )

df = pd.read_csv(data_path)
print("Loaded:", data_path)


# ----------------------------
# 2) Shape and first look
# ----------------------------
print("\nShape (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ----------------------------
# 3) Row and index checks
# ----------------------------
print("\nIndex is unique:", df.index.is_unique)
print("Index is monotonic:", df.index.is_monotonic_increasing)
print("Index range:", df.index.min(), "to", df.index.max())


# ----------------------------
# 4) Missing / NaN values
# ----------------------------
# NOTE: This dataset encodes missing values as 0 in medical columns.
# True NaN check first, then flag biological zeros.
nan_count = df.isna().sum()
nan_pct = (df.isna().mean() * 100).round(2)

missing_summary = pd.DataFrame({
    "nan_count":   nan_count,
    "nan_percent": nan_pct,
}).sort_values("nan_count", ascending=False)

print("\nTrue NaN summary:")
print(missing_summary)

# Columns where 0 is biologically impossible and indicates missing data.
zero_as_missing_cols = ["Glucose", "BloodPressure",
                        "SkinThickness", "Insulin", "BMI"]
zero_check = {col: (df[col] == 0).sum()
              for col in zero_as_missing_cols if col in df.columns}

print("\nZero values (likely missing) in medical columns:")
for col, count in zero_check.items():
    pct = round(count / len(df) * 100, 2)
    print(f"  {col}: {count} zeros ({pct}%)")


# ----------------------------
# 5) Duplicate rows
# ----------------------------
print("\nDuplicate full rows:", df.duplicated().sum())


# ----------------------------
# 6) Target column check
# ----------------------------
if "Outcome" in df.columns:
    print("\nOutcome (target) value counts:")
    print(df["Outcome"].value_counts())
    print("Class balance (%):")
    print((df["Outcome"].value_counts(normalize=True) * 100).round(2))
else:
    print("\nOutcome column not found.")


# ----------------------------
# 7) summary
# ----------------------------
print("\nNumeric summary:")
print(df.describe())

print("\nDone: basic EDA complete.")


# =======================================================
# PREPROCESSING, TRAINING, EVALUATION
# =======================================================


# -------------------------------------------------------
# STEP 1 — Data Preprocessing
# -------------------------------------------------------
# All features are numeric — no categorical encoding needed.
# Zeros in medical columns are kept as-is (treated as outliers per EDA decision).
# Features are scaled inside each pipeline so no column is dropped.

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COL = "Outcome"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# 80/20 stratified split so class balance is preserved in both sets.
train_features, test_features, train_target, test_target = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain size: {len(train_features)} | Test size: {len(test_features)}")
print(
    f"Train class balance:\n{train_target.value_counts(normalize=True).mul(100).round(2)}")


# -------------------------------------------------------
# STEP 2 — Model Training (3 configurations)
# -------------------------------------------------------
# Model 1: Logistic Regression — fast linear baseline, benefits from scaling.
# Model 2: Random Forest — ensemble of decision trees, robust to outliers.
# Model 3: Gradient Boosting — sequential boosting, often best on tabular data.

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                           max_depth=4, random_state=42)),
    ]),
}

for name, pipeline in models.items():
    pipeline.fit(train_features, train_target)
    print(f"\nTrained: {name}")


# -------------------------------------------------------
# STEP 3 — Model Evaluation on held-out test set
# -------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL EVALUATION — Test Set Results")
print("=" * 60)

results = {}

for name, pipeline in models.items():
    y_pred = pipeline.predict(test_features)
    y_prob = pipeline.predict_proba(test_features)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(test_target, y_pred), 4),
        "Precision": round(precision_score(test_target, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(test_target, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(test_target, y_pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(test_target, y_prob), 4),
    }
    results[name] = metrics

    print(f"\n--- {name} ---")
    for metric, value in metrics.items():
        print(f"  {metric:<12}: {value}")
    print(classification_report(test_target, y_pred,
          target_names=["No Diabetes", "Diabetes"]))


# -------------------------------------------------------
# STEP 4 — Select best model
# -------------------------------------------------------
results_df = pd.DataFrame(results).T
print("\nSummary table:")
print(results_df.to_string())

best_model_name = results_df["AUC"].idxmax()
best_metrics = results_df.loc[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"  AUC       : {best_metrics['AUC']}")
print(f"  F1        : {best_metrics['F1']}")
print(f"  Accuracy  : {best_metrics['Accuracy']}")
print(
    f"\nJustification: {best_model_name} was selected based on the highest AUC score, "
    "which measures the model's ability to distinguish between diabetic and non-diabetic "
    "patients across all classification thresholds. For a medical prediction task AUC is "
    "preferred over accuracy alone because the dataset has a class imbalance (65/35)."
)
print("\nDone: preprocessing, training, and evaluation complete.")


# =======================================================
# EXPERIMENT TRACKING (SPRINT 17) - MLFLOW
# =======================================================

# Keep console output focused on run results instead of repeated serialization caution text.
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)


def evaluate_model(model_pipeline, x_test, y_test):
    y_pred = model_pipeline.predict(x_test)
    y_prob = model_pipeline.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
    }


# Use SQLite backend to avoid deprecated filesystem tracking store warnings.
tracking_db_uri = f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve().as_posix()}"
mlflow.set_tracking_uri(tracking_db_uri)
experiment = mlflow.set_experiment("pima_diabetes_sprint17")

print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
print("MLflow experiment:", experiment.name)

# At least five meaningfully different configurations.
tracked_configs = {
    "logreg_default": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "logreg_balanced_c03": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1200, C=0.3,
                                   class_weight="balanced", random_state=42)),
    ]),
    "rf_200_default": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]),
    "rf_400_depth8": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=8,
                                       min_samples_leaf=2, random_state=42)),
    ]),
    "gb_200_lr01_depth4": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                           max_depth=4, random_state=42)),
    ]),
    "gb_300_lr005_depth3": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=3, random_state=42)),
    ]),
}

for run_name, model_pipeline in tracked_configs.items():
    with mlflow.start_run(run_name=run_name):
        model_pipeline.fit(train_features, train_target)
        run_metrics = evaluate_model(
            model_pipeline, test_features, test_target)

        clf_params = {
            f"clf_{k}": v
            for k, v in model_pipeline.named_steps["clf"].get_params().items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }

        mlflow.log_params({
            "run_name": run_name,
            "feature_count": len(FEATURE_COLS),
            "features": ",".join(FEATURE_COLS),
            "target": TARGET_COL,
            "test_size": 0.20,
            "split_random_state": 42,
            "data_version": "uciml/pima-indians-diabetes-database",
            "data_path": str(data_path),
            "data_rows": int(df.shape[0]),
            "data_cols": int(df.shape[1]),
            "preprocessing": "numeric-only; scaler in pipeline; zeros kept as outliers",
        })
        mlflow.log_params(clf_params)
        mlflow.log_metrics(run_metrics)
        mlflow.sklearn.log_model(
            model_pipeline,
            name="model",
            serialization_format="cloudpickle",
            pip_requirements=["scikit-learn", "cloudpickle"],
        )

        print(f"Logged MLflow run: {run_name}")
        print({k: round(v, 4) for k, v in run_metrics.items()})


# Compare experiments and identify best run programmatically.
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.auc DESC"],
)

if runs.empty:
    raise RuntimeError("No MLflow runs were found for comparison.")

cols = [
    "run_id",
    "tags.mlflow.runName",
    "metrics.accuracy",
    "metrics.precision",
    "metrics.recall",
    "metrics.f1",
    "metrics.auc",
    "params.clf_n_estimators",
    "params.clf_C",
]
present_cols = [c for c in cols if c in runs.columns]

print("\nMLflow comparison table (sorted by AUC):")
print(runs[present_cols].head(10).to_string(index=False))

best_run = runs.iloc[0]
print("\nBest MLflow run by AUC:")
print("run_id:", best_run["run_id"])
print("run_name:", best_run.get("tags.mlflow.runName", "n/a"))
print("auc:", round(float(best_run["metrics.auc"]), 4))
print("f1:", round(float(best_run["metrics.f1"]), 4))
print("accuracy:", round(float(best_run["metrics.accuracy"]), 4))
