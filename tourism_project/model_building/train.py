"""
train.py – Model Training with Hyperparameter Tuning & MLflow Tracking
=======================================================================
Builds an end-to-end sklearn Pipeline (imputation → encoding/scaling →
XGBoost), tunes hyperparameters via GridSearchCV, logs everything to
MLflow, and saves the best model for deployment.

GPU Support: Automatically detects NVIDIA GPU and uses CUDA acceleration
             when available (e.g., Colab T4), falls back to CPU otherwise.
"""

import os
import subprocess
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier


# ── Column Definitions ───────────────────────────────────────────────────
# Numerical features: scaled with StandardScaler, missing values imputed with median
NUMERICAL_COLS = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

# Categorical features: one-hot encoded, missing values imputed with most frequent
CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Trusted model types required by MLflow's sklearn flavor when persisting
# an sklearn pipeline that wraps XGBoost estimators.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def detect_gpu():
    """Check for NVIDIA GPU availability."""
    try:
        subprocess.run(
            ["nvidia-smi"], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("GPU detected — using CUDA acceleration (tree_method='hist', device='cuda').")
        return "cuda"
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("No GPU detected — using CPU (tree_method='hist', device='cpu').")
        return "cpu"


def build_pipeline(device):
    """Construct the full preprocessing + model pipeline."""

    # Numerical preprocessing: impute missing → scale
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # Categorical preprocessing: impute missing → one-hot encode
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, NUMERICAL_COLS),
        ("cat", cat_transformer, CATEGORICAL_COLS)
    ])

    # Full pipeline: preprocess → classify
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   XGBClassifier(
            tree_method="hist",
            device=device,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    return full_pipeline


def main():
    """Execute the full training workflow."""
    print("Starting model training pipeline...")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    X_train = pd.read_csv("Xtrain.csv")
    X_test  = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze()
    y_test  = pd.read_csv("ytest.csv").squeeze()

    print(f"Data loaded — Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 2. Build Pipeline ─────────────────────────────────────────────────
    device = detect_gpu()
    pipeline = build_pipeline(device)

    # ── 3. Define Hyperparameter Grid ─────────────────────────────────────
    # 3 × 3 × 2 × 2 × 2 = 72 combinations × 3 CV folds = 216 fits
    param_grid = {
        "classifier__n_estimators":     [100, 200, 300],
        "classifier__max_depth":        [3, 5, 7],
        "classifier__learning_rate":    [0.05, 0.1],
        "classifier__subsample":        [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\nHyperparameter grid: {total_combos} combinations × 3 CV folds = {total_combos * 3} fits")

    # ── 4. GridSearchCV ───────────────────────────────────────────────────
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",           # Optimize for F1 (handles class imbalance better than accuracy)
        n_jobs=-1,              # Use all available CPU cores for CV parallelism
        verbose=2,
        return_train_score=True
    )

    # ── 5. MLflow Experiment Tracking ─────────────────────────────────────
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("wellness-tourism-prediction")

    print("\nStarting MLflow experiment run...")
    with mlflow.start_run(run_name="xgboost-grid-search") as run:

        # Train with GridSearchCV
        print("Fitting GridSearchCV (this may take a few minutes)...")
        grid_search.fit(X_train, y_train)
        print("GridSearchCV training complete!")

        # ── Log Best Parameters ───────────────────────────────────────────
        best_params = grid_search.best_params_
        clean_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}

        print(f"\nBest Parameters:")
        for param, value in clean_params.items():
            print(f"   {param}: {value}")

        mlflow.log_params(clean_params)

        # ── Log CV Score ──────────────────────────────────────────────────
        best_cv_f1 = grid_search.best_score_
        mlflow.log_metric("best_cv_f1", best_cv_f1)
        print(f"\nBest CV F1 Score: {best_cv_f1:.4f}")

        # ── 6. Evaluate on Test Set ───────────────────────────────────────
        best_model = grid_search.best_estimator_
        y_pred  = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "test_accuracy":  accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall":    recall_score(y_test, y_pred),
            "test_f1":        f1_score(y_test, y_pred),
            "test_roc_auc":   roc_auc_score(y_test, y_proba),
        }

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        print(f"\n{'='*60}")
        print(f"  TEST SET EVALUATION METRICS")
        print(f"{'='*60}")
        for metric_name, metric_val in metrics.items():
            print(f"  {metric_name:20s}: {metric_val:.4f}")
        print(f"{'='*60}")

        # Classification Report
        report = classification_report(
            y_test, y_pred,
            target_names=["Not Purchased (0)", "Purchased (1)"]
        )
        print(f"\nClassification Report:\n{report}")

        # ── Log Model to MLflow ───────────────────────────────────────────
        mlflow.sklearn.log_model(
            best_model,
            "xgboost-pipeline",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        print(f"Model logged to MLflow (Run ID: {run.info.run_id})")

    # ── 7. Save Best Model for Deployment ─────────────────────────────────
    model_dir  = "tourism_project/deployment"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to '{model_path}'")
    print("Training pipeline complete!")


if __name__ == "__main__":
    main()
