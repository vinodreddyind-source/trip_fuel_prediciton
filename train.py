"""
Stage 2: Feature engineering + baseline + model training
Time-based split (train on earlier data, test on later) - mirrors real
deployment and avoids leakage, exactly as described in the project doc.

Now with MLflow: every run logs params/metrics/model, and a model is only
registered + promoted to the "production" alias if it beats the baseline
gate - the same principle as the backtest-as-deployment-gate in the real
Trip Fuel document, just made concrete and automatic here.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from pathlib import Path
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow setup ---
# Tracking metadata (params, metrics, run history) lives in a local SQLite
# file - no server needs to run 24/7 for this. Model/artifact FILES go to
# S3 - same bucket as the DVC data, different prefix, so everything
# related to this project lives in one place.
MLFLOW_DB = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
S3_ARTIFACT_ROOT = "s3://trip-fuel-mlops-vinod/mlflow-artifacts"
EXPERIMENT_NAME = "trip_fuel_prediction"
REGISTERED_MODEL_NAME = "trip_fuel_model"

mlflow.set_tracking_uri(MLFLOW_DB)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=S3_ARTIFACT_ROOT)
else:
    experiment_id = experiment.experiment_id

df = pd.read_csv(DATA_DIR / "flights.csv", parse_dates=["date"])

# --- Time-based split: train on 2023, test on 2024 (mirrors production reality) ---
train = df[df["date"] < "2024-01-01"].copy()
test  = df[df["date"] >= "2024-01-01"].copy()
print(f"Train: {len(train)} flights (2023)")
print(f"Test:  {len(test)} flights (2024)")

# --- BASELINE: rolling historical average trip fuel by (route, aircraft type) ---
# Computed ONLY from train data - this is what the real model has to beat.
baseline_lookup = (
    train.groupby(["origin", "dest", "aircraft_type"])["actual_trip_fuel_kg"]
    .mean()
    .rename("baseline_pred")
    .reset_index()
)

test = test.merge(baseline_lookup, on=["origin", "dest", "aircraft_type"], how="left")
# Fallback for any route/aircraft combo not seen in training (shouldn't happen here, but real-world would)
test["baseline_pred"] = test["baseline_pred"].fillna(train["actual_trip_fuel_kg"].mean())

baseline_mae = mean_absolute_error(test["actual_trip_fuel_kg"], test["baseline_pred"])
baseline_rmse = np.sqrt(mean_squared_error(test["actual_trip_fuel_kg"], test["baseline_pred"]))
print(f"\n--- BASELINE (historical average lookup) ---")
print(f"MAE:  {baseline_mae:,.0f} kg")
print(f"RMSE: {baseline_rmse:,.0f} kg")

# --- FEATURES for the real model ---
FEATURES = ["distance_nm", "takeoff_weight_klbs", "headwind_kt", "month", "day_of_week"]
CATEGORICAL = ["origin", "dest", "aircraft_type"]

for col in CATEGORICAL:
    df[col] = df[col].astype("category")

train = df[df["date"] < "2024-01-01"].copy()
test = df[df["date"] >= "2024-01-01"].copy()

X_train, y_train = train[FEATURES + CATEGORICAL], train["actual_trip_fuel_kg"]
X_test, y_test = test[FEATURES + CATEGORICAL], test["actual_trip_fuel_kg"]

MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    random_state=42,
)

with mlflow.start_run(experiment_id=experiment_id) as run:
    print(f"\nMLflow run ID: {run.info.run_id}")

    mlflow.log_params(MODEL_PARAMS)
    mlflow.log_param("train_size", len(train))
    mlflow.log_param("test_size", len(test))

    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, pred)
    model_rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"\n--- MODEL (XGBoost) ---")
    print(f"MAE:  {model_mae:,.0f} kg")
    print(f"RMSE: {model_rmse:,.0f} kg")

    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    gate_passed = model_mae < baseline_mae
    print(f"\n--- BACKTEST GATE ---")
    print(f"Model improves MAE over baseline by {improvement:.1f}%")
    print(f"Deployment gate: {'PASS - model beats baseline' if gate_passed else 'FAIL - model does not beat baseline'}")

    mlflow.log_metric("baseline_mae", baseline_mae)
    mlflow.log_metric("baseline_rmse", baseline_rmse)
    mlflow.log_metric("model_mae", model_mae)
    mlflow.log_metric("model_rmse", model_rmse)
    mlflow.log_metric("improvement_pct", improvement)
    mlflow.log_metric("gate_passed", int(gate_passed))

    # Feature importance - useful for explainability discussion
    importances = pd.Series(model.feature_importances_, index=FEATURES + CATEGORICAL).sort_values(ascending=False)
    print(f"\n--- Feature importances ---")
    print(importances.to_string())
    for feat, imp in importances.items():
        mlflow.log_metric(f"importance_{feat}", float(imp))

    # Log the model to this run's artifacts (S3) regardless of gate result -
    # every run is kept for history/audit, even ones that don't get promoted
    mlflow.xgboost.log_model(model, name="model")

    if gate_passed:
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", registered.version)
        print(f"\nModel registered as '{REGISTERED_MODEL_NAME}' v{registered.version} "
              f"and promoted to the 'production' alias")
    else:
        print(f"\nGate FAILED - model logged for history but NOT registered/promoted. "
              f"The currently-aliased 'production' model, if any, remains in place.")

    # Save model and test predictions locally too - for the evaluation script
    # and for the FastAPI service, which loads directly from disk rather than
    # querying MLflow at request time (keeps inference simple and fast)
    model.save_model(str(MODELS_DIR / "trip_fuel_model.json"))
    test["predicted_trip_fuel_kg"] = pred
    test.to_csv(DATA_DIR / "test_with_predictions.csv", index=False)
    print("\nModel saved locally. Test set with predictions saved for evaluation stage.")
