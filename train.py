"""
Stage 2: Feature engineering + baseline + model training
Time-based split (train on earlier data, test on later) - mirrors real
deployment and avoids leakage, exactly as described in the project doc.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
model_mae = mean_absolute_error(y_test, pred)
model_rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"\n--- MODEL (XGBoost) ---")
print(f"MAE:  {model_mae:,.0f} kg")
print(f"RMSE: {model_rmse:,.0f} kg")

improvement = (baseline_mae - model_mae) / baseline_mae * 100
print(f"\n--- BACKTEST GATE ---")
print(f"Model improves MAE over baseline by {improvement:.1f}%")
print(f"Deployment gate: {'PASS - model beats baseline' if model_mae < baseline_mae else 'FAIL - model does not beat baseline'}")

# Feature importance - useful for explainability discussion
importances = pd.Series(model.feature_importances_, index=FEATURES + CATEGORICAL).sort_values(ascending=False)
print(f"\n--- Feature importances ---")
print(importances.to_string())

# Save model and test predictions for the next stage (worked savings example)
model.save_model(str(MODELS_DIR / "trip_fuel_model.json"))
test["predicted_trip_fuel_kg"] = pred
test.to_csv(DATA_DIR / "test_with_predictions.csv", index=False)
print("\nModel saved. Test set with predictions saved for evaluation stage.")
