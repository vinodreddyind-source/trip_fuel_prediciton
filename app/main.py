"""
Stage 4: FastAPI serving layer.
Loads the trained model once at startup, exposes /predict for single-flight
lookups and /health for container orchestration readiness checks -
mirrors the architecture in Section 8 of the Trip Fuel document.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI(title="Trip Fuel Prediction Service")

model = xgb.XGBRegressor()
model.load_model("models/trip_fuel_model.json")

FEATURES = ["distance_nm", "takeoff_weight_klbs", "headwind_kt", "month", "day_of_week"]
CATEGORICAL = ["origin", "dest", "aircraft_type"]


class FlightRequest(BaseModel):
    origin: str
    dest: str
    aircraft_type: str
    distance_nm: float
    takeoff_weight_klbs: float
    headwind_kt: float
    month: int
    day_of_week: int


class FlightResponse(BaseModel):
    predicted_trip_fuel_kg: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=FlightResponse)
def predict(req: FlightRequest):
    row = pd.DataFrame([req.model_dump()])
    for col in CATEGORICAL:
        row[col] = row[col].astype("category")
    pred = model.predict(row[FEATURES + CATEGORICAL])
    return {"predicted_trip_fuel_kg": round(float(pred[0]), 1)}
