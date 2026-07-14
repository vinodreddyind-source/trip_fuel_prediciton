"""
Fast smoke tests - run on every git push via CI. These are deliberately NOT
a replacement for the drift-triggered retrain workflow's full training and
evaluation - they exist to catch code-level bugs (broken imports, a syntax
error, an endpoint that no longer responds correctly) within seconds of a
push, rather than waiting for the next scheduled retrain cycle to discover
something is broken.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_prediction():
    payload = {
        "origin": "JFK", "dest": "LHR", "aircraft_type": "A320",
        "distance_nm": 3020, "takeoff_weight_klbs": 135,
        "headwind_kt": 15, "month": 1, "day_of_week": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_trip_fuel_kg" in body
    assert body["predicted_trip_fuel_kg"] > 0


def test_predict_endpoint_rejects_incomplete_payload():
    # Mirrors the real CloudWatch test from today - confirms the app's own
    # validation still correctly returns 422 for a missing-fields request,
    # rather than crashing or silently accepting bad input
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_result_is_physically_sensible():
    # A long-haul flight should always predict more fuel than a short-haul
    # one - a cheap sanity check that the model hasn't degenerated into
    # returning a constant or nonsensical value
    long_haul = {
        "origin": "JFK", "dest": "LHR", "aircraft_type": "A320",
        "distance_nm": 3020, "takeoff_weight_klbs": 135,
        "headwind_kt": 0, "month": 1, "day_of_week": 3,
    }
    short_haul = {
        "origin": "SFO", "dest": "SEA", "aircraft_type": "B737",
        "distance_nm": 680, "takeoff_weight_klbs": 115,
        "headwind_kt": 0, "month": 1, "day_of_week": 3,
    }
    long_pred = client.post("/predict", json=long_haul).json()["predicted_trip_fuel_kg"]
    short_pred = client.post("/predict", json=short_haul).json()["predicted_trip_fuel_kg"]
    assert long_pred > short_pred
