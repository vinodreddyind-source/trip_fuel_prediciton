"""
Stage: "New data arrives" simulation, with two DELIBERATE, DISTINCT drift
scenarios engineered in - not random noise, so there's something genuine
and explainable for the drift detector to catch.

Scenario 1 - DATA (feature) drift: a seasonal shift makes headwinds
stronger and more frequent on the JFK-LHR route (simulating a winter
jet-stream pattern) - the INPUT distribution changes.

Scenario 2 - CONCEPT drift: a fleet-wide winglet retrofit reduces actual
fuel burn by ~8% relative to what the ORIGINAL formula would predict -
the RELATIONSHIP between inputs and output changes, even where the inputs
themselves look similar to before. This is the more subtle, more
important case: a model can look "fine" on input distributions while
being systematically wrong, which is exactly why residual/concept drift
monitoring is a separate check from data drift monitoring.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

rng = np.random.default_rng(99)  # different seed - this is genuinely new data

N_FLIGHTS = 3000

ROUTES = [
    ("JFK", "LAX", 2145), ("JFK", "LHR", 3020), ("ORD", "DFW", 720),
    ("SFO", "SEA", 680),  ("ATL", "MIA", 595),  ("BOS", "ORD", 720),
    ("LAX", "SFO", 300),  ("DFW", "DEN", 560),  ("JFK", "MIA", 1090),
    ("SEA", "DEN", 890),
]

AIRCRAFT = {
    "A320":  (93,  40, 6.8),
    "A321":  (99,  46, 7.4),
    "B737":  (91,  42, 6.9),
    "B738":  (95,  46, 7.1),
    "B789":  (254, 90, 12.5),
}
aircraft_types = list(AIRCRAFT.keys())

# Fleet-wide winglet retrofit: rolled out gradually, complete by this date
RETROFIT_COMPLETE_DATE = datetime(2025, 2, 1)
RETROFIT_FUEL_REDUCTION = 0.08  # 8% less fuel burn once retrofitted

rows = []
start_date = datetime(2025, 1, 1)  # picks up right after the original 2023-2024 data

for i in range(N_FLIGHTS):
    origin, dest, distance = ROUTES[rng.integers(0, len(ROUTES))]
    ac = aircraft_types[rng.integers(0, len(aircraft_types))]
    empty_w, max_payload, base_burn = AIRCRAFT[ac]

    day_offset = rng.integers(0, 90)  # Jan-Mar 2025
    flight_date = start_date + timedelta(days=int(day_offset))
    month = flight_date.month

    seasonal_factor = 1.15 if month in (11, 12) else 1.0
    payload = rng.uniform(0.5, 1.0) * max_payload * seasonal_factor
    takeoff_weight = empty_w + payload

    # --- SCENARIO 1: DATA DRIFT ---
    # JFK-LHR specifically sees a seasonal jet-stream shift: stronger,
    # more consistent headwinds westbound in this window than historically
    if origin == "JFK" and dest == "LHR":
        headwind = rng.normal(loc=28, scale=12)  # was loc=8 in original data
    else:
        headwind = rng.normal(loc=8, scale=18)  # unchanged elsewhere

    base_fuel = distance * base_burn
    weight_effect = (takeoff_weight - empty_w) * 0.9
    wind_effect = headwind * distance * 0.018
    efficiency_bonus = -0.05 * distance if distance > 1500 else 0

    true_fuel = base_fuel + weight_effect + wind_effect + efficiency_bonus

    # --- SCENARIO 2: CONCEPT DRIFT ---
    # Winglet retrofit changes the underlying fuel-burn RELATIONSHIP,
    # not the input distribution - the model's old learned relationship
    # between features and fuel burn no longer holds after this date
    if flight_date >= RETROFIT_COMPLETE_DATE:
        true_fuel *= (1 - RETROFIT_FUEL_REDUCTION)

    measurement_noise = rng.normal(0, true_fuel * 0.04)
    actual_fuel = max(true_fuel + measurement_noise, base_fuel * 0.5)

    rows.append({
        "flight_id": f"W2{i:06d}",
        "date": flight_date.strftime("%Y-%m-%d"),
        "origin": origin, "dest": dest, "distance_nm": distance,
        "aircraft_type": ac,
        "takeoff_weight_klbs": round(takeoff_weight, 1),
        "headwind_kt": round(headwind, 1),
        "month": month,
        "day_of_week": flight_date.weekday(),
        "actual_trip_fuel_kg": round(actual_fuel * 1.2, 0),
        "retrofitted": flight_date >= RETROFIT_COMPLETE_DATE,  # ground truth label, for our own validation only - NOT a model feature
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df.to_csv(DATA_DIR / "week2_batch.csv", index=False)

print(f"Generated {len(df)} 'week 2' flights (Jan-Mar 2025)")
print(f"Retrofitted flights: {df['retrofitted'].sum()} / {len(df)}")
print(f"\nJFK-LHR headwind comparison (data drift check):")
jfk_lhr = df[(df.origin == "JFK") & (df.dest == "LHR")]
print(f"  Week 2 batch mean headwind: {jfk_lhr['headwind_kt'].mean():.1f} kt (was ~8 kt in original training data)")
print(f"\nSaved to {DATA_DIR / 'week2_batch.csv'}")
