"""
Stage 1: Synthetic data generation
Mimics the real feature set from the Trip Fuel Prediction project:
route/distance, aircraft type, weight, weather (headwind), and a genuine
per-flight fuel-burn "measurement" (with realistic noise) as the label.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Anchor paths to this script's own location, not the current working
# directory - makes this portable across laptop / Docker / AWS ECS,
# since each of those has a different "current directory" but the
# script's own location relative to the project is always the same.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

N_FLIGHTS = 15000

# --- Route network: (origin, dest, great-circle distance in nm) ---
ROUTES = [
    ("JFK", "LAX", 2145), ("JFK", "LHR", 3020), ("ORD", "DFW", 720),
    ("SFO", "SEA", 680),  ("ATL", "MIA", 595),  ("BOS", "ORD", 720),
    ("LAX", "SFO", 300),  ("DFW", "DEN", 560),  ("JFK", "MIA", 1090),
    ("SEA", "DEN", 890),
]

AIRCRAFT = {
    # type: (empty_weight_klbs, max_payload_klbs, fuel_burn_per_nm_base)
    "A320":  (93,  40, 6.8),
    "A321":  (99,  46, 7.4),
    "B737":  (91,  42, 6.9),
    "B738":  (95,  46, 7.1),
    "B789":  (254, 90, 12.5),
}

aircraft_types = list(AIRCRAFT.keys())

rows = []
start_date = datetime(2023, 1, 1)

for i in range(N_FLIGHTS):
    origin, dest, distance = ROUTES[rng.integers(0, len(ROUTES))]
    ac = aircraft_types[rng.integers(0, len(aircraft_types))]
    empty_w, max_payload, base_burn = AIRCRAFT[ac]

    # Flight date spread across ~2 years, with mild seasonal weight pattern
    day_offset = rng.integers(0, 730)
    flight_date = start_date + timedelta(days=int(day_offset))
    month = flight_date.month

    # Payload: passenger + cargo weight, with seasonal bump in Nov-Dec
    seasonal_factor = 1.15 if month in (11, 12) else 1.0
    payload = rng.uniform(0.5, 1.0) * max_payload * seasonal_factor
    takeoff_weight = empty_w + payload

    # Weather: headwind component (knots), positive = headwind (burns more fuel)
    headwind = rng.normal(loc=8, scale=18)  # can be negative (tailwind)

    # Historical average trip fuel by (route, aircraft) - the naive baseline
    # computed later; for now just generate the TRUE fuel burn with noise.

    # --- The "real" relationship the model has to learn ---
    # Base fuel scales with distance and aircraft burn rate
    base_fuel = distance * base_burn
    # Heavier takeoff weight increases fuel burn (roughly linear effect)
    weight_effect = (takeoff_weight - empty_w) * 0.9
    # Headwind increases fuel burn; tailwind reduces it (per nm of distance)
    wind_effect = headwind * distance * 0.018
    # Small nonlinear altitude/efficiency interaction (longer flights = slightly better fuel/nm)
    efficiency_bonus = -0.05 * distance if distance > 1500 else 0

    true_fuel = base_fuel + weight_effect + wind_effect + efficiency_bonus

    # Real-world measurement noise (real FOQA data isn't perfectly clean either)
    measurement_noise = rng.normal(0, true_fuel * 0.04)
    actual_fuel = max(true_fuel + measurement_noise, base_fuel * 0.5)

    rows.append({
        "flight_id": f"FL{i:06d}",
        "date": flight_date.strftime("%Y-%m-%d"),
        "origin": origin, "dest": dest, "distance_nm": distance,
        "aircraft_type": ac,
        "takeoff_weight_klbs": round(takeoff_weight, 1),
        "headwind_kt": round(headwind, 1),
        "month": month,
        "day_of_week": flight_date.weekday(),
        "actual_trip_fuel_kg": round(actual_fuel * 1.2, 0),  # scale to kg-ish units
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df.to_csv(DATA_DIR / "flights.csv", index=False)

print(f"Generated {len(df)} synthetic flights")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSample rows:")
print(df.head(5).to_string(index=False))
print(f"\nFuel burn stats (kg):")
print(df["actual_trip_fuel_kg"].describe())
