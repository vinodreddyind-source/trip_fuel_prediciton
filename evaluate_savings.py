"""
Stage 3: The worked savings calculation - exactly the methodology from
Section 5.1 of the Trip Fuel document, run on real (synthetic) output.
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

test = pd.read_csv(DATA_DIR / "test_with_predictions.csv")

THRESHOLD_PCT = 0.05  # flag flights where actual exceeds predicted by more than 5%

test["excess_kg"] = test["actual_trip_fuel_kg"] - test["predicted_trip_fuel_kg"]
test["excess_pct"] = test["excess_kg"] / test["predicted_trip_fuel_kg"]
test["flagged"] = test["excess_pct"] > THRESHOLD_PCT

n_flagged = test["flagged"].sum()
pct_flagged = n_flagged / len(test) * 100

total_fuel = test["actual_trip_fuel_kg"].sum()
excess_fuel_flagged = test.loc[test["flagged"], "excess_kg"].sum()
savings_pct = excess_fuel_flagged / total_fuel * 100

print(f"Test set: {len(test)} flights")
print(f"Flagged as inefficient (>{THRESHOLD_PCT*100:.0f}% over prediction): {n_flagged} flights ({pct_flagged:.1f}%)")
print(f"Total fuel burned across test set: {total_fuel:,.0f} kg")
print(f"Excess fuel across flagged flights: {excess_fuel_flagged:,.0f} kg")
print(f"\n>>> Potential savings (backtested): {savings_pct:.2f}% of total fuel <<<")

print(f"\n--- Top 5 most flagged routes (worth prioritizing efficiency review) ---")
route_summary = (
    test[test["flagged"]]
    .groupby(["origin", "dest", "aircraft_type"])
    .agg(flagged_flights=("flight_id", "count"), avg_excess_kg=("excess_kg", "mean"))
    .sort_values("flagged_flights", ascending=False)
    .head(5)
)
print(route_summary.to_string())
