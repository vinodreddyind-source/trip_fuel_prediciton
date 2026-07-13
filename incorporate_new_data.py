"""
Simulates the "new data arrives" step in a real production pipeline: the
week2 batch (the same one that triggered drift detection) gets merged
into the main training dataset once a retrain is decided on. In a real
system this would be new FOQA/QAR records landing on a schedule; here
it's the same engineered batch, reused for the retrain step.
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

old = pd.read_csv(DATA_DIR / "flights.csv", parse_dates=["date"])
new = pd.read_csv(DATA_DIR / "week2_batch.csv", parse_dates=["date"])
new = new.drop(columns=["retrofitted"])  # simulation-only ground-truth label, not a real feature

combined = pd.concat([old, new], ignore_index=True).sort_values("date").reset_index(drop=True)
combined.to_csv(DATA_DIR / "flights.csv", index=False)
print(f"Incorporated {len(new)} new flights - dataset now has {len(combined)} total flights")
print(f"Date range now: {combined['date'].min()} to {combined['date'].max()}")
