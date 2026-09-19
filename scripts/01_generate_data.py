"""
01_generate_data.py
--------------------
Generates a realistic synthetic EV charging sessions dataset.

Why synthetic? No real dataset was supplied with the project brief. This
generator encodes the same statistical patterns found in real public EV
datasets (ACN-Data, Palo Alto/Boulder open charging data):
  - Strong daily seasonality (morning + evening peaks)
  - Weekday vs weekend behavior differences
  - Monthly/seasonal demand shifts (more charging in winter due to heating,
    slightly less driving in extreme summer heat regions, etc.)
  - Station-level heterogeneity (some stations are just busier)
  - Realistic noise, missing values, and outliers so the preprocessing step
    has real work to do.

Replace this script with a real data loader if/when an actual dataset
(e.g. a Kaggle CSV) is provided -- keep the same output schema and every
downstream script keeps working unchanged.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_STATIONS = 25
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
N_DAYS = (END_DATE - START_DATE).days + 1

STATION_IDS = [f"STN_{i:03d}" for i in range(1, N_STATIONS + 1)]
LOCATIONS = RNG.choice(
    ["Downtown", "Mall", "Office_Park", "Residential", "Highway_Rest_Stop"],
    size=N_STATIONS,
)
station_location_map = dict(zip(STATION_IDS, LOCATIONS))

# Each station has an intrinsic "popularity" multiplier
station_popularity = dict(zip(STATION_IDS, RNG.gamma(shape=3.0, scale=1.0, size=N_STATIONS)))

LOCATION_HOURLY_PROFILE = {
    # 24-hour relative weights per location type
    "Downtown":            [1,1,1,1,1,2,4,7,8,6,5,5,6,5,5,6,7,8,7,5,4,3,2,1],
    "Mall":                [1,1,1,1,1,1,1,2,3,5,7,8,8,7,7,8,8,7,6,5,4,3,2,1],
    "Office_Park":         [1,1,1,1,1,2,5,8,9,8,6,5,5,5,6,7,6,4,2,1,1,1,1,1],
    "Residential":         [6,5,4,3,3,3,3,4,3,2,2,2,2,2,2,3,4,6,8,9,9,8,7,6],
    "Highway_Rest_Stop":   [3,3,2,2,2,3,4,5,5,5,5,5,6,6,6,6,6,6,5,5,4,4,4,3],
}

records = []
session_id = 1

for day_offset in range(N_DAYS):
    current_date = START_DATE + timedelta(days=day_offset)
    day_of_week = current_date.weekday()  # 0=Mon
    is_weekend = day_of_week >= 5
    month = current_date.month
    # Seasonal multiplier: winter months slightly higher demand (heating + shorter EV range)
    seasonal_mult = 1.15 if month in (12, 1, 2) else (0.95 if month in (6, 7, 8) else 1.0)
    # Mild year-over-year growth in EV adoption
    growth_mult = 1.0 + 0.25 * (day_offset / N_DAYS)

    for station in STATION_IDS:
        loc = station_location_map[station]
        profile = LOCATION_HOURLY_PROFILE[loc]
        popularity = station_popularity[station]
        weekend_adjust = 0.7 if (is_weekend and loc == "Office_Park") else (
            1.3 if (is_weekend and loc in ("Mall", "Residential")) else 1.0
        )

        # Expected number of sessions today at this station
        base_daily_sessions = popularity * 1.8 * seasonal_mult * growth_mult * weekend_adjust
        n_sessions_today = RNG.poisson(lam=max(base_daily_sessions, 0.1))

        if n_sessions_today == 0:
            continue

        hour_weights = np.array(profile, dtype=float)
        hour_weights = hour_weights / hour_weights.sum()
        start_hours = RNG.choice(24, size=n_sessions_today, p=hour_weights)

        for h in start_hours:
            minute = int(RNG.integers(0, 60))
            start_dt = current_date.replace(hour=int(h), minute=minute)

            # Duration: lognormal, longer sessions overnight/residential
            if loc == "Residential" and (h >= 20 or h <= 6):
                duration_hours = RNG.lognormal(mean=1.9, sigma=0.4)  # long overnight charges
            else:
                duration_hours = RNG.lognormal(mean=0.6, sigma=0.6)
            duration_hours = float(np.clip(duration_hours, 0.1, 14))

            # Energy consumed correlates with duration but with charger-power noise
            charger_kw = RNG.choice([7, 11, 22, 50, 150], p=[0.25, 0.25, 0.25, 0.15, 0.10])
            energy_kwh = duration_hours * charger_kw * RNG.uniform(0.35, 0.85)
            energy_kwh = float(np.clip(energy_kwh, 0.5, 120))

            end_dt = start_dt + timedelta(hours=duration_hours)

            records.append({
                "session_id": f"S{session_id:07d}",
                "station_id": station,
                "location_type": loc,
                "user_id": f"U{int(RNG.integers(1, 4000)):05d}",
                "start_time": start_dt,
                "end_time": end_dt,
                "duration_hours": round(duration_hours, 2),
                "energy_kwh": round(energy_kwh, 2),
                "charger_kw": charger_kw,
            })
            session_id += 1

df = pd.DataFrame.from_records(records)
df = df.sort_values("start_time").reset_index(drop=True)

print(f"Generated {len(df):,} charging sessions across {N_STATIONS} stations "
      f"from {START_DATE.date()} to {END_DATE.date()}")

# ---------------------------------------------------------------
# Inject realistic data-quality issues so preprocessing has real work
# ---------------------------------------------------------------
n = len(df)

# 1. Missing values (MCAR-ish, ~2-4% per column on a few columns)
for col, frac in [("energy_kwh", 0.03), ("duration_hours", 0.02), ("user_id", 0.015)]:
    idx = RNG.choice(n, size=int(n * frac), replace=False)
    df.loc[idx, col] = np.nan

# 2. Outliers: a small number of absurd energy/duration values (sensor glitches)
outlier_idx = RNG.choice(n, size=int(n * 0.008), replace=False)
for i in outlier_idx:
    if RNG.random() < 0.5:
        df.loc[i, "energy_kwh"] = float(RNG.uniform(500, 2000))  # impossible energy spike
    else:
        df.loc[i, "duration_hours"] = float(RNG.uniform(48, 200))  # stuck session

# 3. A few duplicate rows (logging glitches)
dupe_idx = RNG.choice(n, size=int(n * 0.002), replace=False)
df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

df.to_csv("/home/claude/ev_project/data/raw_ev_charging_sessions.csv", index=False)
print(f"Saved raw dataset with injected issues: {len(df):,} rows -> data/raw_ev_charging_sessions.csv")
print(f"  Missing values introduced in: energy_kwh, duration_hours, user_id")
print(f"  Outliers introduced: {len(outlier_idx)} rows")
print(f"  Duplicate rows introduced: {len(dupe_idx)} rows")
