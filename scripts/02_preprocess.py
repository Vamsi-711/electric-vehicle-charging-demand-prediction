"""
02_preprocess.py
-----------------
Cleans the raw EV charging sessions data and engineers features for
EDA and modeling.

Steps:
  1. Remove exact duplicate rows.
  2. Handle missing values (median imputation for numeric, drop rows
     missing user_id since it's not recoverable and not critical to keep).
  3. Detect & treat outliers using the IQR method on energy_kwh and
     duration_hours (capped, not dropped, to preserve session counts).
  4. Feature engineering: hour, day-of-week, month, season, is_weekend flags.
  5. Save cleaned session-level data + an hourly-aggregated demand table
     (the table used for forecasting).
"""

import numpy as np
import pandas as pd

RAW_PATH = "/home/claude/ev_project/data/raw_ev_charging_sessions.csv"
CLEAN_PATH = "/home/claude/ev_project/data/cleaned_ev_charging_sessions.csv"
HOURLY_PATH = "/home/claude/ev_project/data/hourly_station_demand.csv"

df = pd.read_csv(RAW_PATH, parse_dates=["start_time", "end_time"])
n_start = len(df)
print(f"Loaded raw data: {n_start:,} rows")

# --- 1. Remove exact duplicates -------------------------------------------------
df = df.drop_duplicates(subset=["session_id"]).reset_index(drop=True)
print(f"After removing duplicate session_ids: {len(df):,} rows "
      f"({n_start - len(df)} duplicates removed)")

# --- 2. Handle missing values ----------------------------------------------------
missing_before = df.isna().sum()
print("\nMissing values before cleaning:")
print(missing_before[missing_before > 0])

# user_id missing -> not critical for demand forecasting, fill as 'UNKNOWN'
df["user_id"] = df["user_id"].fillna("UNKNOWN")

# energy_kwh / duration_hours missing -> impute with the median for that
# station + location_type + hour-of-day bucket (more accurate than a global median)
df["start_hour_tmp"] = df["start_time"].dt.hour

for col in ["energy_kwh", "duration_hours"]:
    group_median = df.groupby(["location_type", "start_hour_tmp"])[col].transform("median")
    df[col] = df[col].fillna(group_median)
    df[col] = df[col].fillna(df[col].median())  # fallback for any remaining NaNs

df = df.drop(columns=["start_hour_tmp"])
print(f"\nMissing values after imputation: {df.isna().sum().sum()} total remaining")

# --- 3. Outlier detection & treatment (group-aware IQR, capped not dropped) -----
# NOTE: duration_hours is naturally right-skewed and its "normal" range differs
# a lot by location_type (e.g. Residential overnight sessions legitimately run
# much longer than a Downtown top-up). Running a single global IQR over the
# whole column would flag thousands of perfectly valid overnight sessions as
# outliers. Instead we compute IQR bounds *within each location_type group*,
# and use a wider fence (k=3.0, Tukey's "far out" threshold) since we only
# want to catch genuine sensor/logging errors (e.g. the 200-hour "stuck
# session" glitches injected upstream), not the natural long tail.
def cap_outliers_iqr_grouped(frame, col, group_col, k=3.0):
    def _bounds(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return pd.Series({"lower": max(q1 - k * iqr, 0), "upper": q3 + k * iqr})

    bounds = frame.groupby(group_col)[col].apply(_bounds).unstack()
    lower = frame[group_col].map(bounds["lower"])
    upper = frame[group_col].map(bounds["upper"])
    n_outliers = int(((frame[col] < lower) | (frame[col] > upper)).sum())
    capped = frame[col].clip(lower=lower, upper=upper)
    return capped, n_outliers, bounds

for col in ["energy_kwh", "duration_hours"]:
    df[col], n_out, bounds = cap_outliers_iqr_grouped(df, col, "location_type", k=3.0)
    print(f"Outliers capped in '{col}' (group-aware by location_type): {n_out} rows")
    print(bounds.round(2).to_string())

# --- 4. Feature engineering -------------------------------------------------------
df["start_hour"] = df["start_time"].dt.hour
df["day_of_week"] = df["start_time"].dt.day_name()
df["day_of_week_num"] = df["start_time"].dt.dayofweek
df["is_weekend"] = df["day_of_week_num"] >= 5
df["month"] = df["start_time"].dt.month
df["date"] = df["start_time"].dt.date

season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["season"] = df["month"].map(season_map)

df.to_csv(CLEAN_PATH, index=False)
print(f"\nSaved cleaned session-level data -> {CLEAN_PATH} ({len(df):,} rows)")

# --- 5. Build hourly demand aggregation (for forecasting) -------------------------
df["date_hour"] = df["start_time"].dt.floor("h")

hourly = (
    df.groupby(["station_id", "location_type", "date_hour"])
    .agg(
        n_sessions=("session_id", "count"),
        total_energy_kwh=("energy_kwh", "sum"),
        avg_duration_hours=("duration_hours", "mean"),
    )
    .reset_index()
)

# Fill in hours with zero sessions (important for time-series continuity)
all_hours = pd.date_range(df["start_time"].min().floor("h"),
                           df["start_time"].max().floor("h"), freq="h")
stations = df["station_id"].unique()
full_index = pd.MultiIndex.from_product([stations, all_hours], names=["station_id", "date_hour"])
hourly_full = hourly.set_index(["station_id", "date_hour"]).reindex(full_index).reset_index()

loc_map = df.drop_duplicates("station_id").set_index("station_id")["location_type"]
hourly_full["location_type"] = hourly_full["station_id"].map(loc_map)
hourly_full["n_sessions"] = hourly_full["n_sessions"].fillna(0)
hourly_full["total_energy_kwh"] = hourly_full["total_energy_kwh"].fillna(0)
hourly_full["avg_duration_hours"] = hourly_full["avg_duration_hours"].fillna(0)

hourly_full["hour"] = hourly_full["date_hour"].dt.hour
hourly_full["day_of_week"] = hourly_full["date_hour"].dt.dayofweek
hourly_full["is_weekend"] = hourly_full["day_of_week"] >= 5
hourly_full["month"] = hourly_full["date_hour"].dt.month
hourly_full["season"] = hourly_full["month"].map(season_map)

hourly_full.to_csv(HOURLY_PATH, index=False)
print(f"Saved hourly station-demand table -> {HOURLY_PATH} ({len(hourly_full):,} rows)")

print("\nPreprocessing complete.")
