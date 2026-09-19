"""
05_dashboard_data.py
---------------------
Aggregates cleaned data + model outputs into a single compact JSON that
powers the self-contained HTML dashboard artifact (no backend, so all
data the dashboard needs must be pre-computed and embedded).
"""

import json
import numpy as np
import pandas as pd

CLEAN_PATH = "E:\\VS CODE\\ev_project\\data\\cleaned_ev_charging_sessions.csv"
MODEL_DIR = "E:\\VS CODE\\ev_project\\outputs\\models"
OUT_PATH = "E:\\VS CODE\\ev_project\\outputs\\dashboard_data\\dashboard_data.json"

df = pd.read_csv(CLEAN_PATH, parse_dates=["start_time", "end_time"])
df["date"] = pd.to_datetime(df["date"])

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
season_order = ["Winter", "Spring", "Summer", "Fall"]

# --- Summary KPIs -------------------------------------------------------------
total_sessions = int(len(df))
total_energy = float(df["energy_kwh"].sum())
avg_duration = float(df["duration_hours"].mean())
avg_energy = float(df["energy_kwh"].mean())
n_stations = int(df["station_id"].nunique())
date_min, date_max = df["date"].min().strftime("%b %Y"), df["date"].max().strftime("%b %Y")

# YoY growth: compare first 3 months vs last 3 months of available data
first_q = df[df["date"] < df["date"].min() + pd.Timedelta(days=90)]
last_q = df[df["date"] > df["date"].max() - pd.Timedelta(days=90)]
growth_pct = float((len(last_q) - len(first_q)) / max(len(first_q), 1) * 100)

# --- Daily trend (for line chart) ----------------------------------------------
daily = df.groupby("date").size().reset_index(name="count")
daily["rolling_avg"] = daily["count"].rolling(14, min_periods=1).mean()
daily_trend = {
    "dates": daily["date"].dt.strftime("%Y-%m-%d").tolist(),
    "counts": daily["count"].tolist(),
    "rolling_avg": [round(x, 1) for x in daily["rolling_avg"].tolist()],
}

# --- Heatmap: hour x day-of-week -------------------------------------------------
pivot = df.pivot_table(index="day_of_week", columns="start_hour", values="session_id", aggfunc="count")
pivot = pivot.reindex(day_order).fillna(0)
heatmap = {"days": day_order, "hours": list(range(24)), "matrix": pivot.values.astype(int).tolist()}

# --- Weekday vs weekend hourly profile -------------------------------------------
wk = df.groupby(["is_weekend", "start_hour"]).size().unstack(level=0).fillna(0)
wk.columns = [str(c) for c in wk.columns]
weekday_col = "False" if "False" in wk.columns else False
weekend_col = "True" if "True" in wk.columns else True
weekday_vs_weekend = {
    "hours": list(range(24)),
    "weekday": wk.get(weekday_col, wk.iloc[:, 0]).reindex(range(24), fill_value=0).astype(int).tolist(),
    "weekend": wk.get(weekend_col, wk.iloc[:, -1]).reindex(range(24), fill_value=0).astype(int).tolist(),
}

# --- Seasonal demand --------------------------------------------------------------
seasonal = df.groupby("season").size().reindex(season_order).fillna(0)
seasonal_demand = {"seasons": season_order, "counts": seasonal.astype(int).tolist()}

# --- Station leaderboard (top 15) -------------------------------------------------
station_counts = df.groupby(["station_id", "location_type"]).size().reset_index(name="count")
station_counts = station_counts.sort_values("count", ascending=False).head(15)
station_leaderboard = station_counts.to_dict(orient="records")

# --- Energy by location (range summary for a custom range-bar chart) --------------
energy_stats = df.groupby("location_type")["energy_kwh"].describe(percentiles=[.25, .5, .75])
energy_stats = energy_stats.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
energy_by_location = []
for loc, row in energy_stats.iterrows():
    energy_by_location.append({
        "location": loc,
        "min": round(float(row["min"]), 1),
        "q1": round(float(row["q1"]), 1),
        "median": round(float(row["median"]), 1),
        "q3": round(float(row["q3"]), 1),
        "max": round(float(row["max"]), 1),
        "mean": round(float(row["mean"]), 1),
    })
energy_by_location = sorted(energy_by_location, key=lambda x: -x["median"])

# --- Monthly energy trend ---------------------------------------------------------
df["year_month"] = df["start_time"].dt.to_period("M").astype(str)
monthly_energy = df.groupby("year_month")["energy_kwh"].sum().reset_index()
monthly_energy_trend = {
    "months": monthly_energy["year_month"].tolist(),
    "energy_kwh": [round(x, 0) for x in monthly_energy["energy_kwh"].tolist()],
}

# --- Model comparison + actual vs predicted (from modeling script outputs) --------
model_comparison = pd.read_csv(f"{MODEL_DIR}/model_comparison.csv").round(3).to_dict(orient="records")

pred_df = pd.read_csv(f"{MODEL_DIR}/network_wide_predictions.csv", parse_dates=["date_hour"])
# Downsample to daily totals for a cleaner dashboard chart (61 days vs 1464 hourly points)
pred_df["date"] = pred_df["date_hour"].dt.date
pred_daily = pred_df.groupby("date")[["actual", "predicted"]].sum().reset_index()
actual_vs_predicted = {
    "dates": [d.strftime("%Y-%m-%d") for d in pred_daily["date"]],
    "actual": pred_daily["actual"].round(1).tolist(),
    "predicted": pred_daily["predicted"].round(1).tolist(),
}

feature_importance = pd.read_csv(f"{MODEL_DIR}/feature_importance.csv", index_col=0)
feature_importance.columns = ["importance"]
feature_importance = feature_importance.sort_values("importance", ascending=False).head(8)
feature_importance_list = [
    {"feature": idx, "importance": round(float(val), 4)}
    for idx, val in feature_importance["importance"].items()
]

# --- Capacity flags: stations running hot in peak hours (17:00-19:00) -------------
peak = df[df["start_hour"].isin([17, 18, 19])]
peak_util = peak.groupby("station_id").size().sort_values(ascending=False)
hot_stations = peak_util.head(5).index.tolist()
hot_station_details = peak[peak["station_id"].isin(hot_stations)].groupby(
    ["station_id", "location_type"]
).size().reset_index(name="peak_sessions").sort_values("peak_sessions", ascending=False)
capacity_flags = hot_station_details.to_dict(orient="records")

# --- Insight strings for the dashboard --------------------------------------------
peak_hour = int(df["start_hour"].value_counts().idxmax())
busiest_day = df["day_of_week"].value_counts().idxmax()
top_station_row = df["station_id"].value_counts().head(1)
top_station = top_station_row.index[0]
top_station_count = int(top_station_row.iloc[0])

payload = {
    "kpis": {
        "total_sessions": total_sessions,
        "total_energy_kwh": round(total_energy, 0),
        "avg_duration_hours": round(avg_duration, 2),
        "avg_energy_kwh": round(avg_energy, 2),
        "n_stations": n_stations,
        "date_range": f"{date_min} – {date_max}",
        "growth_pct": round(growth_pct, 1),
        "peak_hour": peak_hour,
        "busiest_day": busiest_day,
        "top_station": top_station,
        "top_station_count": top_station_count,
    },
    "daily_trend": daily_trend,
    "heatmap": heatmap,
    "weekday_vs_weekend": weekday_vs_weekend,
    "seasonal_demand": seasonal_demand,
    "station_leaderboard": station_leaderboard,
    "energy_by_location": energy_by_location,
    "monthly_energy_trend": monthly_energy_trend,
    "model_comparison": model_comparison,
    "actual_vs_predicted": actual_vs_predicted,
    "feature_importance": feature_importance_list,
    "capacity_flags": capacity_flags,
}

with open(OUT_PATH, "w") as f:
    json.dump(payload, f)

print(f"Dashboard data saved -> {OUT_PATH}")
print(f"File size: {__import__('os').path.getsize(OUT_PATH) / 1024:.1f} KB")
