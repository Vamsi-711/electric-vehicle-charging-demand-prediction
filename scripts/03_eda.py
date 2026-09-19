"""
03_eda.py
---------
Exploratory Data Analysis on the cleaned EV charging dataset.
Produces the charts called for in the abstract: line charts, bar graphs,
heat maps -- covering charging frequency, peak demand hours, duration,
energy consumption, seasonal variation, and user behavior.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 110

CLEAN_PATH = "/home/claude/ev_project/data/cleaned_ev_charging_sessions.csv"
OUT_DIR = "/home/claude/ev_project/outputs/eda_charts"

df = pd.read_csv(CLEAN_PATH, parse_dates=["start_time", "end_time"])
df["date"] = pd.to_datetime(df["date"])

# 1. Daily charging frequency over time (line chart) -----------------------------
daily_counts = df.groupby("date").size()
plt.figure(figsize=(12, 4.5))
daily_counts.plot(color="#2E86AB", linewidth=0.9)
daily_counts.rolling(14).mean().plot(color="#E63946", linewidth=2, label="14-day rolling avg")
plt.title("Daily Charging Session Volume (2023–2024)")
plt.xlabel("Date"); plt.ylabel("Number of Sessions")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_daily_sessions_trend.png")
plt.close()

# 2. Peak demand hours -- heatmap of hour vs day-of-week --------------------------
pivot = df.pivot_table(index="day_of_week", columns="start_hour",
                        values="session_id", aggfunc="count")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = pivot.reindex(day_order)
plt.figure(figsize=(13, 5))
sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Session Count"})
plt.title("Charging Demand Heatmap: Hour of Day vs Day of Week")
plt.xlabel("Hour of Day"); plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_hour_day_heatmap.png")
plt.close()

# 3. Charging duration distribution (histogram) -----------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df["duration_hours"], bins=50, kde=True, color="#457B9D")
plt.title("Distribution of Charging Session Duration")
plt.xlabel("Duration (hours)"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_duration_distribution.png")
plt.close()

# 4. Energy consumption by location type (box plot) -------------------------------
plt.figure(figsize=(9, 5))
order = df.groupby("location_type")["energy_kwh"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="location_type", y="energy_kwh", order=order, palette="crest")
plt.title("Energy Consumption per Session by Location Type")
plt.xlabel(""); plt.ylabel("Energy (kWh)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_energy_by_location.png")
plt.close()

# 5. Seasonal variation in demand (bar chart) --------------------------------------
season_order = ["Winter", "Spring", "Summer", "Fall"]
seasonal = df.groupby("season").size().reindex(season_order)
plt.figure(figsize=(8, 5))
seasonal.plot(kind="bar", color=["#1D3557", "#457B9D", "#A8DADC", "#E9C46A"])
plt.title("Total Charging Sessions by Season")
plt.xlabel(""); plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_seasonal_demand.png")
plt.close()

# 6. Weekday vs weekend hourly profile (line chart) --------------------------------
hourly_profile = df.groupby(["is_weekend", "start_hour"]).size().unstack(level=0)
hourly_profile.columns = ["Weekday", "Weekend"]
plt.figure(figsize=(10, 5))
hourly_profile.plot(ax=plt.gca(), marker="o", markersize=3)
plt.title("Average Hourly Charging Profile: Weekday vs Weekend")
plt.xlabel("Hour of Day"); plt.ylabel("Total Sessions (2-year period)")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_weekday_vs_weekend.png")
plt.close()

# 7. Station utilization ranking (bar chart, top 15) -------------------------------
station_util = df.groupby("station_id").size().sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 6))
station_util.iloc[::-1].plot(kind="barh", color="#2A9D8F")
plt.title("Top 15 Stations by Total Sessions (Utilization)")
plt.xlabel("Number of Sessions"); plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_top_stations_utilization.png")
plt.close()

# 8. Monthly energy consumption trend (line chart) ---------------------------------
df["year_month"] = df["start_time"].dt.to_period("M").astype(str)
monthly_energy = df.groupby("year_month")["energy_kwh"].sum()
plt.figure(figsize=(12, 4.5))
monthly_energy.plot(marker="o", color="#E76F51")
plt.title("Total Monthly Energy Delivered (kWh)")
plt.xlabel("Month"); plt.ylabel("Energy (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_monthly_energy_trend.png")
plt.close()

print("EDA complete. 8 charts saved to:", OUT_DIR)
for f in sorted(pd := __import__("os").listdir(OUT_DIR)):
    print(" -", f)

# --- Print a few key numeric insights for the report ------------------------------
print("\n=== Key EDA Insights ===")
peak_hour = df["start_hour"].value_counts().idxmax()
print(f"Peak charging hour overall: {peak_hour}:00")
busiest_day = df["day_of_week"].value_counts().idxmax()
print(f"Busiest day of week: {busiest_day}")
avg_duration = df["duration_hours"].mean()
print(f"Average session duration: {avg_duration:.2f} hours")
avg_energy = df["energy_kwh"].mean()
print(f"Average energy per session: {avg_energy:.2f} kWh")
top_station = df["station_id"].value_counts().idxmax()
print(f"Busiest station: {top_station} ({df['station_id'].value_counts().max()} sessions)")
