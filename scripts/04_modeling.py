"""
04_modeling.py
---------------
Predictive modeling: forecast hourly EV charging demand (number of
sessions) per station.

Approach:
  - Time-based train/test split (last 60 days = test set) -- never random
    split for time series, or the model "sees the future".
  - Feature-based models (Linear Regression, Random Forest, XGBoost) using
    engineered calendar features + lag features.
  - A classical time-series baseline (SARIMA) on the network-wide total
    demand for comparison.
  - Evaluation via RMSE, MAE, and MAPE.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

HOURLY_PATH = "E:\\VS CODE\\ev_project\\data\\hourly_station_demand.csv"
MODEL_DIR = "E:\\VS CODE\\ev_project\\outputs\\models"
RESULTS_PATH = "E:\\VS CODE\\ev_project\\outputs\\models\\model_comparison.csv"

df = pd.read_csv(HOURLY_PATH, parse_dates=["date_hour"])
df = df.sort_values(["station_id", "date_hour"]).reset_index(drop=True)

# --- Feature engineering: lag features per station --------------------------------
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["is_weekend"] = df["is_weekend"].astype(int)

df["lag_1h"] = df.groupby("station_id")["n_sessions"].shift(1)
df["lag_24h"] = df.groupby("station_id")["n_sessions"].shift(24)     # same hour yesterday
df["lag_168h"] = df.groupby("station_id")["n_sessions"].shift(168)   # same hour last week
df["rolling_avg_24h"] = (
    df.groupby("station_id")["n_sessions"].shift(1).rolling(24).mean().reset_index(0, drop=True)
)

df = pd.get_dummies(df, columns=["location_type"], prefix="loc")
df = df.dropna().reset_index(drop=True)

FEATURE_COLS = [c for c in df.columns if c.startswith("loc_")] + [
    "hour", "day_of_week", "is_weekend", "month",
    "dow_sin", "dow_cos", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h", "rolling_avg_24h",
]
TARGET_COL = "n_sessions"

# --- Time-based split: last 60 days as test set ------------------------------------
cutoff = df["date_hour"].max() - pd.Timedelta(days=60)
train = df[df["date_hour"] <= cutoff]
test = df[df["date_hour"] > cutoff]
print(f"Train: {len(train):,} rows ({train['date_hour'].min().date()} to {train['date_hour'].max().date()})")
print(f"Test:  {len(test):,} rows ({test['date_hour'].min().date()} to {test['date_hour'].max().date()})")

X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]


def mape(y_true, y_pred):
    mask = y_true > 0.5  # avoid divide-by-near-zero blowing up the metric
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


results = []
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
}

fitted_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    m = mape(y_test.values, preds)
    results.append({"model": name, "RMSE": rmse, "MAE": mae, "MAPE_%": m})
    fitted_models[name] = model
    print(f"{name:20s} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {m:.2f}%")

# --- Naive baseline for comparison: "same hour last week" -------------------------
naive_preds = test["lag_168h"].values
rmse_naive = np.sqrt(mean_squared_error(y_test, naive_preds))
mae_naive = mean_absolute_error(y_test, naive_preds)
mape_naive = mape(y_test.values, naive_preds)
results.append({"model": "Naive (same hour last week)", "RMSE": rmse_naive, "MAE": mae_naive, "MAPE_%": mape_naive})
print(f"{'Naive baseline':20s} | RMSE: {rmse_naive:.3f} | MAE: {mae_naive:.3f} | MAPE: {mape_naive:.2f}%")

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(RESULTS_PATH, index=False)
print(f"\nModel comparison saved -> {RESULTS_PATH}")
print(results_df.to_string(index=False))

# --- Save the best model + feature importance --------------------------------------
best_name = results_df.iloc[0]["model"]
if best_name in fitted_models:
    best_model = fitted_models[best_name]
    joblib.dump(best_model, f"{MODEL_DIR}/best_model_{best_name.replace(' ', '_')}.pkl")
    print(f"\nBest model '{best_name}' saved to {MODEL_DIR}/")

    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        importance = importance.sort_values(ascending=False)
        importance.to_csv(f"{MODEL_DIR}/feature_importance.csv")
        print("\nTop 8 most important features:")
        print(importance.head(8).to_string())

# --- Save test-set predictions (network-wide, aggregated by hour) for the dashboard
test_copy = test.copy()
if best_name in fitted_models:
    test_copy["predicted"] = np.clip(fitted_models[best_name].predict(X_test), 0, None)
else:
    test_copy["predicted"] = naive_preds

agg_pred = (
    test_copy.groupby("date_hour")[["n_sessions", "predicted"]]
    .sum()
    .reset_index()
    .rename(columns={"n_sessions": "actual"})
)
agg_pred.to_csv(f"{MODEL_DIR}/network_wide_predictions.csv", index=False)
print(f"\nSaved network-wide actual-vs-predicted series -> {MODEL_DIR}/network_wide_predictions.csv")
