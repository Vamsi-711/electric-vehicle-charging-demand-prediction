# Electric Vehicle Charging Demand Prediction

A working data science pipeline covering data collection, cleaning, exploratory
analysis, and predictive modeling for EV charging demand — plus an interactive
dashboard.

## Project structure

```
ev_project/
├── scripts/
│   ├── 01_generate_data.py    # Synthetic dataset generator (see note below)
│   ├── 02_preprocess.py       # Cleaning, outlier handling, feature engineering
│   ├── 03_eda.py              # Exploratory analysis + 9 charts
│   ├── 04_modeling.py         # Forecasting models + evaluation
│   └── 05_dashboard_data.py   # Aggregates everything into dashboard_data.json
├── data/
│   ├── raw_ev_charging_sessions.csv        # ~112K sessions, with injected issues
│   ├── cleaned_ev_charging_sessions.csv    # Post-cleaning, feature-engineered
│   └── hourly_station_demand.csv           # Hourly per-station demand table
├── outputs/
│   ├── eda_charts/             # 9 PNG charts from the EDA phase
│   ├── models/                 # Model comparison, feature importance, saved model
│   └── dashboard_data/         # dashboard_data.json (powers the dashboard)
└── README.md
```

## Run order

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost joblib

python scripts/01_generate_data.py
python scripts/02_preprocess.py
python scripts/03_eda.py
python scripts/04_modeling.py
python scripts/05_dashboard_data.py
```

## About the dataset

**No real dataset was supplied with the project brief**, so `01_generate_data.py`
generates a synthetic but statistically realistic dataset: 25 stations across
5 location types (Downtown, Mall, Office Park, Residential, Highway Rest
Stop), ~112K sessions over 2 years (2023–2024), with genuine daily/weekly/
seasonal patterns, plus deliberately injected missing values, outliers, and
duplicates so the preprocessing step has real problems to solve.

**To use a real dataset instead** (e.g. Kaggle's "EV Charging Station Usage",
Caltech's ACN-Data, or Palo Alto/Boulder open charging data): replace
`01_generate_data.py` with a loader that outputs a CSV with these columns,
and every downstream script keeps working unchanged:

| column          | type     | description                          |
|-----------------|----------|---------------------------------------|
| session_id      | string   | unique session identifier             |
| station_id      | string   | charging station identifier           |
| location_type   | string   | station category (used in EDA/model)  |
| user_id         | string   | driver/account identifier             |
| start_time      | datetime | session start                         |
| end_time        | datetime | session end                           |
| duration_hours  | float    | session length in hours               |
| energy_kwh      | float    | energy delivered                      |
| charger_kw      | float    | charger power rating                  |

## Key results (on the synthetic data)

- **Peak demand hour:** 17:00, with a secondary residential peak in the evening
- **Busiest day:** Sunday (mall/residential locations skew weekend-heavy)
- **Best forecasting model:** XGBoost (RMSE 0.554 vs 0.773 for the naive
  "same hour last week" baseline — a ~28% improvement)
- **Top demand driver:** location type (Residential) and hour-of-day, per
  feature importance

## Dashboard

`ev_charging_dashboard.html` (published separately as a Claude artifact) is a
self-contained interactive dashboard built from `dashboard_data.json` —
KPIs, demand heatmap, station leaderboard, energy distributions, capacity
watch-list, and the forecast-vs-actual comparison.

## Suggested next steps

1. Swap in a real dataset following the schema above.
2. Try LSTM/Prophet for the time-series forecast and compare against XGBoost.
3. Add weather data as a feature (temperature strongly affects EV range/charging behavior).
4. Extend the dashboard with a per-station drill-down view.
