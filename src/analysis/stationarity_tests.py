"""
stationarity_tests.py — Analysis Stage
Performs Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity on log return features.

Input  : data/processed/log_returns.csv
Outputs: data/results/stationarity_results.csv
         data/processed/stationary_data.csv
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from pathlib import Path

warnings.filterwarnings("ignore", category=InterpolationWarning)

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "results" / "log_returns.csv"
# RESULTS_CSV = PROJECT_ROOT / "data" / "results" / "stationarity_results.csv"
# STATIONARY_DATA_CSV = PROJECT_ROOT / "data" / "results" / "stationary_data.csv"

df = pd.read_csv(INPUT_CSV)
numeric_cols = df.select_dtypes(include="number").columns

rows = []
stationary_data = {}

for col in numeric_cols:
    series = df[col].dropna()

    adf_p = adfuller(series, autolag="AIC")[1]

    try:
        kpss_p = kpss(series, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = None

    # Stationary if ADF rejects unit root (p<0.05) AND KPSS fails to reject (p>0.05)
    if kpss_p is not None:
        status = "Stationary" if adf_p < 0.05 and kpss_p > 0.05 else "Non-Stationary"
    else:
        status = "Stationary" if adf_p < 0.05 else "Non-Stationary"

    rows.append({
        "Column": col,
        "ADF_p_value": round(adf_p, 6),
        "KPSS_p_value": round(kpss_p, 6) if kpss_p is not None else None,
        "Status": status,
    })

    if status == "Stationary":
        stationary_data[col] = series.values

results = pd.DataFrame(rows)
# results.to_csv(RESULTS_CSV, index=False)

min_len = min(len(v) for v in stationary_data.values())
stat_df = pd.DataFrame({col: vals[:min_len] for col, vals in stationary_data.items()})
# stat_df.to_csv(STATIONARY_DATA_CSV, index=False)

# print(f"Stationarity results saved to {RESULTS_CSV}.")
# print(f"Stationary data saved to {STATIONARY_DATA_CSV}.")
print(results.to_string(index=False))
