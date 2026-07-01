import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import os
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore", category=InterpolationWarning)

df = pd.read_csv("data/results/log_return_data.csv")
numeric_cols = df.select_dtypes(include="number").columns

os.makedirs("data/results", exist_ok=True)

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
results.to_csv("data/results/stationarity_results.csv", index=False)

min_len = min(len(v) for v in stationary_data.values())
stat_df = pd.DataFrame({col: vals[:min_len] for col, vals in stationary_data.items()})
stat_df.to_csv("data/results/stationary_data.csv", index=False)

print(f"Stationarity results saved. {results['Status'].value_counts().to_dict()}")
print(results.to_string(index=False))
