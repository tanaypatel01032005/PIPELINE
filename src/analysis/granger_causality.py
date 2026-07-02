"""
granger_causality.py — Analysis Stage
Performs Pearson correlation and Granger causality tests on stationary variables to select modeling features.

Input  : data/processed/stationary_data.csv
Outputs: data/results/correlation_granger_results.csv
         data/processed/selected_features.csv
"""

import warnings
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "results" / "stationary_data.csv"
RESULTS_CSV = PROJECT_ROOT / "data" / "results" / "correlation_granger_results.csv"
SELECTED_FEATURES_CSV = PROJECT_ROOT / "data" / "results" / "granger_selected_features.csv"

df = pd.read_csv(INPUT_CSV).dropna()

target = "usd_zar_logret"
commodity_cols = [c for c in df.columns if c != target]
maxlag = 5
alpha = 0.05

corr_rows = []
granger_rows = []
selected_features = []

for col in commodity_cols:
    # Pearson correlation
    r, p_r = pearsonr(df[col], df[target])

    # Granger causality: does col cause target?
    test_data = df[[target, col]]
    gc_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

    # Take minimum p-value across all lags (F-test)
    min_p = min(
        gc_result[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)
    )
    best_lag = min(
        gc_result, key=lambda lag: gc_result[lag][0]["ssr_ftest"][1]
    )

    corr_rows.append({
        "Commodity": col,
        "Pearson_r": round(r, 6),
        "Pearson_p": round(p_r, 6),
        "Granger_min_p": round(min_p, 6),
        "Best_Lag": best_lag,
        "Granger_Significant": "Yes" if min_p < alpha else "No",
    })

    if min_p < alpha:
        selected_features.append(col)
        granger_rows.append({"Feature": col, "Granger_min_p": round(min_p, 6)})

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(RESULTS_CSV, index=False)

selected_cols = [target] + selected_features
df[selected_cols].to_csv(SELECTED_FEATURES_CSV, index=False)

print(f"Granger-significant features ({len(selected_features)}): {selected_features}")
print(corr_df.to_string(index=False))
