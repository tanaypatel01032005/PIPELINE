"""
normality_tests.py — Analysis Stage
Performs Jarque-Bera tests for normality on the stationary log return features.

Input  : data/processed/stationary_data.csv
Output : data/results/normality_results.csv
"""

import pandas as pd
from scipy.stats import jarque_bera
from pathlib import Path

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "results" / "stationary_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "results" / "normality_results.csv"

df = pd.read_csv(INPUT_CSV)

rows = []
for col in df.columns:
    series = df[col].dropna()
    stat, p = jarque_bera(series)
    rows.append({
        "Column": col,
        "JB_Statistic": round(stat, 6),
        "p_value": round(p, 6),
        "Result": "Normal" if p > 0.05 else "Non-Normal",
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_CSV, index=False)

print(f"Normality results saved to {OUTPUT_CSV}.")
print(results.to_string(index=False))
