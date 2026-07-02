"""
descriptive_stats.py — Analysis Stage
Computes descriptive statistics (mean, median, std, skewness, kurtosis) for log return features.

Input  : data/processed/log_returns.csv
Output : data/results/descriptive_stats.csv
"""

import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "results" / "log_return_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "results" / "descriptive_statistics.csv"

df = pd.read_csv(INPUT_CSV)
numeric_cols = df.select_dtypes(include="number").columns

rows = []
for col in numeric_cols:
    s = df[col].dropna()
    rows.append({
        "Column": col,
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(),
        "Min": s.min(),
        "Max": s.max(),
        "Skewness": skew(s),
        "Kurtosis": kurtosis(s),
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_CSV, index=False)

print(f"Descriptive statistics saved for {len(rows)} columns.")
print(results.to_string(index=False))
