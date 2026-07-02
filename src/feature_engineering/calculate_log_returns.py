"""
calculate_log_returns.py - Feature Engineering Stage
Computes log returns for all numeric variables.

Input  : data/processed/preprocessed_data.csv
Output : data/results/log_return_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "results" / "log_return_data.csv"

df = pd.read_csv(INPUT_CSV)
numeric_cols = df.select_dtypes(include="number").columns

log_ret_df = pd.DataFrame()
log_ret_df["Date"] = df["Date"]

for col in numeric_cols:
    log_ret_df[col + "_logret"] = np.log(df[col]).diff()

# Remove the first row containing NaN from differencing
log_ret_df = log_ret_df.iloc[1:].reset_index(drop=True)

log_ret_df.to_csv(OUTPUT_CSV, index=False)

print(f"Log returns saved: {log_ret_df.shape[0]} rows, {log_ret_df.shape[1] - 1} columns.")
print(log_ret_df.head())
