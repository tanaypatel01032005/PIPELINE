import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/processed/mergedFinalData_preprocessed.csv")
numeric_cols = df.select_dtypes(include="number").columns

os.makedirs("data/results", exist_ok=True)

log_ret_df = pd.DataFrame()
log_ret_df["Date"] = df["Date"]

for col in numeric_cols:
    log_ret_df[col + "_logret"] = np.log(df[col]).diff()

log_ret_df = log_ret_df.iloc[1:].reset_index(drop=True)

log_ret_df.to_csv("data/results/log_return_data.csv", index=False)

print(f"Log returns saved: {log_ret_df.shape[0]} rows, {log_ret_df.shape[1] - 1} columns.")
print(log_ret_df.head())
