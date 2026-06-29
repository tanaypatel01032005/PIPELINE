import pandas as pd
from scipy.stats import skew, kurtosis
import os

df = pd.read_csv("data/results/log_return_data.csv")
numeric_cols = df.select_dtypes(include="number").columns

os.makedirs("data/results", exist_ok=True)

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
results.to_csv("data/results/descriptive_statistics.csv", index=False)

print(f"Descriptive statistics saved for {len(rows)} columns.")
print(results.to_string(index=False))
