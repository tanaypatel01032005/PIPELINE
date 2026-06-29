import pandas as pd
from scipy.stats import jarque_bera
import os

df = pd.read_csv("data/results/stationary_data.csv")

os.makedirs("data/results", exist_ok=True)

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
results.to_csv("data/results/normality_results.csv", index=False)

print(f"Normality results saved for {len(rows)} columns.")
print(results.to_string(index=False))
