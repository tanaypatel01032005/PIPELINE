"""
adf_test.py — Stage 2: Stationarity Testing via ADF on log returns.

Pipeline: Raw Prices → Log Returns → Drop first NaN → ADF test →
          If non-stationary: first-difference log return → re-test → Save
Inputs:  data/processed/mergedFinalData_preprocessed.csv
Outputs: data/results/adf_results.csv
         data/results/stationary_data.csv
Column naming: <commodity>_logret | <commodity>_logret_diff1
"""

import os, sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT    = "data/processed/mergedFinalData_preprocessed.csv"
OUT_DIR  = "data/results"
ADF_CSV  = f"{OUT_DIR}/adf_results.csv"
STAT_CSV = f"{OUT_DIR}/stationary_data.csv"
ALPHA    = 0.05

os.makedirs(OUT_DIR, exist_ok=True)


def run_adf(series, label):
    s = series.dropna()
    if len(s) < 20:
        print(f"  [SKIP] '{label}': too few observations ({len(s)}) for ADF.")
        return _null_adf()
    if s.nunique() < 2:
        print(f"  [SKIP] '{label}': constant series — ADF not meaningful.")
        return _null_adf()
    try:
        adf_stat, p_val, _, _, crit_vals, _ = adfuller(s, autolag="AIC")
        return {
            "adf_stat":   round(float(adf_stat), 4),
            "p_value":    round(float(p_val),    4),
            "crit_1pct":  round(crit_vals["1%"],  4),
            "crit_5pct":  round(crit_vals["5%"],  4),
            "crit_10pct": round(crit_vals["10%"], 4),
            "stationary": float(p_val) < ALPHA,
        }
    except Exception as e:
        print(f"  [ERROR] ADF failed for '{label}': {e}")
        return _null_adf()


def _null_adf():
    return dict(adf_stat=None, p_value=None, crit_1pct=None,
                crit_5pct=None, crit_10pct=None, stationary=None)


def compute_log_return(series):
    s = series.copy()
    n = int((s <= 0).sum())
    if n > 0:
        print(f"    ⚠  '{series.name}': {n} non-positive price(s) replaced with NaN before log.")
        s[s <= 0] = np.nan
    return np.log(s).diff()


def run():
    print("=" * 70)
    print("ADF STATIONARITY TEST  (log-return based)")
    print("=" * 70)

    df = pd.read_csv(INPUT, index_col="Date", parse_dates=True)
    price_cols = df.select_dtypes(include="number").columns.tolist()
    print(f"\n[1] Loaded '{INPUT}'\n    Rows: {len(df)}  Price cols: {len(price_cols)} → {price_cols}")

    print("\n[2] Computing log returns …")
    logret_df = pd.DataFrame(
        {f"{col}_logret": compute_log_return(df[col]) for col in price_cols},
        index=df.index
    )

    logret_df = logret_df.iloc[1:].copy()
    print(f"\n[3] Log-return DataFrame: {logret_df.shape}  (first NaN row dropped)")

    print("\n[4] Running ADF tests …\n")
    records, stat_data = [], {}

    for logret_col in logret_df.columns:
        base_name = logret_col.replace("_logret", "")
        series    = logret_df[logret_col]
        print(f"  Testing : {logret_col}")

        adf1 = run_adf(series, logret_col)

        if adf1["stationary"] is True:
            stat_data[logret_col] = series
            final_transform = "Log Return"
            final_col_name  = logret_col
            adf2, used_diff = {}, False
        else:
            diff1_series = series.diff().dropna()
            diff1_col    = f"{base_name}_logret_diff1"
            adf2         = run_adf(diff1_series, diff1_col)
            if adf2.get("stationary") is True:
                stat_data[diff1_col] = diff1_series
                final_transform = "Log Return + First Difference"
                final_col_name  = diff1_col
            else:
                final_transform = "Log Return + First Difference (still non-stationary)"
                final_col_name  = "–"
            used_diff = True

        records.append({
            "Feature":                   base_name,
            "Original Transformation":   "Log Return",
            "ADF Statistic (logret)":    adf1.get("adf_stat",   "N/A"),
            "p-value (logret)":          adf1.get("p_value",    "N/A"),
            "Crit 1% (logret)":          adf1.get("crit_1pct",  "N/A"),
            "Crit 5% (logret)":          adf1.get("crit_5pct",  "N/A"),
            "Crit 10% (logret)":         adf1.get("crit_10pct", "N/A"),
            "Stationary (logret)":       "Yes" if adf1.get("stationary") else "No",
            "ADF Statistic (diff1)":     adf2.get("adf_stat",   "N/A") if used_diff else "N/A",
            "p-value (diff1)":           adf2.get("p_value",    "N/A") if used_diff else "N/A",
            "Stationary (diff1)":        ("Yes" if adf2.get("stationary") else "No") if used_diff else "N/A",
            "Final Transformation Used": final_transform,
            "Final Column Name":         final_col_name,
            "Overall Stationary":        "Yes" if final_col_name != "–" else "No",
        })

    results_df    = pd.DataFrame(records)
    stationary_df = pd.DataFrame(stat_data)
    results_df.to_csv(ADF_CSV, index=False)
    stationary_df.to_csv(STAT_CSV)

    print("\n" + "=" * 70)
    print("ADF RESULTS SUMMARY")
    print("=" * 70)
    print(results_df[[
        "Feature", "p-value (logret)", "Stationary (logret)",
        "p-value (diff1)", "Stationary (diff1)", "Final Transformation Used"
    ]].to_string(index=False))

    n_stat = results_df["Overall Stationary"].eq("Yes").sum()
    print(f"\n✅ Stationary series saved : {n_stat}")
    print(f"❌ Could not achieve stationarity : {len(results_df) - n_stat}")
    print(f"\n📁 Stationary columns in '{STAT_CSV}':")
    for col in stationary_df.columns:
        print(f"   • {col}")
    print(f"\nSaved → {ADF_CSV}")
    print(f"Saved → {STAT_CSV}")


if __name__ == "__main__":
    run()