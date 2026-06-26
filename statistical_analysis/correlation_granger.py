"""
correlation_granger.py — Stage 3: Pearson Correlation & Granger Causality Analysis.

Pearson correlation: descriptive/statistical interpretation only (NOT used for selection).
Granger causality:  sole feature selection criterion — selected if min p-value < 0.05.
Inputs:  data/results/stationary_data.csv
Outputs: data/results/correlation_granger_results.csv
         data/results/granger_selected_features.csv  (top-8 by Granger p-value)
"""

import os, sys, warnings
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT      = "data/results/stationary_data.csv"
OUT_DIR    = "data/results"
ALL_CSV    = f"{OUT_DIR}/correlation_granger_results.csv"
SEL_CSV    = f"{OUT_DIR}/granger_selected_features.csv"
MAX_LAG    = 5
ALPHA      = 0.05
TOP_N      = 8
TARGET_COL = "usd_zar_logret"

os.makedirs(OUT_DIR, exist_ok=True)


def pearson_corr(x, y):
    try:
        data = pd.concat([x, y], axis=1).dropna()
        if len(data) < 10:
            return None, None
        r, p = pearsonr(data.iloc[:, 0], data.iloc[:, 1])
        return round(float(r), 4), round(float(p), 4)
    except Exception:
        return None, None


def granger_test(df, cause, target):
    # Convention: data = [target, cause] (target column first)
    data = df[[target, cause]].dropna()
    if len(data) < MAX_LAG + 20:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
        lag_pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, MAX_LAG + 1)}
        best_lag  = min(lag_pvals, key=lag_pvals.get)
        return best_lag, round(lag_pvals[best_lag], 4)
    except Exception as e:
        print(f"    [WARN] Granger test failed for '{cause}': {e}")
        return None, None


def run():
    print("=" * 70)
    print("CORRELATION & GRANGER CAUSALITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(INPUT, index_col="Date", parse_dates=True)
    print(f"\n[1] Loaded '{INPUT}'\n    Rows: {len(df)}  Columns: {len(df.columns)} → {list(df.columns)}")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in {INPUT}.\n"
            f"Available columns: {list(df.columns)}\n"
            "Ensure adf_test.py has been run and produces '*_logret' columns."
        )
    print(f"\n[2] Target column confirmed : '{TARGET_COL}'")

    commodity_cols = [c for c in df.columns if c != TARGET_COL]
    print(f"\n[3] Commodity log-return columns ({len(commodity_cols)}): {commodity_cols}")
    print(f"\n[4] Running Pearson + Granger tests (maxlag={MAX_LAG}) …\n")

    records = []
    for col in commodity_cols:
        base_name        = col.replace("_logret_diff1", "").replace("_logret", "")
        r, r_pval        = pearson_corr(df[col], df[TARGET_COL])
        best_lag, min_pv = granger_test(df, cause=col, target=TARGET_COL)
        selected         = "Yes" if (min_pv is not None and min_pv < ALPHA) else "No"

        records.append({
            "Commodity":        base_name,
            "LogRet_Column":    col,
            "Pearson_r":        r        if r        is not None else "N/A",
            "Pearson_p":        r_pval   if r_pval   is not None else "N/A",
            "Abs_Pearson_r":    round(abs(r), 4) if r is not None else "N/A",
            "Best_Granger_Lag": best_lag if best_lag is not None else "N/A",
            "Min_Granger_p":    min_pv   if min_pv   is not None else "N/A",
            "Granger_Selected": selected,
        })
        status = "✅ Selected" if selected == "Yes" else "❌ Rejected"
        print(f"  {col:<35s} | Pearson r={str(r):<7s} | Granger p={str(min_pv):<7s} | Lag={str(best_lag):<2s} | {status}")

    results_df = pd.DataFrame(records).sort_values(
        by=["Granger_Selected", "Min_Granger_p"],
        ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    selected_df = results_df[results_df["Granger_Selected"] == "Yes"].head(TOP_N).reset_index(drop=True)

    results_df.to_csv(ALL_CSV, index=False)
    selected_df.to_csv(SEL_CSV, index=False)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 70)
    rejected_names = results_df[results_df["Granger_Selected"] == "No"]["Commodity"].tolist()
    print(f"\n  Number of selected features : {len(selected_df)}  (top {TOP_N} by Granger p-value)")
    print(f"  Number of rejected features : {len(results_df) - len(selected_df)}")
    print(f"\n  Selected features (ranked by Granger p-value):")
    for i, row in selected_df.iterrows():
        print(f"    {i+1}. {row['Commodity']:<30s} | Granger p = {str(row['Min_Granger_p']):<7s} | Lag = {row['Best_Granger_Lag']}")
    print(f"\n  Rejected features : {rejected_names}")

    if not selected_df.empty:
        best = selected_df.iloc[0]
        print(f"\n  Best feature      : {best['Commodity']}")
        print(f"  Best Granger lag  : {best['Best_Granger_Lag']}")
        print(f"  Best Granger p    : {best['Min_Granger_p']}")

    print(f"\n  Criterion : Min Granger p-value < {ALPHA}  (maxlag = {MAX_LAG})")
    print(f"\nSaved → {ALL_CSV}")
    print(f"Saved → {SEL_CSV}")

    return results_df, selected_df


if __name__ == "__main__":
    run()