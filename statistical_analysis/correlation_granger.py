import os
import warnings
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

# ── Config ────────────────────────────────────────────────────────────────────
INPUT   = "data/results/stationary_data.csv"
OUT_DIR = "data/results"
ALL_CSV = f"{OUT_DIR}/correlation_granger_results.csv"
SEL_CSV = f"{OUT_DIR}/granger_selected_features.csv"

MAX_LAG = 5
ALPHA   = 0.05

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def pearson_corr(x: pd.Series, y: pd.Series) -> float | None:
    """
    Pearson correlation for statistical analysis only — not used for selection.
    Returns None if insufficient or invalid data.
    """
    try:
        data = pd.concat([x, y], axis=1).dropna()
        if len(data) < 10:
            return None
        r, _ = pearsonr(data.iloc[:, 0], data.iloc[:, 1])
        return round(float(r), 4)
    except Exception:
        return None


def granger_test(df: pd.DataFrame, cause: str, target: str):
    """
    Granger causality test: does `cause` Granger-cause `target`?
    Tests lags 1..MAX_LAG using the ssr_ftest statistic.
    Returns (best_lag, min_pvalue) or (None, None) on failure.
    """
    data = df[[target, cause]].dropna()
    if len(data) < MAX_LAG + 10:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
        lag_pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, MAX_LAG + 1)}
        best_lag  = min(lag_pvals, key=lag_pvals.get)
        return best_lag, round(lag_pvals[best_lag], 4)
    except Exception:
        return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = pd.read_csv(INPUT, index_col="Date", parse_dates=True)

    if "usd_zar" not in df.columns:
        raise ValueError("Target 'usd_zar' not found in stationary_data.csv")

    commodities = [c for c in df.columns if c != "usd_zar"]
    records = []

    for col in commodities:
        # Pearson — descriptive/statistical analysis only
        r = pearson_corr(df[col], df["usd_zar"])

        # Granger — sole basis for feature selection
        best_lag, min_pval = granger_test(df, cause=col, target="usd_zar")

        granger_selected = (
            "Yes" if (min_pval is not None and min_pval < ALPHA) else "No"
        )

        records.append({
            "Commodity":                col,
            "Pearson Correlation":      r         if r        is not None else "N/A",
            "Absolute Pearson Corr":    round(abs(r), 4) if r is not None else "N/A",
            "Best Granger Lag":         best_lag  if best_lag is not None else "N/A",
            "Min Granger p-value":      min_pval  if min_pval is not None else "N/A",
            "Granger Selected":         granger_selected,
        })

    # Sort: selected first, then by Granger p-value ascending
    results_df = pd.DataFrame(records).sort_values(
        by=["Granger Selected", "Min Granger p-value"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    selected_df = results_df[results_df["Granger Selected"] == "Yes"].reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df.to_csv(ALL_CSV, index=False)
    selected_df.to_csv(SEL_CSV, index=False)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(results_df.to_string(index=False))
    print(f"\n✅ Granger-selected features ({len(selected_df)}): {selected_df['Commodity'].tolist()}")
    print(f"   Criterion: Min Granger p-value < {ALPHA}  (maxlag={MAX_LAG})")
    print(f"\nSaved → {ALL_CSV}\n        {SEL_CSV}")

    return results_df, selected_df


if __name__ == "__main__":
    run()