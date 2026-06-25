import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT    = "data/processed/mergedFinalData_preprocessed.csv"
OUT_DIR  = "data/results"
ADF_CSV  = f"{OUT_DIR}/adf_results.csv"
STAT_CSV = f"{OUT_DIR}/stationary_data.csv"
ALPHA    = 0.05

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def adf_pvalue(series: pd.Series) -> float | None:
    """
    Run ADF test and return p-value.
    Returns None if the series is constant, too short, or raises an error.
    """
    s = series.dropna()
    if len(s) < 20 or s.nunique() < 2:  
        return None
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return None


def try_transform(series: pd.Series):
    """
    Attempt transformations in order: Log → Log+Diff / Diff.
    Returns (final_series, log_p, diff_p, transformation_label).

    Log is skipped entirely if any value is ≤ 0.
    """
    log_p = diff_p = None
    can_log = (series > 0).all()

    if can_log:
        log_s = np.log(series)
        log_p = adf_pvalue(log_s)
        if log_p is not None and log_p < ALPHA:
            return log_s, log_p, None, "Log"

        # Log didn't help → difference the log series (log-return)
        diff_s = log_s.diff().dropna()
        diff_p = adf_pvalue(diff_s)
        return diff_s, log_p, diff_p, "Log + Difference"
    else:
        # Values ≤ 0 → skip log, apply differencing directly
        diff_s = series.diff().dropna()
        diff_p = adf_pvalue(diff_s)
        return diff_s, None, diff_p, "Difference"


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = pd.read_csv(INPUT, index_col="Date", parse_dates=True)
    cols = df.select_dtypes(include="number").columns.tolist()

    records   = []   # summary rows for adf_results.csv
    stat_data = {}   # col → final stationary series

    for col in cols:
        series = df[col].dropna()
        p_orig = adf_pvalue(series)

        if p_orig is not None and p_orig < ALPHA:
            # ── Already stationary ────────────────────────────────────────────
            stat_data[col] = series
            records.append({
                "Feature":              col,
                "Original p-value":     round(p_orig, 4),
                "Log p-value":          "N/A",
                "Difference p-value":   "N/A",
                "Final p-value":        round(p_orig, 4),
                "Transformation":       "None",
                "Final Status":         "Stationary",
            })

        else:
            # ── Apply transformations ─────────────────────────────────────────
            final_s, log_p, diff_p, label = try_transform(series)

            # Determine the p-value of the last transformation applied
            last_p = diff_p if diff_p is not None else log_p
            is_stationary = last_p is not None and last_p < ALPHA

            if is_stationary:
                stat_data[col] = final_s          # save only if stationary

            records.append({
                "Feature":            col,
                "Original p-value":   round(p_orig, 4) if p_orig is not None else "N/A",
                "Log p-value":        round(log_p,  4) if log_p  is not None else "N/A",
                "Difference p-value": round(diff_p, 4) if diff_p is not None else "N/A",
                "Final p-value":      round(last_p, 4) if last_p is not None else "N/A",
                "Transformation":     label,
                "Final Status":       "Stationary" if is_stationary else "Still Non-Stationary",
            })

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(ADF_CSV, index=False)

    stationary_df = pd.DataFrame(stat_data)
    stationary_df.to_csv(STAT_CSV)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(results_df.to_string(index=False))
    ok  = results_df["Final Status"].eq("Stationary").sum()
    bad = len(results_df) - ok
    print(f"\n✅ Stationary: {ok}  |  ❌ Still Non-Stationary: {bad}")
    print(f"Saved → {ADF_CSV}\n        {STAT_CSV}")


if __name__ == "__main__":
    run()