"""
preprocess.py — Stage 1: Load raw merged CSV, clean, validate, and save clean daily prices.

Pipeline: Load → Parse/sort dates → Dedup → Missing date report → Numeric coerce →
          Missing value summary → ffill/bfill → Non-negative check → Zero check → Save
Inputs:  data/processed/mergedFinalData.csv
Outputs: data/processed/mergedFinalData_preprocessed.csv
         data/results/preprocessing_summary.txt
"""

import os, sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_CSV   = "data/processed/mergedFinalData.csv"
OUTPUT_CSV  = "data/processed/mergedFinalData_preprocessed.csv"
SUMMARY_TXT = "data/results/preprocessing_summary.txt"

os.makedirs("data/results", exist_ok=True)

COMMODITIES = [
    "Brent_Oil", "Natural_Gas", "US_Soybean_Oil", "US_Soybean_Meal", "Gold",
    "US_Copper", "Silver", "US_Sugar", "US_Soybeans", "Hard_Red_Winter_Wheat",
    "usd_zar", "Nickel", "Orange_Juice", "platinum", "Lead", "US_beef",
    "Palladium", "US_Cotton", "Tin", "Iron_ore_62Fe",
]


def report_duplicate_dates(df, log):
    exact_dupes = df.duplicated().sum()
    log.append(f"  Exact duplicate rows removed : {exact_dupes}")
    df = df.drop_duplicates().reset_index(drop=True)

    date_dupes = df.duplicated(subset="Date", keep=False)
    n = date_dupes.sum()
    if n > 0:
        log.append(f"  [WARN] Duplicate DATE rows (different values, kept first): {n}")
        dates = df.loc[date_dupes, "Date"].dt.strftime("%Y-%m-%d").unique()
        log.append(f"     Affected dates: {list(dates[:10])}{'...' if len(dates) > 10 else ''}")
    else:
        log.append("  No duplicate date rows detected.")

    return df.drop_duplicates(subset="Date", keep="first").reset_index(drop=True)


def report_missing_dates(df, log):
    # Informational only — no rows inserted
    full_bdays = pd.bdate_range(start=df["Date"].min(), end=df["Date"].max(), freq="B")
    missing = full_bdays.difference(df["Date"])
    log.append(f"  Missing weekday dates (report only, not filled): {len(missing)}")
    if 0 < len(missing) <= 10:
        log.append(f"     Dates: {list(missing.strftime('%Y-%m-%d'))}")
    elif len(missing) > 10:
        log.append(f"     First 10: {list(missing[:10].strftime('%Y-%m-%d'))} ...")


def report_numeric_conversion(df_raw, df_coerced, log):
    for col in COMMODITIES:
        if col not in df_raw.columns:
            continue
        n = int((df_raw[col].notna() & df_coerced[col].isna()).sum())
        if n > 0:
            log.append(f"    [WARN] '{col}': {n} non-numeric value(s) coerced to NaN")


def missing_value_summary(df, stage, log):
    total = len(df)
    log.append(f"\n  Missing values — {stage}:")
    any_missing = False
    for col in COMMODITIES:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n > 0:
            log.append(f"    {col:<30s}: {n:>5d}  ({100*n/total:5.2f}%)")
            any_missing = True
    if not any_missing:
        log.append("    All columns complete -- no missing values.")


def check_non_negative(df, log):
    log.append("\n  Non-negative price check:")
    violations = False
    for col in COMMODITIES:
        if col not in df.columns:
            continue
        n = int((df[col] < 0).sum())
        if n > 0:
            log.append(f"    [WARN] '{col}': {n} negative value(s) detected")
            violations = True
    if not violations:
        log.append("    All prices are non-negative. [OK]")


def report_zero_values(df, log):
    log.append("\n  Zero-value column report:")
    zero_cols = []
    for col in COMMODITIES:
        if col not in df.columns:
            continue
        n = int((df[col] == 0).sum())
        if n > 0:
            zero_cols.append(col)
            log.append(f"    [WARN] '{col}': {n} zero value(s)")
    if not zero_cols:
        log.append("    No columns contain exact zero values. [OK]")


def preprocess(path):
    log = []
    log.append("=" * 70)
    log.append("PREPROCESSING SUMMARY")
    log.append("=" * 70)

    all_cols   = ["Date"] + COMMODITIES
    df_raw     = pd.read_csv(path, usecols=lambda c: c in all_cols, low_memory=False)
    avail_comm = [c for c in COMMODITIES if c in df_raw.columns]

    log.append(f"\n[1] File loaded : {path}")
    log.append(f"    Rows        : {len(df_raw)}")
    log.append(f"    Columns     : {len(df_raw.columns)}  → {list(df_raw.columns)}")

    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)
    log.append(f"\n[2] Date range  : {df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}")

    log.append("\n[3] Duplicate checks:")
    df_raw = report_duplicate_dates(df_raw, log)

    log.append("\n[4] Calendar completeness:")
    report_missing_dates(df_raw, log)

    df_before_coerce = df_raw[avail_comm].copy()
    df_raw[avail_comm] = df_raw[avail_comm].apply(pd.to_numeric, errors="coerce")

    log.append("\n[5] Numeric conversion report:")
    report_numeric_conversion(df_before_coerce, df_raw[avail_comm], log)
    if not any(int(((df_before_coerce[c].notna()) & (df_raw[c].isna())).sum()) > 0
               for c in avail_comm if c in df_before_coerce.columns):
        log.append("    All values converted cleanly -- no coercion issues. [OK]")

    missing_value_summary(df_raw, "Before filling", log)

    # ffill carries last known price forward; bfill handles leading NaNs
    df_raw[avail_comm] = df_raw[avail_comm].ffill().bfill()

    missing_value_summary(df_raw, "After filling", log)
    check_non_negative(df_raw, log)
    report_zero_values(df_raw, log)

    log.append("\n[11] Descriptive statistics (after cleaning):")
    log.append(df_raw[avail_comm].describe().round(4).to_string())
    log.append(f"\n[12] Final dataset shape : {df_raw.shape}")
    log.append("=" * 70)

    return df_raw, log


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df, summary_lines = preprocess(INPUT_CSV)
    print("\n".join(summary_lines))

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n[SAVED] Summary -> {SUMMARY_TXT}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Clean prices saved -> {OUTPUT_CSV}")
    print(f"     Shape: {df.shape}")