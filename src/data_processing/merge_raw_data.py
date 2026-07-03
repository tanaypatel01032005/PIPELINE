"""
merge_raw_data.py
-----------------
Creates merged_raw_data.csv by:
  1. Building a weekday date spine: 01/01/2000 - 31/12/2025 (no Sat/Sun)
  2. Adding usd_zar column from _usd_zar.csv
  3. For every numbered CSV (1_*.csv to 31_*.csv):
       - If only ONE data column: use it directly
       - If multiple columns:     always pick 'Price'
     Renames the column to the commodity name derived from the filename.
  4. Left-joins each commodity onto the date spine, so every weekday row is
     preserved even if a commodity has no data for that day (NaN).
"""

import sys
import re
import pandas as pd
from pathlib import Path

# Force UTF-8 output so print works on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV   = PROCESSED_DIR / "merged_raw_data.csv"

# -- 1. Build weekday date spine ---------------------------------------------
print("Building date spine (Mon-Fri, 2000-01-01 to 2025-12-31) ...")
date_spine = pd.DataFrame({
    "Date": pd.bdate_range(start="2000-01-01", end="2025-12-31", freq="B")
})
date_spine["Date"] = pd.to_datetime(date_spine["Date"]).dt.normalize()
print("  -> " + str(len(date_spine)) + " business days")

merged = date_spine.copy()

# -- Helper: parse date column -----------------------------------------------
def parse_dates(series):
    """Try multiple date formats; return a datetime Series."""
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(series)

# -- Helper: clean numeric column (strip commas, %) --------------------------
def clean_numeric(series):
    if series.dtype == object:
        series = series.str.replace(",", "", regex=False)
        series = series.str.replace("%", "", regex=False)
        series = series.str.strip()
    return pd.to_numeric(series, errors="coerce")

# -- 2. Load USD/ZAR ---------------------------------------------------------
print("\nLoading USD/ZAR ...")
zar_path = RAW_DIR / "_usd_zar.csv"
zar_df   = pd.read_csv(zar_path)

date_col = next((c for c in zar_df.columns if "date" in c.lower()), zar_df.columns[0])
val_col  = next((c for c in zar_df.columns if c != date_col), zar_df.columns[1])

zar_df = zar_df[[date_col, val_col]].rename(columns={date_col: "Date", val_col: "usd_zar"})
zar_df["Date"]    = parse_dates(zar_df["Date"]).dt.normalize()
zar_df["usd_zar"] = clean_numeric(zar_df["usd_zar"])

merged = merged.merge(zar_df, on="Date", how="left")
print("  -> USD/ZAR merged  (" + str(zar_df["usd_zar"].notna().sum()) + " non-null values)")


# -- 3. Load each numbered commodity CSV ------------------------------------
commodity_pattern = re.compile(r"^(\d+)_(.+)\.csv$", re.IGNORECASE)

csv_files = sorted(
    [f for f in RAW_DIR.iterdir() if commodity_pattern.match(f.name)],
    key=lambda f: int(commodity_pattern.match(f.name).group(1))
)

print("\nFound " + str(len(csv_files)) + " commodity files.\n")

for csv_path in csv_files:
    match    = commodity_pattern.match(csv_path.name)
    file_num = match.group(1)
    raw_name = match.group(2)
    col_name = raw_name.replace(" ", "_").strip("_")

    print("  [" + file_num.rjust(2) + "] " + csv_path.name + "  ->  column '" + col_name + "'")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print("       WARNING: Could not read file: " + str(e))
        continue

    # Identify the Date column
    date_col  = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    data_cols = [c for c in df.columns if c != date_col]

    if len(data_cols) == 0:
        print("       WARNING: No data columns found, skipping.")
        continue
    elif len(data_cols) == 1:
        chosen = data_cols[0]
        print("       single column -> '" + chosen + "'")
    else:
        price_matches = [c for c in data_cols if c.strip().lower() == "price"]
        chosen = price_matches[0] if price_matches else data_cols[0]
        print("       multiple columns, picking -> '" + chosen + "'")

    df = df[[date_col, chosen]].copy()
    df.columns = ["Date", col_name]
    df["Date"]   = parse_dates(df["Date"]).dt.normalize()
    df[col_name] = clean_numeric(df[col_name])

    # Drop duplicate dates (keep first occurrence)
    df = df.drop_duplicates(subset="Date", keep="first")

    merged = merged.merge(df, on="Date", how="left")

# -- 4. Save -----------------------------------------------------------------
merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
merged.to_csv(OUTPUT_CSV, index=False)

print("\nDone!  Saved -> " + str(OUTPUT_CSV))
print("Rows    : " + str(len(merged)))
print("Columns : " + str(len(merged.columns)))
print("Names   : " + str(list(merged.columns)))
