"""
preprocess.py — Preprocessing Stage
Cleans and prepares the raw merged commodity-currency dataset.

Input  : data/processed/mergedFinalData.csv
Output : data/processed/mergedFinalData_preprocessed.csv
"""

import pandas as pd

INPUT_CSV  = "data/processed/mergedFinalData.csv"
OUTPUT_CSV = "data/processed/mergedFinalData_preprocessed.csv"

# Some columns in the CSV use spaces; we rename them to underscores.
RENAME_MAP = {
    "Heating Oil"   : "Heating_Oil",
    "Lean Hogs"     : "Lean_Hogs",
    "RBOB Gasoline" : "RBOB_Gasoline",
    "WTI Crude Oil" : "WTI_Crude_Oil",
}

# Final required columns (after renaming).
COLUMNS = [
    "Date",
    "usd_zar",
    "Brent_Oil", "Natural_Gas", "US_Soybean_Oil", "US_Soybean_Meal", "Gold",
    "US_Copper", "Silver", "US_Sugar", "US_Soybeans", "Hard_Red_Winter_Wheat",
    "Nickel", "Orange_Juice", "platinum", "Lead", "US_beef", "Palladium",
    "US_Cotton", "Tin",
    "Cocoa", "Coffee", "Corn", "Heating_Oil", "Lean_Hogs", "Oats",
    "RBOB_Gasoline", "Rhodium", "Rice", "WTI_Crude_Oil",
]

# ── Step 1: Load CSV ──────────────────────────────────────────────────────────
# Build the set of raw column names to read (accounting for space variants).
raw_names = set(COLUMNS) | set(RENAME_MAP.keys())
df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in raw_names, low_memory=False)

# Rename space-separated columns to underscore versions.
df = df.rename(columns=RENAME_MAP)

# Keep only the required columns that actually exist in the file.
existing_cols = [c for c in COLUMNS if c in df.columns]
df = df[existing_cols]

print(f"[1] Loaded  : {INPUT_CSV}")
print(f"    Shape   : {df.shape}")
print(f"    Columns : {list(df.columns)}")

# ── Step 2: Parse dates and sort chronologically ──────────────────────────────
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"\n[2] Date range : {df['Date'].min().date()} to {df['Date'].max().date()}")

# ── Step 3: Remove duplicates ─────────────────────────────────────────────────
exact_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

date_dupes = df.duplicated(subset="Date").sum()
df = df.drop_duplicates(subset="Date", keep="first").reset_index(drop=True)

print(f"\n[3] Duplicate exact rows removed : {exact_dupes}")
print(f"    Duplicate date rows removed  : {date_dupes}")

# ── Step 4: Convert all value columns to numeric ──────────────────────────────
value_cols = [c for c in existing_cols if c != "Date"]
df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

print(f"\n[4] Numeric conversion applied. Columns: {value_cols}")

# ── Step 5: Handle missing values ─────────────────────────────────────────────
print(f"\n[5] Missing values before filling:")
print(df[value_cols].isnull().sum().to_string())

# Forward fill: propagates last known price through gaps and market holidays.
# Backward fill: fills any leading NaNs for commodities that start late.
df[value_cols] = df[value_cols].ffill().bfill()

# Verify zero missing values remain.
remaining = df[value_cols].isnull().sum().sum()
print(f"\n    Missing values after filling : {remaining}")
assert remaining == 0, "ERROR: Missing values still present after ffill/bfill!"
print("    All columns complete — no missing values. [OK]")

# ── Step 6: Save cleaned dataset ──────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[6] Saved : {OUTPUT_CSV}")
print(f"    Final shape : {df.shape}")