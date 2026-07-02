"""
clean_data.py — Preprocessing & Cleaning Stage
Cleans and prepares the raw merged commodity-currency dataset.

Input  : data/processed/merged_raw_data.csv
Output : data/processed/cleaned_data.csv
"""

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from pathlib import Path

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "mergedFinalData.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"

# Some columns in the CSV use spaces; we rename them to underscores.
RENAME_MAP = {
    "Heating Oil": "Heating_Oil",
    "Lean Hogs": "Lean_Hogs",
    "RBOB Gasoline": "RBOB_Gasoline",
    "WTI Crude Oil": "WTI_Crude_Oil",
}

# Columns required for modelling
COLUMNS = [
    "Date",
    "usd_zar",
    "Brent_Oil",
    "Natural_Gas",
    "US_Soybean_Oil",
    "US_Soybean_Meal",
    "Gold",
    "US_Copper",
    "Silver",
    "US_Sugar",
    "US_Soybeans",
    "Hard_Red_Winter_Wheat",
    "Nickel",
    "Orange_Juice",
    "platinum",
    "Lead",
    "US_beef",
    "Palladium",
    "US_Cotton",
    "Tin",
    "Cocoa",
    "Coffee",
    "Corn",
    "Heating_Oil",
    "Lean_Hogs",
    "Oats",
    "RBOB_Gasoline",
    "Rhodium",
    "Rice",
    "WTI_Crude_Oil",
]

# Columns to remove completely
DROP_COLUMNS = [
    "Lead",
    "Tin",
    "Iron_ore_62Fe",
    "Maize_Feed",
    "Dubai_Crude_oil",
    "Aluminium",
    "Rubber_RSS3",
    "Coal",
    "EU_Natural_Gas",
    "Urea_Granular_FOB_Middle_East",
    "Chromium",
    "Sorghum",
    "Diamond_Index",
    "Manganese",
    "Barley",
]

# Columns that should use Kalman Smoother
KALMAN_COLUMNS = [
    "Orange_Juice",
    "Lean_Hogs",
]

# ==============================================================================
# Step 1 : Load CSV
# ==============================================================================

raw_names = set(COLUMNS) | set(RENAME_MAP.keys()) | set(DROP_COLUMNS)

df = pd.read_csv(
    INPUT_CSV,
    usecols=lambda c: c in raw_names,
    low_memory=False,
)

df.rename(columns=RENAME_MAP, inplace=True)

print(f"[1] Loaded : {INPUT_CSV}")
print(f"    Shape  : {df.shape}")

# ==============================================================================
# Step 2 : Drop unwanted columns
# ==============================================================================

df.drop(columns=DROP_COLUMNS, errors="ignore", inplace=True)

existing_cols = [c for c in COLUMNS if c in df.columns]
df = df[existing_cols]

print(f"\n[2] Remaining Columns ({len(df.columns)}):")
print(list(df.columns))

# ==============================================================================
# Step 3 : Parse Date
# ==============================================================================

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
      .reset_index(drop=True)
)

print(f"\n[3] Date Range")
print(df["Date"].min().date(), "to", df["Date"].max().date())

# ==============================================================================
# Step 4 : Remove Duplicates
# ==============================================================================

exact_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

date_dupes = df.duplicated(subset="Date").sum()
df = df.drop_duplicates(subset="Date").reset_index(drop=True)

print(f"\n[4] Exact duplicates removed : {exact_dupes}")
print(f"    Date duplicates removed  : {date_dupes}")

# ==============================================================================
# Step 5 : Numeric Conversion
# ==============================================================================

value_cols = [c for c in df.columns if c != "Date"]

for col in value_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n[5] Numeric conversion completed.")

# ==============================================================================
# Step 6 : Missing Value Treatment
# ==============================================================================

print("\n[6] Missing Values Before Filling")
print(df[value_cols].isna().sum().to_string())

# Set Date as index for time interpolation
df = df.set_index("Date")

for col in value_cols:

    if col in KALMAN_COLUMNS:

        print(f"Processing {col:<25} --> Kalman Smoother")

        values = df[col].values
        masked = np.ma.masked_invalid(values)

        try:

            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
            )

            # Learn optimal parameters automatically
            kf = kf.em(masked, n_iter=20)

            smoothed_state_means, _ = kf.smooth(masked)

            df[col] = smoothed_state_means.flatten()

        except Exception as e:

            print(f"Kalman failed for {col}: {e}")
            print("Using Time Interpolation instead.")

            df[col] = df[col].interpolate(
                method="time",
                limit_area="inside",
            )

    else:

        print(f"Processing {col:<25} --> Time Interpolation")

        df[col] = df[col].interpolate(
            method="time",
            limit_area="inside",
        )

# Restore Date column
df.reset_index(inplace=True)

print("\nMissing Values After Filling")
print(df[value_cols].isna().sum().to_string())

remaining = df[value_cols].isna().sum().sum()

print(f"\nTotal Remaining Missing Values : {remaining}")

if remaining == 0:
    print("Dataset contains no missing values.")
else:
    print(
        "Remaining NaNs are leading/trailing gaps that were intentionally "
        "left unfilled by time interpolation."
    )

# ==============================================================================
# Step 7 : Save Dataset
# ==============================================================================

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n[7] Saved : {OUTPUT_CSV}")
print(f"    Final Shape : {df.shape}")

print("\nPreprocessing completed successfully.")
