import pandas as pd

# Selected commodities only
COMMODITIES = [
    "Brent_Oil", "Natural_Gas", "US_Soybean_Oil", "US_Soybean_Meal", "Gold",
    "US_Copper", "Silver", "US_Sugar", "US_Soybeans", "Hard_Red_Winter_Wheat",
    "usd_zar", "Nickel", "Orange_Juice", "platinum", "Lead", "US_beef",
    "Palladium", "US_Cotton", "Tin", "Iron_ore_62Fe"
]

def preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Date"] + COMMODITIES)

    # Parse and sort dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates().reset_index(drop=True)

    # Numeric conversion (coerce bad values to NaN)
    df[COMMODITIES] = df[COMMODITIES].apply(pd.to_numeric, errors="coerce")

    # Forward-fill then back-fill missing values.
    # ffill: uses the last known price (natural for financial time-series).
    # bfill: handles any NaNs at the very start of the series.
    df[COMMODITIES] = df[COMMODITIES].ffill().bfill()

    return df


if __name__ == "__main__":
    df = preprocess("data/processed/mergedFinalData.csv")
    print(df.shape)
    print(df.head())
    df.to_csv("data/processed/mergedFinalData.csv", index=False)
    print("Saved → mergedFinalData_clean.csv")