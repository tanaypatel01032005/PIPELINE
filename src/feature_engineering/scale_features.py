"""
scale_features.py — Feature Engineering Stage
Standardizes all feature columns in features.csv using StandardScaler.
Saves scaled features to features_scaled.csv and the fitted scaler to data/processed/scaler.pkl.

Input  : data/processed/features.csv
Outputs: data/processed/features_scaled.csv
         data/processed/scaler.pkl
"""

import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    features_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    output_scaled_path = PROJECT_ROOT / "data" / "processed" / "features_scaled.csv"
    scaler_pickle_path = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
    
    # Load features
    df = pd.read_csv(features_path)
    df.set_index("Date", inplace=True)
    
    # Separate numeric features to scale
    # Calendar features (sin/cos) and binary flags do not strictly need scaling,
    # but standardizing everything is safe. Let's scale all columns.
    columns_to_scale = df.columns.tolist()
    
    print("Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    
    # Fit and transform
    scaled_values = scaler.fit_transform(df[columns_to_scale])
    
    # Create scaled DataFrame
    df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=columns_to_scale)
    
    # Save to CSV
    df_scaled.to_csv(output_scaled_path, index=True)
    print(f"Scaled features saved to {output_scaled_path}")
    
    # Save the fitted scaler pickle for out-of-sample scaling
    os.makedirs(scaler_pickle_path.parent, exist_ok=True)
    with open(scaler_pickle_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Fitted scaler saved to {scaler_pickle_path}")
    print("\nScaling completed successfully.")

if __name__ == "__main__":
    main()
