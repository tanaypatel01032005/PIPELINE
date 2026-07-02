"""
multicollinearity_check.py — Analysis Stage
Computes correlation matrix for all features in features.csv, 
identifies highly correlated feature pairs (|r| > 0.85), and saves a heatmap.

Input  : data/processed/features.csv
Outputs: data/results/feature_correlation_matrix.csv
         data/results/feature_correlation_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    features_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    matrix_output_path = PROJECT_ROOT / "data" / "results" / "feature_correlation_matrix.csv"
    heatmap_output_path = PROJECT_ROOT / "data" / "results" / "feature_correlation_heatmap.png"
    
    # Load features
    df = pd.read_csv(features_path)
    df.set_index("Date", inplace=True)
    
    # Drop calendar variables if needed, or keep all numeric features
    numeric_df = df.select_dtypes(include="number").dropna()
    
    # Compute correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = numeric_df.corr()
    
    # Save matrix to CSV
    os.makedirs(matrix_output_path.parent, exist_ok=True)
    corr_matrix.to_csv(matrix_output_path)
    print(f"Correlation matrix saved to {matrix_output_path}")
    
    # Identify highly correlated pairs (|r| > 0.85)
    print("\nPairs with high correlation (|r| > 0.85):")
    print(f"{'Feature 1':<35} | {'Feature 2':<35} | {'Correlation (r)':<15}")
    print("-" * 90)
    
    # Avoid duplicate pairs and self-correlation
    checked_pairs = set()
    high_corr_count = 0
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            r_val = corr_matrix.iloc[i, j]
            
            if abs(r_val) > 0.85:
                print(f"{col1:<35} | {col2:<35} | {r_val:>15.6f}")
                high_corr_count += 1
                
    print(f"\nTotal highly correlated pairs found: {high_corr_count}")
    
    # Generate Heatmap (limit features to prevent unreadable plot if feature set is very large)
    # Since we have 39 features, a 15x15 inches plot will be readable
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        corr_matrix, 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(heatmap_output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {heatmap_output_path}")

if __name__ == "__main__":
    main()
