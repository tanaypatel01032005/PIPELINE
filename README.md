# Modeling Relationship Between Commodity Price and Currency Exchange

This project models and analyzes the relationship between commodity prices and currency exchange rates (specifically USD to ZAR). It provides a full data processing, analysis, and feature engineering pipeline to generate optimal features for forecasting models.

---

## Project Structure

The codebase is organized into a clean, modular structure:

```
├── data/
│   ├── raw/                           # Raw input files (USD/ZAR and commodity CSVs)
│   ├── processed/                     # Cleaned, engineered, and scaled intermediate datasets
│   │   ├── preprocessed_data.csv      # Cleaned daily prices (25 commodities + USD/ZAR)
│   │   ├── features.csv               # Raw engineered features (39 columns)
│   │   ├── features_scaled.csv        # Z-score standardized features (39 columns)
│   │   └── scaler.pkl                 # Fitted StandardScaler object
│   └── results/                       # Outputs of statistical tests & visual plots
│       ├── descriptive_statistics.csv # Descriptive statistics output
│       ├── stationarity_results.csv   # ADF & KPSS stationarity outcomes
│       ├── normality_results.csv      # Jarque-Bera normality outcomes
│       ├── correlation_granger_results.csv # Granger causality test statistics
│       ├── feature_correlation_matrix.csv  # Pearson correlation matrix
│       └── feature_correlation_heatmap.png # Visual correlation heatmap
├── src/                               # Source code directory
│   ├── data_processing/               # Data cleaning and merging scripts
│   │   ├── merge_raw_data.py          # Merges raw commodity CSVs with USD/ZAR price
│   │   └── clean_data.py              # Handles column mappings and Kalman smoothing
│   ├── feature_engineering/           # Feature extraction, scaling, and returns
│   │   ├── calculate_log_returns.py   # Computes log returns of prices
│   │   ├── build_features.py          # Constructs the 39-feature set (optimal lags/rolling/technical)
│   │   └── scale_features.py          # Standardizes features and exports StandardScaler pickle
│   └── analysis/                      # Statistical tests and diagnostics
│       ├── descriptive_stats.py       # Computes descriptive statistics
│       ├── stationarity_tests.py      # Performs ADF and KPSS stationarity tests
│       ├── normality_tests.py         # Performs Jarque-Bera normality tests
│       ├── granger_causality.py       # Granger causality feature selection (95% confidence)
│       └── multicollinearity_check.py # Computes correlation matrix and prints warning pairs (>0.85)
├── requirements.txt                   # Python package dependencies
└── README.md                          # Project documentation
```

---

## Setup and Dependencies

Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Pipeline

The scripts must be executed in the following order:

### 1. Data Cleaning
Calculates log returns of the preprocessed prices:
```bash
python src/feature_engineering/calculate_log_returns.py
```

### 2. Statistical Analysis (EDA)
Runs diagnostic tests on stationary log returns to understand their properties:
```bash
python src/analysis/descriptive_stats.py
python src/analysis/stationarity_tests.py
python src/analysis/normality_tests.py
```

### 3. Granger Causality & Feature Selection
Identifies which commodities Granger-cause USD/ZAR price movements at a **95% confidence level ($\alpha = 0.05$)**:
```bash
python src/analysis/granger_causality.py
```
This isolates exactly **9 significant commodities** and registers their optimal lead-lag relationship (`Best_Lag`):
* `platinum_logret` (Best Lag: 1)
* `Gold_logret` (Best Lag: 2)
* `Silver_logret` (Best Lag: 2)
* `Natural_Gas_logret` (Best Lag: 1)
* `Brent_Oil_logret` (Best Lag: 1)
* `Palladium_logret` (Best Lag: 3)
* `Lean_Hogs_logret` (Best Lag: 1)
* `Oats_logret` (Best Lag: 4)
* `RBOB_Gasoline_logret` (Best Lag: 1)

### 4. Feature Generation
Generates a highly-focused, non-redundant set of **39 forecasting features**:
```bash
python src/feature_engineering/build_features.py
```
Features are grouped into:
1. **Lags**: Target returns lagged 1-2 days, and each of the 9 commodities lagged at their optimal `Best_Lag`.
2. **Rolling Statistics**: 10-day rolling mean & standard deviation for each commodity.
3. **Macro Spreads**: Gold-Silver, Brent-WTI, and Platinum-Palladium return spreads.
4. **Volatility Regime**: USD/ZAR 20-day rolling volatility and high/low volatility regime flags.
5. **Technical Indicators**: RSI(14), MACD Line, and Bollinger Band Width(20) on USD/ZAR.
6. **Calendar cycle**: cyclical month sine and cosine components.

### 5. Diagnostics and Multicollinearity Check
Analyzes the feature matrix to ensure no highly correlated variables ($|r| > 0.85$) are fed into linear/regression models:
```bash
python src/analysis/multicollinearity_check.py
```
*Outputs: `feature_correlation_matrix.csv` and a visual `feature_correlation_heatmap.png` in the results folder.*

### 6. Feature Scaling
Standardizes the raw feature matrix for model training:
```bash
python src/feature_engineering/scale_features.py
```
*Outputs: `features_scaled.csv` (scaled features) and `scaler.pkl` (fitted StandardScaler).*

---

## Data Leakage & Look-Ahead Bias Mitigation

Data leakage occurs when information from the future (e.g. test set or validation folds) is inadvertently used to train a model or extract features, leading to overly optimistic results that fail in live trading.

This project implements strict guards and clear guidelines to eliminate leakage:

### 1. Shifted Features (Temporal Guard)
In [build_features.py](file:///t:/MINOR/GITHUB/Modeling-Relationship-Between-Commodity-Price-and-Currency-Exchange/src/feature_engineering/build_features.py), every single contemporaneous indicator (including technical indicators, spreads, and rolling volatilities) is **shifted by at least 1 day** (using `.shift(1)`).
* At any trading day $t$, the features only contain historical information from day $t-1$ or earlier to predict the exchange rate return at day $t$.
* There is no look-ahead leakage in the feature matrix `features.csv`.

### 2. Boundary Between EDA and Modeling
Diagnostic scripts (such as Granger Causality selection, normality checks, and KPSS/ADF tests) are run globally over the full dataset (2000–2025). This is standard for **Exploratory Data Analysis (EDA)**.
* **Warning**: Using these global test outcomes to select features for final model testing introduces minor selection bias (leakage).
* **Mitigation**: When evaluating a machine learning model, you should wrap feature selection (e.g. running Granger causality or selecting indicators) inside your **walk-forward validation folds**. Specifically, you must run Granger causality *only* on the training split of the current fold, and use those selected features to forecast the test split.

### 3. Scaler and Parameter Leakage
Fitting a scaler (like `StandardScaler` or `MinMaxScaler`) globally over the entire dataset leaks the future mean and variance of the test set into the training phase.
* **Mitigation**:
  1. Fit your scaler **only** on the training partition of your dataset: `scaler.fit(X_train)`
  2. Scale both the train and test sets using the fitted parameters: `X_train_scaled = scaler.transform(X_train)`, `X_test_scaled = scaler.transform(X_test)`
  3. The utility script `scale_features.py` saves a fitted `scaler.pkl` model to demonstrate how a fitted scaler should be persisted and reused for out-of-sample data.