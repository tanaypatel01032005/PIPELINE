# Modeling Relationship Between Commodity Price and Currency Exchange

This project models and analyzes the relationship between commodity prices and currency exchange rates (specifically USD to ZAR).

## Project Structure

The codebase has been restructured into a clean modular layout:

```
├── data/
│   ├── raw/                           # Raw input files (USD/ZAR and commodity CSVs)
│   ├── processed/                     # Processed intermediate data files
│   └── results/                       # Outputs of statistical tests & results
├── src/                               # Source code directory
│   ├── data_processing/               # Data cleaning and merging scripts
│   │   ├── merge_raw_data.py          # Merges all raw commodity data with ZAR
│   │   └── clean_data.py              # Preprocesses data, handles NaNs (Kalman/Interpolation)
│   ├── feature_engineering/           # Feature extraction and transformation
│   │   └── calculate_log_returns.py   # Computes log returns of variables
│   └── analysis/                      # Statistical tests and analysis
│       ├── descriptive_stats.py       # Computes descriptive statistics
│       ├── stationarity_tests.py      # Performs ADF and KPSS stationarity tests
│       ├── normality_tests.py         # Performs Jarque-Bera normality tests
│       └── granger_causality.py       # Pearson correlation & Granger causality feature selection
├── requirements.txt                   # Python package dependencies
└── README.md                          # Project documentation
```

## Setup and Dependencies

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Running the Code

Scripts can be executed individually from the project root directory. All script paths are resolved dynamically using absolute path lookup relative to their script files, making execution robust from any directory.

For example, to run the descriptive statistics test:
```bash
python src/analysis/descriptive_stats.py
```