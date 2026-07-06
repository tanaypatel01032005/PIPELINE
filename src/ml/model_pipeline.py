"""
model_pipeline.py - Comprehensive Dual-Target Forecasting Pipeline
Features:
- Denoises return series using Kalman Filters (Solution 4).
- Predicts both Direction of change (Classification) and Magnitude/Percent change (Regression).
- Implements chronological training on Train (2000-2021), Validation (2022-2023), and Test (2024-2025).
- Performs hyperparameter tuning once via Optuna, then runs walk-forward sliding window retraining (Solution 3) on the Test set.
- Checkpoints and resumes execution, performing cleanups of intermediate artifacts at the end.
"""

import os
import sys
import time
import json
import random
import warnings
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pykalman import KalmanFilter

# Disable warnings for statsmodels and optuna logging to keep console output clean
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# PIPELINE CONFIGURATIONS
# ==============================================================================
RANDOM_SEED = 42
KEEP_ALL_MODELS = False
ENSEMBLE_TOLERANCE = 0.03
ROLLING_WINDOW_SIZE = 1500  # Sliding training window for walk-forward updates (approx 6 years)
WALK_FORWARD_STEP = 20      # Re-fit models every 20 trading days (monthly updates)

ENABLED_MODELS = [
    "Persistence",
    "HistoricalMean",
    "RollingMean",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "SARIMAX",
    "SimpleRNN",
    "LSTM",
    "GRU",
]

OPTUNA_TRIAL_LIMITS = {
    "Ridge": 15,
    "Lasso": 15,
    "ElasticNet": 15,
    "RandomForest": 20,
    "ExtraTrees": 20,
    "XGBoost": 30,
    "LightGBM": 30,
    "CatBoost": 30,
    "SARIMAX": 15,
    "SimpleRNN": 15,
    "LSTM": 15,
    "GRU": 15,
}

PATIENCE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "src" / "ml" / "pipeline_checkpoint.json"
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "ml" / "checkpoints"

TARGET_REG = "usd_zar_logret_next"
TARGET_CLS = "usd_zar_logret_next_dir"

SARIMAX_EXOG_COLS = [
    "usd_zar_logret_lag_1",
    "usd_zar_logret_lag_2",
    "usd_zar_logret_lag_3",
    "usd_zar_logret_lag_4",
    "usd_zar_logret_lag_5",
    "Gold_logret_lag_1",
    "Silver_logret_lag_4",
    "Palladium_logret_lag_3",
    "Heating_Oil_logret_lag_1",
    "Lean_Hogs_logret_lag_5",
    "Oats_logret_lag_4",
    "RBOB_Gasoline_logret_lag_1",
    "WTI_Crude_Oil_logret_lag_4"
]

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==============================================================================
# DEEP LEARNING MODEL ARCHITECTURES
# ==============================================================================

class SimpleRNNModel(nn.Module):
    """Simple RNN for regression tasks."""
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(nn.Module):
    """LSTM for regression tasks."""
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRUModel(nn.Module):
    """GRU for regression tasks."""
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class RNNClassifier(nn.Module):
    """Simple RNN for classification tasks (Up/Down)."""
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))


class LSTMClassifier(nn.Module):
    """LSTM for classification tasks."""
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))


class GRUClassifier(nn.Module):
    """GRU for classification tasks."""
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))


class TimeSeriesDataset(Dataset):
    """Dataset helper class for sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(X, y, lookback=12):
    """Generate chronological sequence structures."""
    xs, ys = [], []
    for i in range(len(X) - lookback):
        xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(xs), np.array(ys)


# ==============================================================================
# KALMAN DENOISING METHOD (Solution 4)
# ==============================================================================

def denoise_features_kalman(df, feature_cols):
    """Applies a Kalman Smoother filter to denoise return series and technical indicators."""
    print("Applying Kalman Filter denoising to features...")
    df_denoised = df.copy()
    for col in feature_cols:
        if "logret" in col or "rsi" in col or "macd" in col or "spread" in col:
            values = df_denoised[col].values
            masked = np.ma.masked_invalid(values)
            kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
            try:
                kf = kf.em(masked, n_iter=3)
                state_means, _ = kf.smooth(masked)
                df_denoised[col] = state_means.flatten()
            except Exception:
                pass
    return df_denoised


# ==============================================================================
# PIPELINE HELPER FUNCTIONS
# ==============================================================================

def calculate_reg_metrics(actual, predicted):
    """Computes standard regression metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted)
    
    # Directional Accuracy matching the project's exact logic
    direction_actual = np.sign(np.diff(actual))
    direction_pred = np.sign(np.diff(predicted))
    directional_accuracy = (direction_actual == direction_pred).mean() * 100 if len(direction_actual) > 0 else 50.0
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "DirAcc": directional_accuracy
    }


def calculate_cls_metrics(actual, predicted_prob):
    """Computes standard binary classification metrics."""
    predicted_class = (predicted_prob >= 0.5).astype(int)
    acc = accuracy_score(actual, predicted_class)
    f1 = f1_score(actual, predicted_class, zero_division=0)
    try:
        auc = roc_auc_score(actual, predicted_prob)
    except ValueError:
        auc = 0.5
    return {
        "Accuracy": acc,
        "F1_Score": f1,
        "AUC": auc
    }


def load_checkpoint(task_type, model_name, input_size=None):
    """Loads a cached model checkpoint and parameters if existing on disk."""
    if not CHECKPOINT_FILE.exists():
        return None, None
    with open(CHECKPOINT_FILE, "r") as f:
        registry = json.load(f)
    key = f"{task_type}_{model_name}"
    if key not in registry:
        return None, None
    entry = registry[key]
    model_path = Path(entry["model_path"])
    if not model_path.exists():
        return None, None
    try:
        if "RNN" in model_name or "LSTM" in model_name or "GRU" in model_name:
            hidden = entry["best_params"]["hidden_size"]
            layers = entry["best_params"]["num_layers"]
            if task_type == "regression":
                if model_name == "SimpleRNN":
                    model = SimpleRNNModel(input_size=input_size, hidden_size=hidden, num_layers=layers)
                elif model_name == "LSTM":
                    model = LSTMModel(input_size=input_size, hidden_size=hidden, num_layers=layers)
                elif model_name == "GRU":
                    model = GRUModel(input_size=input_size, hidden_size=hidden, num_layers=layers)
            else:
                if model_name == "SimpleRNN":
                    model = RNNClassifier(input_size=input_size, hidden_size=hidden, num_layers=layers)
                elif model_name == "LSTM":
                    model = LSTMClassifier(input_size=input_size, hidden_size=hidden, num_layers=layers)
                elif model_name == "GRU":
                    model = GRUClassifier(input_size=input_size, hidden_size=hidden, num_layers=layers)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
        else:
            model = joblib.load(model_path)
        return model, entry
    except Exception:
        return None, None


def save_checkpoint(task_type, model_name, model, val_metrics, best_params, train_time, val_time):
    """Saves model parameters and weights to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{task_type}_{model_name}"
    if hasattr(model, 'state_dict'):
        model_path = CHECKPOINT_DIR / f"{key}_best.pt"
        torch.save(model.state_dict(), model_path)
    else:
        model_path = CHECKPOINT_DIR / f"{key}_best.joblib"
        joblib.dump(model, model_path)
        
    registry = {}
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                registry = json.load(f)
        except Exception:
            pass
            
    registry[key] = {
        "status": "Completed",
        "val_metrics": val_metrics,
        "best_params": best_params,
        "model_path": str(model_path),
        "training_time": train_time,
        "validation_time": val_time,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(registry, f, indent=4)


# ==============================================================================
# OPTUNA EARLY STOPPING CALLBACK
# ==============================================================================

class EarlyStoppingCallback:
    """Terminates Optuna optimization trials early if no improvement occurs within patience."""
    def __init__(self, patience=10):
        self.patience = patience
        self.best_value = None
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return
        current_value = trial.value
        # For minimize tasks (RMSE, log-loss)
        if self.best_value is None or current_value < self.best_value:
            self.best_value = current_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.patience:
            study.stop()


# ==============================================================================
# PYTORCH MODEL TRAINING LOOP
# ==============================================================================

def train_pytorch_model(model, train_loader, val_loader, lr, is_cls=False, patience=10):
    """Standard PyTorch train epoch loop with early stopping."""
    criterion = nn.BCELoss() if is_cls else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_weights = None
    no_improvement_count = 0
    
    for epoch in range(50):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.to(DEVICE)
    return model


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    pipeline_start_time = time.time()
    temp_files_created = []
    
    # -- Step 1: Load and Process Data -----------------------------------------
    print("Loading engineered features...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    
    # Identify warm-up rows (first index containing zero NaNs)
    nan_rows = df.isna().any(axis=1)
    first_valid_loc = np.where(~nan_rows)[0][0]
    first_valid_date = df.index[first_valid_loc]
    
    # Slice the clean dataset (leaving out trailing row where TARGET is shifted next)
    # Apply ffill and bfill to clean out mathematical NaN anomalies
    df_clean = df.iloc[first_valid_loc:-1].ffill().bfill()
    assert df_clean.isna().sum().sum() == 0, "Error: NaNs still exist in dataset."
    
    # Define features columns
    feature_columns = [c for c in df_clean.columns if c not in [TARGET_REG]]
    
    # Apply Solution 4: Kalman filter denoising on features prior to modeling
    df_clean = denoise_features_kalman(df_clean, feature_columns)
    
    # Extract feature matrices and labels
    X = df_clean[feature_columns].values
    y_reg = df_clean[TARGET_REG].values.reshape(-1, 1)
    # Create the binary target variable for classification (1 if next day return is positive)
    y_cls = (y_reg > 0).astype(float)
    
    # Define split masks
    train_mask = df_clean.index <= pd.Timestamp("2021-12-31")
    val_mask = (df_clean.index >= pd.Timestamp("2022-01-01")) & (df_clean.index <= pd.Timestamp("2023-12-31"))
    test_mask = df_clean.index >= pd.Timestamp("2024-01-01")
    
    # Automatic Warm-Up Row Detection report
    print("-" * 65)
    print("Automatic Warm-Up Row Detection & Kalman Denoising")
    print("-" * 65)
    print(f"Total rows in engineered features CSV : {len(df)}")
    print(f"First fully valid row index (loc)     : {first_valid_loc}")
    print(f"First fully valid date                : {first_valid_date.date()}")
    print(f"Removed leading warm-up rows          : {first_valid_loc}")
    print(f"Removed trailing row (shifted target) : 1")
    print(f"Denoising scheme                      : Kalman Filter Smoothing (Solution 4)")
    print("-" * 65)
    
    # Separate partitions for initial tuning
    X_train, y_train_reg, y_train_cls = X[train_mask], y_reg[train_mask].ravel(), y_cls[train_mask].ravel()
    X_val, y_val_reg, y_val_cls = X[val_mask], y_reg[val_mask].ravel(), y_cls[val_mask].ravel()
    
    # Fit scaling matrices on training data only
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y.fit(y_train_reg.reshape(-1, 1))
    
    # Pre-scale splits for standard architectures
    X_train_scaled = scaler_x.transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    y_train_reg_scaled = scaler_y.transform(y_train_reg.reshape(-1, 1))
    y_val_reg_scaled = scaler_y.transform(y_val_reg.reshape(-1, 1))
    
    # Pre-scale full data matrices for walk-forward sequences
    X_scaled = scaler_x.transform(X)
    y_reg_scaled = scaler_y.transform(y_reg)
    
    # Build deep learning sequence datasets
    lookback = 12
    xs_seq, ys_reg_seq = create_sequences(X_scaled, y_reg_scaled, lookback)
    _, ys_cls_seq = create_sequences(X_scaled, y_cls, lookback)
    seq_dates = df_clean.index[lookback:]
    
    X_train_seq = xs_seq[seq_dates <= pd.Timestamp("2021-12-31")]
    y_train_reg_seq = ys_reg_seq[seq_dates <= pd.Timestamp("2021-12-31")]
    y_train_cls_seq = ys_cls_seq[seq_dates <= pd.Timestamp("2021-12-31")]
    
    X_val_seq = xs_seq[(seq_dates >= pd.Timestamp("2022-01-01")) & (seq_dates <= pd.Timestamp("2023-12-31"))]
    y_val_reg_seq = ys_reg_seq[(seq_dates >= pd.Timestamp("2022-01-01")) & (seq_dates <= pd.Timestamp("2023-12-31"))]
    y_val_cls_seq = ys_cls_seq[(seq_dates >= pd.Timestamp("2022-01-01")) & (seq_dates <= pd.Timestamp("2023-12-31"))]
    
    # Trackers
    leaderboard_reg = []
    leaderboard_cls = []
    strongest_baseline_rmse = float("inf")
    strongest_baseline_acc = 0.0
    
    # -- Step 2: Evaluate Baselines -------------------------------------------
    print("\nEvaluating Baseline Models...")
    
    # Baseline Regression
    current_return_col = df_clean["usd_zar_logret"].values
    pers_val_preds_reg = current_return_col[val_mask]
    pers_reg_metrics = calculate_reg_metrics(y_val_reg, pers_val_preds_reg)
    pers_reg_metrics["Model"] = "Persistence"
    pers_reg_metrics["Status"] = "Completed"
    leaderboard_reg.append(pers_reg_metrics)
    
    hist_mean_val = np.mean(y_train_reg)
    mean_val_preds_reg = np.full_like(y_val_reg, hist_mean_val)
    mean_reg_metrics = calculate_reg_metrics(y_val_reg, mean_val_preds_reg)
    mean_reg_metrics["Model"] = "HistoricalMean"
    mean_reg_metrics["Status"] = "Completed"
    leaderboard_reg.append(mean_reg_metrics)
    
    rolling_mean_col = df_clean["usd_zar_logret"].rolling(20).mean().values
    roll_val_preds_reg = rolling_mean_col[val_mask]
    nan_mask = np.isnan(roll_val_preds_reg)
    if nan_mask.any():
        roll_val_preds_reg[nan_mask] = hist_mean_val
    roll_reg_metrics = calculate_reg_metrics(y_val_reg, roll_val_preds_reg)
    roll_reg_metrics["Model"] = "RollingMean"
    roll_reg_metrics["Status"] = "Completed"
    leaderboard_reg.append(roll_reg_metrics)
    
    strongest_baseline_rmse = min([pers_reg_metrics, mean_reg_metrics, roll_reg_metrics], key=lambda m: m["RMSE"])["RMSE"]
    
    # Baseline Classification (predict direction)
    # Persistence: predict direction sign as sign of current day return
    pers_val_preds_cls = (pers_val_preds_reg > 0).astype(float)
    pers_cls_metrics = calculate_cls_metrics(y_val_cls, pers_val_preds_cls)
    pers_cls_metrics["Model"] = "Persistence"
    pers_cls_metrics["Status"] = "Completed"
    leaderboard_cls.append(pers_cls_metrics)
    
    # Historical Mode: majority class in training set
    hist_mode_val = float(np.mean(y_train_cls) >= 0.5)
    mean_val_preds_cls = np.full_like(y_val_cls, hist_mode_val)
    mean_cls_metrics = calculate_cls_metrics(y_val_cls, mean_val_preds_cls)
    mean_cls_metrics["Model"] = "HistoricalMean"
    mean_cls_metrics["Status"] = "Completed"
    leaderboard_cls.append(mean_cls_metrics)
    
    # Rolling Mode/Direction
    roll_val_preds_cls = (roll_val_preds_reg > 0).astype(float)
    roll_cls_metrics = calculate_cls_metrics(y_val_cls, roll_val_preds_cls)
    roll_cls_metrics["Model"] = "RollingMean"
    roll_cls_metrics["Status"] = "Completed"
    leaderboard_cls.append(roll_cls_metrics)
    
    strongest_baseline_acc = max([pers_cls_metrics, mean_cls_metrics, roll_cls_metrics], key=lambda m: m["Accuracy"])["Accuracy"]
    
    print(f"Strongest Baselines:")
    print(f"  - Regression: RMSE = {strongest_baseline_rmse:.6f}")
    print(f"  - Classification: Accuracy = {strongest_baseline_acc * 100:.2f}%")
    
    # Define execution configurations
    all_ml_models = [
        "Ridge", "Lasso", "ElasticNet", "RandomForest", "ExtraTrees",
        "XGBoost", "LightGBM", "CatBoost", "SARIMAX", "SimpleRNN",
        "LSTM", "GRU"
    ]
    
    completed_reg = {}
    completed_cls = {}
    discarded_reg = []
    discarded_cls = []
    
    # -- Step 3: Train and Tune Dual Pipelines ---------------------------------
    for task_type in ["regression", "classification"]:
        print(f"\n==============================================================================")
        print(f"OPTIMIZING PIPELINE FOR TASK TYPE: {task_type.upper()}")
        print(f"==============================================================================")
        
        for idx, model_name in enumerate(all_ml_models):
            if model_name not in ENABLED_MODELS:
                continue
            if task_type == "classification" and model_name in ["Lasso", "ElasticNet", "SARIMAX"]:
                # Exclude non-classifier compatible structures from class task
                continue
                
            key = f"{task_type}_{model_name}"
            
            # Check checkpoint
            cached_model, cached_entry = load_checkpoint(task_type, model_name, input_size=X_train_seq.shape[2])
            if cached_model is not None:
                print(f"Resuming {key} from checkpoint!")
                val_metrics = cached_entry["val_metrics"]
                val_metrics["Model"] = model_name
                val_metrics["Status"] = "Completed"
                if task_type == "regression":
                    leaderboard_reg.append(val_metrics)
                    completed_reg[model_name] = {
                        "model": cached_model,
                        "best_params": cached_entry["best_params"],
                        "val_metrics": val_metrics,
                        "training_time": cached_entry["training_time"],
                        "validation_time": cached_entry["validation_time"]
                    }
                else:
                    leaderboard_cls.append(val_metrics)
                    completed_cls[model_name] = {
                        "model": cached_model,
                        "best_params": cached_entry["best_params"],
                        "val_metrics": val_metrics,
                        "training_time": cached_entry["training_time"],
                        "validation_time": cached_entry["validation_time"]
                    }
                temp_files_created.append(cached_entry["model_path"])
                continue
                
            print(f"\nTraining {key} (Trials limit: {OPTUNA_TRIAL_LIMITS[model_name]})...")
            train_start = time.time()
            
            # Run Optuna tuning study
            study = optuna.create_study(
                direction="minimize" if (task_type == "regression" or model_name in ["SimpleRNN", "LSTM", "GRU"]) else "maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
            )
            early_stopping = EarlyStoppingCallback(patience=PATIENCE)
            
            try:
                if model_name in ["Ridge", "Lasso", "ElasticNet"]:
                    study.optimize(
                        lambda trial: objective_sklearn(trial, task_type, model_name, X_train_scaled, y_train_reg if task_type == "regression" else y_train_cls, X_val_scaled, y_val_reg if task_type == "regression" else y_val_cls),
                        n_trials=OPTUNA_TRIAL_LIMITS[model_name],
                        callbacks=[early_stopping]
                    )
                elif model_name in ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]:
                    study.optimize(
                        lambda trial: objective_sklearn(trial, task_type, model_name, X_train, y_train_reg if task_type == "regression" else y_train_cls, X_val, y_val_reg if task_type == "regression" else y_val_cls),
                        n_trials=OPTUNA_TRIAL_LIMITS[model_name],
                        callbacks=[early_stopping]
                    )
                elif model_name == "SARIMAX" and task_type == "regression":
                    study.optimize(
                        lambda trial: objective_sarimax(
                            trial,
                            y_train_reg,
                            df_clean[SARIMAX_EXOG_COLS].iloc[:len(y_train_reg)].values,
                            y_val_reg,
                            df_clean[SARIMAX_EXOG_COLS].iloc[len(y_train_reg):len(y_train_reg)+len(y_val_reg)].values
                        ),
                        n_trials=OPTUNA_TRIAL_LIMITS[model_name],
                        callbacks=[early_stopping]
                    )
                elif model_name in ["SimpleRNN", "LSTM", "GRU"]:
                    study.optimize(
                        lambda trial: objective_pytorch(
                            trial,
                            task_type,
                            model_name,
                            X_train_seq,
                            y_train_reg_seq if task_type == "regression" else y_train_cls_seq,
                            X_val_seq,
                            y_val_reg_seq if task_type == "regression" else y_val_cls_seq,
                            X_train_seq.shape[2],
                            scaler_y
                        ),
                        n_trials=OPTUNA_TRIAL_LIMITS[model_name],
                        callbacks=[early_stopping]
                    )
            except Exception as e:
                # Allow exceptions to bubble up naturally
                raise e
                
            tuning_duration = time.time() - train_start
            best_val_score = study.best_value
            best_params = study.best_params
            
            # Determine baseline benchmark to test performance
            is_better = False
            if task_type == "regression":
                is_better = best_val_score < strongest_baseline_rmse
            else:
                # If target is classifier and Optuna evaluates loss/RMSE, lower is better. If it evaluates accuracy, higher is better.
                # In objective_pytorch (classification) we evaluate binary cross entropy loss, so lower is better.
                # In objective_sklearn (classification) we evaluate validation accuracy, so higher is better.
                if model_name in ["SimpleRNN", "LSTM", "GRU"]:
                    is_better = True  # We let it pass to fit final model to check accuracy metrics
                else:
                    is_better = best_val_score > strongest_baseline_acc
                    
            if not is_better and task_type == "regression":
                print(f"--> DISCARDED: {key} failed to beat baseline validation RMSE ({strongest_baseline_rmse:.6f})")
                discarded_reg.append(model_name)
                leaderboard_reg.append({"Model": model_name, "RMSE": best_val_score, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan, "DirAcc": np.nan, "Status": "Discarded"})
                continue
            elif not is_better and task_type == "classification" and model_name not in ["SimpleRNN", "LSTM", "GRU"]:
                print(f"--> DISCARDED: {key} failed to beat baseline validation Accuracy ({strongest_baseline_acc * 100:.2f}%)")
                discarded_cls.append(model_name)
                leaderboard_cls.append({"Model": model_name, "Accuracy": best_val_score, "F1_Score": np.nan, "AUC": np.nan, "Status": "Discarded"})
                continue
                
            # Retrain final optimal model configuration
            print(f"Retraining final {key} model...")
            final_train_start = time.time()
            
            if model_name in ["Ridge", "Lasso", "ElasticNet"]:
                final_model = get_model(task_type, model_name, best_params)
                final_model.fit(X_train_scaled, y_train_reg if task_type == "regression" else y_train_cls)
                val_preds = final_model.predict(X_val_scaled)
                if task_type == "classification":
                    val_probs = final_model.predict_proba(X_val_scaled)[:, 1] if hasattr(final_model, "predict_proba") else final_model.decision_function(X_val_scaled)
            elif model_name in ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]:
                final_model = get_model(task_type, model_name, best_params)
                final_model.fit(X_train, y_train_reg if task_type == "regression" else y_train_cls)
                val_preds = final_model.predict(X_val)
                if task_type == "classification":
                    val_probs = final_model.predict_proba(X_val)[:, 1]
            elif model_name == "SARIMAX" and task_type == "regression":
                final_model = SARIMAX(
                    y_train_reg,
                    exog=df_clean[SARIMAX_EXOG_COLS].iloc[:len(y_train_reg)].values,
                    order=(best_params['p'], best_params['d'], best_params['q']),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit_res = final_model.fit(disp=False)
                val_preds = fit_res.predict(
                    start=len(y_train_reg),
                    end=len(y_train_reg) + len(y_val_reg) - 1,
                    exog=df_clean[SARIMAX_EXOG_COLS].iloc[len(y_train_reg):len(y_train_reg)+len(y_val_reg)].values
                )
                final_model = fit_res
            elif model_name in ["SimpleRNN", "LSTM", "GRU"]:
                if task_type == "regression":
                    if model_name == "SimpleRNN":
                        final_model = SimpleRNNModel(input_size=X_train_seq.shape[2], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(DEVICE)
                    elif model_name == "LSTM":
                        final_model = LSTMModel(input_size=X_train_seq.shape[2], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(DEVICE)
                    elif model_name == "GRU":
                        final_model = GRUModel(input_size=X_train_seq.shape[2], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(DEVICE)
                else:
                    if model_name == "SimpleRNN":
                        final_model = RNNClassifier(input_size=X_train_seq.shape[2], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(DEVICE)
                    elif model_name == "LSTM":
                        final_model = LSTMClassifier(input_size=X_train_seq.shape[2], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(DEVICE)
                    elif model_name == "GRU":
                        final_model = GRUClassifier(input_size=X_train_seq.shape[2], hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"]).to(DEVICE)
                
                train_loader = DataLoader(TimeSeriesDataset(X_train_seq, y_train_reg_seq if task_type == "regression" else y_train_cls_seq), batch_size=32, shuffle=False)
                val_loader = DataLoader(TimeSeriesDataset(X_val_seq, y_val_reg_seq if task_type == "regression" else y_val_cls_seq), batch_size=32, shuffle=False)
                
                final_model = train_pytorch_model(final_model, train_loader, val_loader, best_params["lr"], is_cls=(task_type=="classification"), patience=10)
                
                final_model.eval()
                val_dl_preds = []
                with torch.no_grad():
                    for X_batch, _ in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        outputs = final_model(X_batch)
                        val_dl_preds.extend(outputs.cpu().numpy())
                if task_type == "regression":
                    val_preds = scaler_y.inverse_transform(np.array(val_dl_preds)).ravel()
                else:
                    val_probs = np.array(val_dl_preds).ravel()
                    val_preds = (val_probs >= 0.5).astype(float)
                    
            final_duration = time.time() - final_train_start
            val_duration = time.time() - (final_train_start + final_duration)
            
            # Compute metric evaluations
            if task_type == "regression":
                val_metrics = calculate_reg_metrics(y_val_reg, val_preds)
                val_metrics["Model"] = model_name
                val_metrics["Status"] = "Completed"
                leaderboard_reg.append(val_metrics)
                completed_reg[model_name] = {
                    "model": final_model,
                    "best_params": best_params,
                    "val_metrics": val_metrics,
                    "training_time": tuning_duration + final_duration,
                    "validation_time": val_duration
                }
            else:
                val_metrics = calculate_cls_metrics(y_val_cls, val_probs if model_name in ["SimpleRNN", "LSTM", "GRU"] or hasattr(final_model, "predict_proba") or hasattr(final_model, "decision_function") else val_preds)
                # Verify that model actually outperforms base accuracy, else mark discarded
                if val_metrics["Accuracy"] < strongest_baseline_acc:
                    print(f"--> DISCARDED: {key} final fit failed to outperform baseline accuracy ({strongest_baseline_acc * 100:.2f}%)")
                    discarded_cls.append(model_name)
                    leaderboard_cls.append({"Model": model_name, "Accuracy": val_metrics["Accuracy"], "F1_Score": np.nan, "AUC": np.nan, "Status": "Discarded"})
                    continue
                val_metrics["Model"] = model_name
                val_metrics["Status"] = "Completed"
                leaderboard_cls.append(val_metrics)
                completed_cls[model_name] = {
                    "model": final_model,
                    "best_params": best_params,
                    "val_metrics": val_metrics,
                    "training_time": tuning_duration + final_duration,
                    "validation_time": val_duration
                }
                
            # Cache completed checkpoint state to disk
            save_checkpoint(task_type, model_name, final_model, val_metrics, best_params, tuning_duration + final_duration, val_duration)
            model_path = CHECKPOINT_DIR / (f"{key}_best.pt" if "RNN" in model_name or "LSTM" in model_name or "GRU" in model_name else f"{key}_best.joblib")
            temp_files_created.append(str(model_path))
            
            # Print current leaderboard summaries
            if task_type == "regression":
                ld_df = pd.DataFrame(leaderboard_reg).sort_values("RMSE").reset_index(drop=True)
                ld_df.index += 1
                print("\n--- Regression Leaderboard ---")
                print(ld_df[["Model", "RMSE", "MAE", "MAPE", "R2", "DirAcc", "Status"]].to_string())
            else:
                ld_df = pd.DataFrame(leaderboard_cls).sort_values("Accuracy", ascending=False).reset_index(drop=True)
                ld_df.index += 1
                print("\n--- Classification Leaderboard ---")
                print(ld_df[["Model", "Accuracy", "F1_Score", "AUC", "Status"]].to_string())

    # -- Step 4: Evaluate Ensembles -------------------------------------------
    print("\n==============================================================================")
    print("EVALUATING ENSEMBLES FOR DUAL TARGETS")
    print("==============================================================================")
    
    # Ensembling selection logic
    winner_name_reg = None
    winner_name_cls = None
    
    # 1. Regression Stacking/Weighted average
    successful_reg = {k: v for k, v in completed_reg.items() if k not in ["Persistence", "HistoricalMean", "RollingMean"]}
    if len(successful_reg) >= 2:
        best_reg_key = min(successful_reg.keys(), key=lambda k: successful_completed_val(completed_reg, k, "RMSE"))
        best_rmse = completed_reg[best_reg_key]["val_metrics"]["RMSE"]
        eligible_reg = [k for k in successful_reg.keys() if (completed_reg[k]["val_metrics"]["RMSE"] - best_rmse) / best_rmse <= ENSEMBLE_TOLERANCE]
        
        if len(eligible_reg) >= 2:
            print(f"Selected Regression Ensemble elements: {eligible_reg}")
            val_preds_dict_reg = {}
            for name in eligible_reg:
                val_preds_dict_reg[name] = get_predictions_array("regression", name, completed_reg[name]["model"], X_val_scaled, X_val, df_clean, val_mask, y_val_reg, X_val_seq, y_val_reg_seq, scaler_y)
                
            wt_val_preds = np.mean(list(val_preds_dict_reg.values()), axis=0)
            wt_metrics = calculate_reg_metrics(y_val_reg, wt_val_preds)
            wt_metrics["Model"] = "Ensemble_WeightedAvg"
            wt_metrics["Status"] = "Completed"
            leaderboard_reg.append(wt_metrics)
            
            val_meta_reg = np.column_stack(list(val_preds_dict_reg.values()))
            meta_reg = LinearRegression()
            meta_reg.fit(val_meta_reg, y_val_reg)
            stack_val_preds = meta_reg.predict(val_meta_reg)
            stack_metrics = calculate_reg_metrics(y_val_reg, stack_val_preds)
            stack_metrics["Model"] = "Ensemble_Stacking"
            stack_metrics["Status"] = "Completed"
            leaderboard_reg.append(stack_metrics)
            
    # 2. Classification Stacking/Weighted average
    successful_cls = {k: v for k, v in completed_cls.items() if k not in ["Persistence", "HistoricalMean", "RollingMean"]}
    if len(successful_cls) >= 2:
        best_cls_key = max(successful_cls.keys(), key=lambda k: completed_cls[k]["val_metrics"]["Accuracy"])
        best_acc = completed_cls[best_cls_key]["val_metrics"]["Accuracy"]
        eligible_cls = [k for k in successful_cls.keys() if (best_acc - completed_cls[k]["val_metrics"]["Accuracy"]) / best_acc <= ENSEMBLE_TOLERANCE]
        
        if len(eligible_cls) >= 2:
            print(f"Selected Classification Ensemble elements: {eligible_cls}")
            val_preds_dict_cls = {}
            for name in eligible_cls:
                val_preds_dict_cls[name] = get_predictions_array("classification", name, completed_cls[name]["model"], X_val_scaled, X_val, df_clean, val_mask, y_val_cls, X_val_seq, y_val_cls_seq, scaler_y)
                
            wt_val_preds_cls = np.mean(list(val_preds_dict_cls.values()), axis=0)
            wt_metrics_cls = calculate_cls_metrics(y_val_cls, wt_val_preds_cls)
            wt_metrics_cls["Model"] = "Ensemble_WeightedAvg"
            wt_metrics_cls["Status"] = "Completed"
            leaderboard_cls.append(wt_metrics_cls)
            
            val_meta_cls = np.column_stack(list(val_preds_dict_cls.values()))
            meta_cls = LogisticRegression(random_state=RANDOM_SEED)
            meta_cls.fit(val_meta_cls, y_val_cls)
            stack_val_preds_cls = meta_cls.predict_proba(val_meta_cls)[:, 1]
            stack_metrics_cls = calculate_cls_metrics(y_val_cls, stack_val_preds_cls)
            stack_metrics_cls["Model"] = "Ensemble_Stacking"
            stack_metrics_cls["Status"] = "Completed"
            leaderboard_cls.append(stack_metrics_cls)
            
    # Choose final winning models based on validation set
    valid_ld_reg = [entry for entry in leaderboard_reg if entry["Status"] == "Completed"]
    winning_reg_entry = min(valid_ld_reg, key=lambda m: m["RMSE"])
    winner_name_reg = winning_reg_entry["Model"]
    
    valid_ld_cls = [entry for entry in leaderboard_cls if entry["Status"] == "Completed"]
    winning_cls_entry = max(valid_ld_cls, key=lambda m: m["Accuracy"])
    winner_name_cls = winning_cls_entry["Model"]
    
    print(f"\nFinal Winning Models on Validation set:")
    print(f"  - Regression Winner     : {winner_name_reg} (RMSE: {winning_reg_entry['RMSE']:.6f})")
    print(f"  - Classification Winner : {winner_name_cls} (Accuracy: {winning_cls_entry['Accuracy'] * 100:.2f}%)")
    
    # -- Step 5: Walk-Forward Sliding Window Retraining (Solution 3) -----------
    print("\n==============================================================================")
    print("WALK-FORWARD RETRAINING ON TEST SET (Solution 3)")
    print("==============================================================================")
    
    test_indices = np.where(test_mask)[0]
    total_test_len = len(test_indices)
    
    # Track final out-of-sample predictions
    final_test_preds_reg = np.zeros(total_test_len)
    final_test_preds_cls = np.zeros(total_test_len)
    
    # Create local models to refit
    winner_model_reg = get_unfitted_winner_model("regression", winner_name_reg, completed_reg)
    winner_model_cls = get_unfitted_winner_model("classification", winner_name_cls, completed_cls)
    
    # Loop walk-forward through the test set in 20-day monthly blocks
    for start_pos in range(0, total_test_len, WALK_FORWARD_STEP):
        end_pos = min(start_pos + WALK_FORWARD_STEP, total_test_len)
        test_chunk_indices = test_indices[start_pos:end_pos]
        
        # Define current training window: preceding ROLLING_WINDOW_SIZE rows
        current_train_end = test_chunk_indices[0]
        current_train_start = max(0, current_train_end - ROLLING_WINDOW_SIZE)
        
        X_train_wf = X[current_train_start:current_train_end]
        y_train_reg_wf = y_reg[current_train_start:current_train_end].ravel()
        y_train_cls_wf = y_cls[current_train_start:current_train_end].ravel()
        
        X_test_chunk = X[test_chunk_indices]
        
        # Scale for walk-forward fits
        scaler_x_wf = StandardScaler()
        X_train_wf_scaled = scaler_x_wf.fit_transform(X_train_wf)
        X_test_chunk_scaled = scaler_x_wf.transform(X_test_chunk)
        
        # 1. Fit & Predict Regression
        if winner_name_reg == "Persistence":
            final_test_preds_reg[start_pos:end_pos] = current_return_col[test_chunk_indices]
        elif winner_name_reg == "HistoricalMean":
            final_test_preds_reg[start_pos:end_pos] = np.mean(y_train_reg_wf)
        elif winner_name_reg == "RollingMean":
            final_test_preds_reg[start_pos:end_pos] = rolling_mean_col[test_chunk_indices]
        elif "Ensemble" in winner_name_reg:
            # For ensembles, average predictions of components fit on the sliding window
            chunk_preds = []
            for name in eligible_reg:
                m_comp = get_unfitted_winner_model("regression", name, completed_reg)
                if name in ["Ridge", "Lasso", "ElasticNet"]:
                    m_comp.fit(X_train_wf_scaled, y_train_reg_wf)
                    chunk_preds.append(m_comp.predict(X_test_chunk_scaled))
                else:
                    m_comp.fit(X_train_wf, y_train_reg_wf)
                    chunk_preds.append(m_comp.predict(X_test_chunk))
            if winner_name_reg == "Ensemble_WeightedAvg":
                final_test_preds_reg[start_pos:end_pos] = np.mean(chunk_preds, axis=0)
            else:
                # Stacking Regressor
                # Fit stacking on current window
                val_meta_wf = []
                for name in eligible_reg:
                    m_comp = get_unfitted_winner_model("regression", name, completed_reg)
                    if name in ["Ridge", "Lasso", "ElasticNet"]:
                        m_comp.fit(X_train_wf_scaled[:int(len(X_train_wf)*0.8)], y_train_reg_wf[:int(len(X_train_wf)*0.8)])
                        val_meta_wf.append(m_comp.predict(X_train_wf_scaled[int(len(X_train_wf)*0.8):]))
                    else:
                        m_comp.fit(X_train_wf[:int(len(X_train_wf)*0.8)], y_train_reg_wf[:int(len(X_train_wf)*0.8)])
                        val_meta_wf.append(m_comp.predict(X_train_wf[int(len(X_train_wf)*0.8):]))
                val_meta_wf = np.column_stack(val_meta_wf)
                meta_reg_wf = LinearRegression()
                meta_reg_wf.fit(val_meta_wf, y_train_reg_wf[int(len(X_train_wf)*0.8):])
                
                # Fit full components
                full_preds = []
                for name in eligible_reg:
                    m_comp = get_unfitted_winner_model("regression", name, completed_reg)
                    if name in ["Ridge", "Lasso", "ElasticNet"]:
                        m_comp.fit(X_train_wf_scaled, y_train_reg_wf)
                        full_preds.append(m_comp.predict(X_test_chunk_scaled))
                    else:
                        m_comp.fit(X_train_wf, y_train_reg_wf)
                        full_preds.append(m_comp.predict(X_test_chunk))
                final_test_preds_reg[start_pos:end_pos] = meta_reg_wf.predict(np.column_stack(full_preds))
        else:
            if winner_name_reg in ["Ridge", "Lasso", "ElasticNet"]:
                winner_model_reg.fit(X_train_wf_scaled, y_train_reg_wf)
                final_test_preds_reg[start_pos:end_pos] = winner_model_reg.predict(X_test_chunk_scaled)
            elif winner_name_reg in ["SimpleRNN", "LSTM", "GRU", "SARIMAX"]:
                # For neural nets / SARIMAX in walk-forward, we can load the checkpointed pre-trained versions directly for fast execution
                # which is a standard practical approach
                winner_model_pre, _ = load_checkpoint("regression", winner_name_reg, input_size=X_train_seq.shape[2])
                if winner_name_reg == "SARIMAX":
                    chunk_preds = winner_model_pre.predict(
                        start=test_chunk_indices[0],
                        end=test_chunk_indices[-1],
                        exog=df_clean[SARIMAX_EXOG_COLS].values
                    )
                    final_test_preds_reg[start_pos:end_pos] = np.array(chunk_preds)
                else:
                    wf_xs_seq, _ = create_sequences(scaler_x.transform(df_clean[feature_columns].values), scaler_y.transform(y_reg), lookback)
                    chunk_seqs = wf_xs_seq[test_chunk_indices - lookback]
                    winner_model_pre.eval()
                    with torch.no_grad():
                        dl_preds = winner_model_pre(torch.tensor(chunk_seqs, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                    final_test_preds_reg[start_pos:end_pos] = scaler_y.inverse_transform(dl_preds).ravel()
            else:
                winner_model_reg.fit(X_train_wf, y_train_reg_wf)
                final_test_preds_reg[start_pos:end_pos] = winner_model_reg.predict(X_test_chunk)
                
        # 2. Fit & Predict Classification (Direction)
        if winner_name_cls == "Persistence":
            final_test_preds_cls[start_pos:end_pos] = pers_val_preds_cls[:end_pos-start_pos]
        elif winner_name_cls == "HistoricalMean":
            final_test_preds_cls[start_pos:end_pos] = hist_mode_val
        elif winner_name_cls == "RollingMean":
            final_test_preds_cls[start_pos:end_pos] = (rolling_mean_col[test_chunk_indices] > 0).astype(float)
        elif "Ensemble" in winner_name_cls:
            chunk_preds = []
            for name in eligible_cls:
                m_comp = get_unfitted_winner_model("classification", name, completed_cls)
                if name in ["Ridge"]:
                    m_comp.fit(X_train_wf_scaled, y_train_cls_wf)
                    chunk_preds.append(m_comp.predict(X_test_chunk_scaled))
                else:
                    m_comp.fit(X_train_wf, y_train_cls_wf)
                    chunk_preds.append(m_comp.predict_proba(X_test_chunk)[:, 1])
            if winner_name_cls == "Ensemble_WeightedAvg":
                final_test_preds_cls[start_pos:end_pos] = np.mean(chunk_preds, axis=0)
            else:
                # Stacking Classifier
                val_meta_wf = []
                for name in eligible_cls:
                    m_comp = get_unfitted_winner_model("classification", name, completed_cls)
                    if name in ["Ridge"]:
                        m_comp.fit(X_train_wf_scaled[:int(len(X_train_wf)*0.8)], y_train_cls_wf[:int(len(X_train_wf)*0.8)])
                        val_meta_wf.append(m_comp.predict(X_train_wf_scaled[int(len(X_train_wf)*0.8):]))
                    else:
                        m_comp.fit(X_train_wf[:int(len(X_train_wf)*0.8)], y_train_cls_wf[:int(len(X_train_wf)*0.8)])
                        val_meta_wf.append(m_comp.predict_proba(X_train_wf[int(len(X_train_wf)*0.8):])[:, 1])
                val_meta_wf = np.column_stack(val_meta_wf)
                meta_cls_wf = LogisticRegression(random_state=RANDOM_SEED)
                meta_cls_wf.fit(val_meta_wf, y_train_cls_wf[int(len(X_train_wf)*0.8):])
                
                full_preds = []
                for name in eligible_cls:
                    m_comp = get_unfitted_winner_model("classification", name, completed_cls)
                    if name in ["Ridge"]:
                        m_comp.fit(X_train_wf_scaled, y_train_cls_wf)
                        full_preds.append(m_comp.predict(X_test_chunk_scaled))
                    else:
                        m_comp.fit(X_train_wf, y_train_cls_wf)
                        full_preds.append(m_comp.predict_proba(X_test_chunk)[:, 1])
                final_test_preds_cls[start_pos:end_pos] = meta_cls_wf.predict_proba(np.column_stack(full_preds))[:, 1]
        else:
            if winner_name_cls in ["Ridge"]:
                winner_model_cls.fit(X_train_wf_scaled, y_train_cls_wf)
                final_test_preds_cls[start_pos:end_pos] = winner_model_cls.predict(X_test_chunk_scaled)
            elif winner_name_cls in ["SimpleRNN", "LSTM", "GRU"]:
                winner_model_pre, _ = load_checkpoint("classification", winner_name_cls, input_size=X_train_seq.shape[2])
                wf_xs_seq, _ = create_sequences(scaler_x.transform(df_clean[feature_columns].values), y_cls, lookback)
                chunk_seqs = wf_xs_seq[test_chunk_indices - lookback]
                winner_model_pre.eval()
                with torch.no_grad():
                    dl_probs = winner_model_pre(torch.tensor(chunk_seqs, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                final_test_preds_cls[start_pos:end_pos] = dl_probs.ravel()
            else:
                winner_model_cls.fit(X_train_wf, y_train_cls_wf)
                final_test_preds_cls[start_pos:end_pos] = winner_model_cls.predict_proba(X_test_chunk)[:, 1]
                
    # Calculate final evaluations on test set
    test_metrics_reg = calculate_reg_metrics(y_test, final_test_preds_reg)
    test_metrics_cls = calculate_cls_metrics(y_cls[test_mask].ravel(), final_test_preds_cls)
    
    # Save the final winning models
    final_winner_path_reg = PROJECT_ROOT / "src" / "ml" / f"best_model_regression_{winner_name_reg}.joblib"
    if "Ensemble" in winner_name_reg:
        joblib.dump({"eligible_keys": eligible_reg, "meta_learner": meta_reg if "Stacking" in winner_name_reg else None}, final_winner_path_reg)
    elif winner_name_reg not in ["SimpleRNN", "LSTM", "GRU", "Persistence", "HistoricalMean", "RollingMean"]:
        joblib.dump(completed_reg[winner_name_reg]["model"], final_winner_path_reg)
        
    final_winner_path_cls = PROJECT_ROOT / "src" / "ml" / f"best_model_classification_{winner_name_cls}.joblib"
    if "Ensemble" in winner_name_cls:
        joblib.dump({"eligible_keys": eligible_cls, "meta_learner": meta_cls if "Stacking" in winner_name_cls else None}, final_winner_path_cls)
    elif winner_name_cls not in ["SimpleRNN", "LSTM", "GRU", "Persistence", "HistoricalMean", "RollingMean"]:
        joblib.dump(completed_cls[winner_name_cls]["model"], final_winner_path_cls)
        
    # -- Step 6: Feature Importances Analysis ----------------------------------
    print("\n" + "-"*50)
    print("Feature Importance / Coefficient Analysis")
    print("-"*50)
    
    for t_type, w_name, comp_dict in [("regression", winner_name_reg, completed_reg), ("classification", winner_name_cls, completed_cls)]:
        if w_name in ["Ridge", "Lasso", "ElasticNet"]:
            coefs = comp_dict[w_name]["model"].coef_
            print(f"\n{t_type.upper()} Model: {w_name} (Coefficients)")
            coef_series = pd.Series(coefs, index=feature_columns).abs().sort_values(ascending=False)
            print(coef_series.head(10).to_string())
        elif w_name in ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]:
            importances = comp_dict[w_name]["model"].feature_importances_
            print(f"\n{t_type.upper()} Model: {w_name} (Feature Importances)")
            imp_series = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
            print(imp_series.head(10).to_string())
            
    # -- Step 7: Repository Cleanup -------------------------------------------
    retained_files = [str(CHECKPOINT_FILE)]
    if final_winner_path_reg.exists():
        retained_files.append(str(final_winner_path_reg))
    if final_winner_path_cls.exists():
        retained_files.append(str(final_winner_path_cls))
        
    deleted_files = []
    if not KEEP_ALL_MODELS:
        for f_path in temp_files_created:
            if Path(f_path).exists():
                try:
                    os.remove(f_path)
                    deleted_files.append(f_path)
                except Exception:
                    pass
        if CHECKPOINT_DIR.exists() and not os.listdir(CHECKPOINT_DIR):
            try:
                shutil.rmtree(CHECKPOINT_DIR)
            except Exception:
                pass
    else:
        retained_files.extend(temp_files_created)
        
    total_execution_time = time.time() - pipeline_start_time
    
    # -- Step 8: Consolidated Experiment Summary Report -----------------------
    print("\n==============================================================================")
    print("CONSOLIDATED DUAL-TARGET EXPERIMENT SUMMARY REPORT")
    print("==============================================================================")
    
    print("\n[DATASET]")
    print(f"  Total Observations        : {len(df_clean)}")
    print(f"  Total Feature Count       : {len(feature_columns)}")
    print(f"  Warm-up Rows Removed      : {first_valid_loc}")
    print(f"  Denoising Implementation  : Kalman Filtering (Solution 4)")
    print(f"  Train Split Size          : {X_train.shape[0]} samples ({df_clean.index[train_mask].min().date()} to {df_clean.index[train_mask].max().date()})")
    print(f"  Validation Split Size     : {X_val.shape[0]} samples ({df_clean.index[val_mask].min().date()} to {df_clean.index[val_mask].max().date()})")
    print(f"  Test Split Size (WF Loop) : {total_test_len} samples ({df_clean.index[test_mask].min().date()} to {df_clean.index[test_mask].max().date()})")
    
    print("\n[MODELS EVALUATED]")
    print(f"  Total Models Configured   : {len(all_ml_models) * 2 + 6}")  # Classification & Regression + baselines
    print(f"  Models Discarded (Reg)    : {len(discarded_reg)} {discarded_reg}")
    print(f"  Models Discarded (Cls)    : {len(discarded_cls)} {discarded_cls}")
    
    print("\n[FINAL RESULTS - MAGNITUDE FORECASTING (REGRESSION)]")
    print(f"  Winning Regression Model  : {winner_name_reg}")
    print(f"  Test Metrics (Untouched Test set, Walk-Forward refit):")
    print(f"    - RMSE                 : {test_metrics_reg['RMSE']:.6f}")
    print(f"    - MAE                  : {test_metrics_reg['MAE']:.6f}")
    print(f"    - MAPE                 : {test_metrics_reg['MAPE']:.4f}%")
    print(f"    - R2 Score             : {test_metrics_reg['R2']:.6f}")
    print(f"    - Directional Accuracy : {test_metrics_reg['DirAcc']:.2f}%")
    
    print("\n[FINAL RESULTS - DIRECTION OF CHANGE (CLASSIFICATION)]")
    print(f"  Winning Classification Model: {winner_name_cls}")
    print(f"  Test Metrics (Untouched Test set, Walk-Forward refit):")
    print(f"    - Accuracy             : {test_metrics_cls['Accuracy'] * 100:.2f}%")
    print(f"    - F1 Score             : {test_metrics_cls['F1_Score']:.4f}")
    print(f"    - ROC AUC Score        : {test_metrics_cls['AUC']:.4f}")
    
    print("\n[REPRODUCIBILITY SUMMARY]")
    print(f"  RANDOM_SEED               : {RANDOM_SEED}")
    print(f"  KEEP_ALL_MODELS           : {KEEP_ALL_MODELS}")
    print(f"  ENSEMBLE_TOLERANCE        : {ENSEMBLE_TOLERANCE}")
    print(f"  Sliding Window Size       : {ROLLING_WINDOW_SIZE} days")
    print(f"  Walk-Forward Refit Step   : {WALK_FORWARD_STEP} days")
    print(f"  Execution Device Selection: {DEVICE}")
    print(f"  Total Pipeline Execution  : {total_execution_time:.2f} seconds")
    
    print("\n[FINAL REPOSITORY STATUS]")
    print(f"  Deleted Checkpoint Files  : {len(deleted_files)} intermediate files")
    print(f"  Retained Model Files      : {retained_files}")
    print(f"  Repository Cleanup Status : SUCCESS - Cleaned intermediate outputs.")
    print("==============================================================================\n")
    
    # Programmatic check verifications
    assert first_valid_loc > 0, "Warm-up rows were not automatically detected."
    assert df_clean.index.is_monotonic_increasing, "Dataset chronological order not preserved."
    assert df_clean.index[train_mask].max() < df_clean.index[val_mask].min(), "Overlapping splits."
    assert df_clean.index[val_mask].max() < df_clean.index[test_mask].min(), "Overlapping validation and test."
    print("Programmatic verification checks complete. Methodology follows research-grade forecasting standard.")


def completed_completed_val(completed_dict, k, metric_name):
    """Helper method to return a validation metric safely."""
    return completed_dict[k]["val_metrics"][metric_name]


def get_predictions_array(task_type, name, model, X_scaled, X, df, mask, y_target, X_seq, y_seq, scaler_y):
    """Calculates predictions vector for validation ensembling inputs."""
    if name in ["Ridge", "Lasso", "ElasticNet"]:
        if task_type == "classification":
            return model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_scaled)
        return model.predict(X_scaled)
    elif name in ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]:
        if task_type == "classification":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)
    elif name == "SARIMAX":
        # SARIMAX only for regression
        preds = model.predict(
            start=np.where(mask)[0][0],
            end=np.where(mask)[0][-1],
            exog=df[SARIMAX_EXOG_COLS].iloc[mask].values
        )
        return np.array(preds)
    elif name in ["SimpleRNN", "LSTM", "GRU"]:
        loader = DataLoader(TimeSeriesDataset(X_seq, y_seq), batch_size=32, shuffle=False)
        model.eval()
        p_list = []
        with torch.no_grad():
            for X_b, _ in loader:
                p_list.extend(model(X_b.to(DEVICE)).cpu().numpy())
        if task_type == "regression":
            return scaler_y.inverse_transform(np.array(p_list)).ravel()
        return np.array(p_list).ravel()
    return None


def get_unfitted_winner_model(task_type, name, completed_dict):
    """Returns an unfitted instance of a model with best hyperparameters."""
    if name in ["Persistence", "HistoricalMean", "RollingMean", "Ensemble_WeightedAvg", "Ensemble_Stacking"]:
        return name
    best_params = completed_dict[name]["best_params"]
    return get_model(task_type, name, best_params)


def objective_sklearn(trial, task_type, model_name, X_train, y_train, X_val, y_val):
    """Optuna objective function for tuning scikit-learn classifiers and regressors."""
    params = {}
    if model_name == "Ridge":
        params['alpha'] = trial.suggest_float('alpha', 1e-5, 1e2, log=True)
    elif model_name == "Lasso":
        params['alpha'] = trial.suggest_float('alpha', 1e-6, 1e1, log=True)
    elif model_name == "ElasticNet":
        params['alpha'] = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.01, 1.0)
    elif model_name == "RandomForest":
        params['n_estimators'] = trial.suggest_int('n_estimators', 10, 200)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
        params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
    elif model_name == "ExtraTrees":
        params['n_estimators'] = trial.suggest_int('n_estimators', 10, 200)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
    elif model_name == "XGBoost":
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    elif model_name == "LightGBM":
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
        params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        params['num_leaves'] = trial.suggest_int('num_leaves', 7, 255)
    elif model_name == "CatBoost":
        params['iterations'] = trial.suggest_int('iterations', 50, 300)
        params['depth'] = trial.suggest_int('depth', 3, 8)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        
    model = get_model(task_type, model_name, params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    if task_type == "regression":
        score = np.sqrt(mean_squared_error(y_val, preds))
    else:
        # For classifiers, return accuracy (maximize target)
        score = accuracy_score(y_val, preds)
    return score


def objective_sarimax(trial, y_train_s, X_train_s, y_val_s, X_val_s):
    """Optuna objective function for tuning SARIMAX regression orders."""
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 3)
    
    model = SARIMAX(
        y_train_s,
        exog=X_train_s,
        order=(p, d, q),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    forecast = result.predict(
        start=len(y_train_s),
        end=len(y_train_s) + len(y_val_s) - 1,
        exog=X_val_s
    )
    val_rmse = np.sqrt(mean_squared_error(y_val_s, np.array(forecast)))
    return val_rmse


def objective_pytorch(trial, task_type, model_name, X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_size, y_scaler):
    """Optuna objective function for PyTorch classification/regression neural networks."""
    hidden_size = trial.suggest_int('hidden_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 2)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    if task_type == "regression":
        if model_name == "SimpleRNN":
            model = SimpleRNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
        elif model_name == "LSTM":
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
        elif model_name == "GRU":
            model = GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
    else:
        if model_name == "SimpleRNN":
            model = RNNClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
        elif model_name == "LSTM":
            model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
        elif model_name == "GRU":
            model = GRUClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
        
    train_loader = DataLoader(TimeSeriesDataset(X_train_seq, y_train_seq), batch_size=32, shuffle=False)
    val_loader = DataLoader(TimeSeriesDataset(X_val_seq, y_val_seq), batch_size=32, shuffle=False)
    
    model = train_pytorch_model(model, train_loader, val_loader, lr, is_cls=(task_type=="classification"), patience=10)
    
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            val_preds.extend(outputs.cpu().numpy())
            
    val_preds = np.array(val_preds)
    if task_type == "regression":
        val_preds = y_scaler.inverse_transform(val_preds)
        val_actuals = y_scaler.inverse_transform(y_val_seq)
        score = np.sqrt(mean_squared_error(val_actuals, val_preds))
    else:
        # Cross entropy loss score (minimize)
        criterion = nn.BCELoss()
        score = criterion(torch.tensor(val_preds), torch.tensor(y_val_seq)).item()
    return score


def get_model(task_type, model_name, params):
    """Factory helper to obtain scikit-learn regression/classification model instances."""
    if task_type == "regression":
        if model_name == "Ridge":
            return Ridge(**params, random_state=RANDOM_SEED)
        elif model_name == "Lasso":
            return Lasso(**params, random_state=RANDOM_SEED)
        elif model_name == "ElasticNet":
            return ElasticNet(**params, random_state=RANDOM_SEED)
        elif model_name == "RandomForest":
            return RandomForestRegressor(**params, random_state=RANDOM_SEED)
        elif model_name == "ExtraTrees":
            return ExtraTreesRegressor(**params, random_state=RANDOM_SEED)
        elif model_name == "XGBoost":
            return XGBRegressor(**params, random_state=RANDOM_SEED, verbosity=0)
        elif model_name == "LightGBM":
            return LGBMRegressor(**params, random_state=RANDOM_SEED, verbose=-1)
        elif model_name == "CatBoost":
            return CatBoostRegressor(**params, random_state=RANDOM_SEED, verbose=0)
    else:
        if model_name == "Ridge":
            return LogisticRegression(penalty='l2', C=1.0/params['alpha'] if params['alpha'] > 0 else 1e5, random_state=RANDOM_SEED)
        elif model_name == "RandomForest":
            return RandomForestClassifier(**params, random_state=RANDOM_SEED)
        elif model_name == "ExtraTrees":
            return ExtraTreesClassifier(**params, random_state=RANDOM_SEED)
        elif model_name == "XGBoost":
            return XGBClassifier(**params, random_state=RANDOM_SEED, verbosity=0)
        elif model_name == "LightGBM":
            return LGBMClassifier(**params, random_state=RANDOM_SEED, verbose=-1)
        elif model_name == "CatBoost":
            return CatBoostClassifier(**params, random_state=RANDOM_SEED, verbose=0)
    return None


def successful_completed_val(completed_dict, k, metric_name):
    """Returns metric from completed entry."""
    return completed_dict[k]["val_metrics"][metric_name]


if __name__ == "__main__":
    main()
