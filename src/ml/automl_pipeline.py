"""
automl_pipeline.py

Comprehensive Research-Oriented Forecasting Pipeline
Strictly Maximizes Directional Accuracy using Statistical, ML, and DL models.
"""

import os
import sys
import time
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS
try:
    from prophet import Prophet
except ImportError:
    print("WARNING: prophet is not installed. Please install it via `pip install prophet`.")
    Prophet = None

# Import existing utilities
from model_pipeline import denoise_features_kalman
from dl_pipeline import (
    SimpleRNNModel, LSTMModel, GRUModel, 
    Seq2SeqModel, TCNModel, NBeatsModel,
    TimeSeriesDataset, create_sequences, 
    get_optimizer, get_scheduler
)

# Disable warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import (
    PROJECT_ROOT, DATA_PATH, CHECKPOINT_DIR, FINAL_OUTPUT_DIR,
    TARGET_REG, RANDOM_SEED, MAX_OPTUNA_TRIALS, CV_SPLITS,
    DEVICE, set_seeds, print_gpu_info
)

# Set seeds immediately
set_seeds()

# ==============================================================================
# METRICS
# ==============================================================================
def find_optimal_threshold(actual, predicted):
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    thresholds = np.linspace(np.min(predicted), np.max(predicted), 100)
    best_thresh = 0.0
    best_dir_acc = 0.0
    
    actual_dir = np.sign(actual)
    actual_dir[actual_dir == 0] = 1
    
    for th in thresholds:
        pred_dir = np.where(predicted > th, 1, -1)
        acc = (actual_dir == pred_dir).mean() * 100 if len(actual_dir) > 0 else 50.0
        if acc > best_dir_acc:
            best_dir_acc = acc
            best_thresh = th
            
    return best_thresh

def calculate_metrics(actual, predicted, threshold=0.0):
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted)

    # Directional Accuracy
    actual_dir = np.sign(actual)
    actual_dir[actual_dir == 0] = 1
    pred_dir = np.where(predicted > threshold, 1, -1)
    dir_acc = (actual_dir == pred_dir).mean() * 100 if len(actual_dir) > 0 else 50.0

    return {"DirAcc": dir_acc, "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}

# ==============================================================================
# NEW DL ARCHITECTURES
# ==============================================================================
class PatchTSTModel(nn.Module):
    def __init__(self, input_size, seq_len, patch_len=5, stride=5, d_model=32, n_heads=4, e_layers=2, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.value_embedding = nn.Linear(input_size * patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Linear(d_model * self.num_patches, 1)

    def forward(self, x):
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end > x.shape[1]: break
            patches.append(x[:, start:end, :].reshape(x.shape[0], -1))
        if not patches:
            patches.append(x[:, -self.patch_len:, :].reshape(x.shape[0], -1) if x.shape[1] >= self.patch_len else torch.zeros(x.shape[0], x.shape[2]*self.patch_len).to(x.device))
            self.num_patches = 1
        x = torch.stack(patches, dim=1)
        x = self.value_embedding(x)
        x = x + self.position_embedding[:, :x.shape[1], :]
        x = self.transformer_encoder(x)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)

class NHitsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, expansion_coef, pool_kernel):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        pool_out_size = max(1, input_size // pool_kernel)
        self.fc_stack = nn.Sequential(
            nn.Linear(pool_out_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.theta_b = nn.Linear(hidden_size, expansion_coef, bias=False)
        self.theta_f = nn.Linear(hidden_size, expansion_coef, bias=False)
        self.backcast_basis = nn.Linear(expansion_coef, input_size, bias=False)
        self.forecast_basis = nn.Linear(expansion_coef, 1, bias=False)

    def forward(self, x):
        pooled_x = self.pool(x.unsqueeze(1)).squeeze(1)
        h = self.fc_stack(pooled_x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        return backcast, forecast

class NHitsModel(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=128, expansion_coef=32):
        super().__init__()
        flat_size = seq_len * input_size
        self.blocks = nn.ModuleList([
            NHitsBlock(flat_size, hidden_size, expansion_coef, pool_kernel=4),
            NHitsBlock(flat_size, hidden_size, expansion_coef, pool_kernel=2),
            NHitsBlock(flat_size, hidden_size, expansion_coef, pool_kernel=1)
        ])
    def forward(self, x):
        batch = x.size(0)
        residual = x.reshape(batch, -1)
        forecast = torch.zeros(batch, 1, device=x.device)
        for block in self.blocks:
            backcast, block_fc = block(residual)
            residual = residual - backcast
            forecast = forecast + block_fc
        return forecast

class SimplifiedTFTModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        return self.fc(attn_out[:, -1, :])

# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
class ForecastingPipeline:
    def __init__(self):
        self.leaderboard = []
        self.best_model_name = None
        self.best_diracc = -float("inf")
        self.best_metrics = None
        
    def generate_candidate_features(self, df):
        # ONLY keep the raw return and the target. Discard all pre-engineered commodity features.
        df_aug = df[["usd_zar_logret", TARGET_REG]].copy()
        base = df_aug["usd_zar_logret"]
        
        # Lags
        for lag in range(1, 10):
            df_aug[f"dir_lag_{lag}"] = np.sign(base).shift(lag)
            df_aug[f"logret_lag_{lag}"] = base.shift(lag)
            
        # Rate of Change
        for span in [3, 5, 10]:
            df_aug[f"roc_{span}"] = (base / (base.shift(span) + 1e-8) - 1).shift(1)
            
        # Momentum
        for lag in [3, 5, 10]:
            df_aug[f"mom_{lag}"] = (base - base.shift(lag)).shift(1)
            
        # EMA
        for span in [5, 10, 20]:
            df_aug[f"ema_{span}"] = base.ewm(span=span, adjust=False).mean().shift(1)
            
        # Volatility (Rolling Std)
        for window in [5, 10, 21]:
            df_aug[f"vol_{window}"] = base.rolling(window=window).std().shift(1)
            
        return df_aug.dropna()

    def prepare_data(self):
        print(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        nan_rows = df.isna().any(axis=1)
        first_valid = np.where(~nan_rows)[0][0]
        df = df.iloc[first_valid:-1].ffill().bfill()
        
        df = self.generate_candidate_features(df)
        
        # Denoise all features except target
        orig_feat_cols = [c for c in df.columns if c != TARGET_REG]
        df = denoise_features_kalman(df, orig_feat_cols)
        
        self.df = df.copy()
        
        self.train_mask = self.df.index <= pd.Timestamp("2021-12-31")
        self.val_mask   = ((self.df.index >= pd.Timestamp("2022-01-01")) & (self.df.index <= pd.Timestamp("2023-12-31")))
        self.test_mask  = self.df.index >= pd.Timestamp("2024-01-01")
        
        self.df_train = self.df[self.train_mask]
        self.df_val   = self.df[self.val_mask]
        self.df_test  = self.df[self.test_mask]

        # Scaler fit only on Train (used for DL models)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.df_train.drop(columns=[TARGET_REG]))
        self.X_val_scaled   = self.scaler.transform(self.df_val.drop(columns=[TARGET_REG]))
        self.X_test_scaled  = self.scaler.transform(self.df_test.drop(columns=[TARGET_REG]))
        
        self.y_train = self.df_train[TARGET_REG].values
        self.y_val   = self.df_val[TARGET_REG].values
        self.y_test  = self.df_test[TARGET_REG].values
        print("Data preparation complete.")

    def update_leaderboard(self, model_name, metrics, best_params, train_time, threshold=0.0):
        self.leaderboard.append({
            "Model": model_name,
            "DirAcc": metrics["DirAcc"],
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "TrainTime": train_time,
            "Threshold": threshold,
            "BestParams": str(best_params)
        })
        self.leaderboard.sort(key=lambda x: (x["DirAcc"], x["R2"], -x["RMSE"]), reverse=True)
        if self.leaderboard[0]["Model"] == model_name:
            self.best_model_name = model_name
            self.best_diracc = metrics["DirAcc"]
            self.best_metrics = metrics
            print(f"  >>> NEW BEST MODEL! {model_name} (DirAcc: {metrics['DirAcc']:.2f}% | Thresh: {threshold:.4f})")

    def train_ml_models(self):
        models_to_test = {
            "RandomForest": RandomForestRegressor,
            "ExtraTrees": ExtraTreesRegressor,
            "XGBoost": XGBRegressor,
            "LightGBM": LGBMRegressor,
            "CatBoost": CatBoostRegressor
        }
        
        for name, ModelClass in models_to_test.items():
            print(f"\n--- Optimizing ML Model: {name} ---")
            start_time = time.time()
            
            def ml_objective(trial):
                X_tr_full = self.df_train.drop(columns=[TARGET_REG]).values
                y_tr_full = self.y_train
                
                if name in ["RandomForest", "ExtraTrees"]:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                        "random_state": RANDOM_SEED,
                        "n_jobs": -1
                    }
                elif name == "XGBoost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "random_state": RANDOM_SEED,
                        "n_jobs": -1
                    }
                elif name == "LightGBM":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                        "random_state": RANDOM_SEED,
                        "n_jobs": -1,
                        "verbose": -1
                    }
                elif name == "CatBoost":
                    params = {
                        "iterations": trial.suggest_int("iterations", 50, 300, step=50),
                        "depth": trial.suggest_int("depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                        "random_seed": RANDOM_SEED,
                        "verbose": 0
                    }
                
                tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_tr_full):
                    X_t, X_v = X_tr_full[train_idx], X_tr_full[val_idx]
                    y_t, y_v = y_tr_full[train_idx], y_tr_full[val_idx]
                    
                    model = ModelClass(**params)
                    model.fit(X_t, y_t)
                    preds = model.predict(X_v)
                    
                    th = find_optimal_threshold(y_v, preds)
                    metrics = calculate_metrics(y_v, preds, threshold=th)
                    cv_scores.append(metrics["DirAcc"])
                    
                return -np.mean(cv_scores)

            study = optuna.create_study(direction="minimize")
            study.optimize(ml_objective, n_trials=MAX_OPTUNA_TRIALS)
            
            p = study.best_params
            X_tr = self.df_train.drop(columns=[TARGET_REG]).values
            X_v  = self.df_val.drop(columns=[TARGET_REG]).values
            
            if name in ["RandomForest", "ExtraTrees"]: p.update({"random_state": RANDOM_SEED, "n_jobs": -1})
            elif name == "XGBoost": p.update({"random_state": RANDOM_SEED, "n_jobs": -1})
            elif name == "LightGBM": p.update({"random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1})
            elif name == "CatBoost": p.update({"random_seed": RANDOM_SEED, "verbose": 0})
                
            best_model = ModelClass(**p)
            best_model.fit(X_tr, self.y_train)
            val_preds = best_model.predict(X_v)
            th = find_optimal_threshold(self.y_val, val_preds)
            val_metrics = calculate_metrics(self.y_val, val_preds, threshold=th)
            
            joblib.dump(best_model, CHECKPOINT_DIR / f"automl_{name}.joblib")
            self.update_leaderboard(name, val_metrics, p, time.time() - start_time, threshold=th)

    def train_dl_models(self):
        dl_models = {
            "SimpleRNN": SimpleRNNModel,
            "GRU": GRUModel,
            "LSTM": LSTMModel,
            "Seq2Seq": Seq2SeqModel,
            "TCN": TCNModel,
            "NBEATS": NBeatsModel,
            "NHITS": NHitsModel,
            "PatchTST": PatchTSTModel,
            "TFT": SimplifiedTFTModel
        }
        
        # No global sequence length evaluation; optimize per model via Optuna.
        criterion = nn.MSELoss()
        
        for name, ModelClass in dl_models.items():
            print(f"\n--- Optimizing DL Model: {name} ---")
            start_time = time.time()
            
            def dl_objective(trial):
                hidden = trial.suggest_int("hidden_size", 32, 256, step=32)
                dropout = trial.suggest_float("dropout", 0.0, 0.4)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                seq_len = trial.suggest_categorical("seq_len", [5, 10, 20, 30, 60, 90, 120])
                
                X_tr, y_tr = create_sequences(self.X_train_scaled, self.y_train, seq_len)
                X_v, y_v   = create_sequences(self.X_val_scaled, self.y_val, seq_len)
                train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=32, shuffle=False)
                val_loader   = DataLoader(TimeSeriesDataset(X_v, y_v), batch_size=32, shuffle=False)
                
                input_size = self.X_train_scaled.shape[1]
                
                if name == "Seq2Seq":
                    model = ModelClass(input_size, hidden_size=hidden, enc_layers=1, dec_layers=1, dropout=dropout).to(DEVICE)
                elif name == "TCN":
                    model = ModelClass(input_size, num_channels=hidden, kernel_size=3, num_blocks=2, dropout=dropout).to(DEVICE)
                elif name in ["NBEATS"]:
                    model = ModelClass(input_size, hidden_size=hidden).to(DEVICE)
                elif name in ["NHITS"]:
                    model = ModelClass(input_size, seq_len=seq_len, hidden_size=hidden).to(DEVICE)
                elif name == "PatchTST":
                    model = ModelClass(input_size, seq_len=seq_len, d_model=hidden, dropout=dropout).to(DEVICE)
                elif name == "TFT":
                    model = ModelClass(input_size, hidden_size=hidden, dropout=dropout).to(DEVICE)
                else:
                    layers = trial.suggest_int("num_layers", 1, 3)
                    model = ModelClass(input_size, hidden_size=hidden, num_layers=layers, dropout=dropout).to(DEVICE)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                best_dir_acc = 0
                no_improve = 0
                
                for epoch in range(15):
                    model.train()
                    for bx, by in train_loader:
                        bx, by = bx.to(DEVICE), by.unsqueeze(1).to(DEVICE)
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        
                    model.eval()
                    all_preds = []
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx = bx.to(DEVICE)
                            out = model(bx)
                            all_preds.extend(out.cpu().numpy())
                            
                    th = find_optimal_threshold(y_v, np.array(all_preds))
                    mets = calculate_metrics(y_v, np.array(all_preds), threshold=th)
                    
                    if mets["DirAcc"] > best_dir_acc:
                        best_dir_acc = mets["DirAcc"]
                        no_improve = 0
                    else:
                        no_improve += 1
                        
                    if no_improve >= 3:
                        break
                        
                return -best_dir_acc

            study = optuna.create_study(direction="minimize")
            study.optimize(dl_objective, n_trials=MAX_OPTUNA_TRIALS)
            
            p = study.best_params
            input_size = self.X_train_scaled.shape[1]
            best_seq_len = p["seq_len"]
            
            # Recreate datasets for the winning sequence length
            X_tr, y_tr = create_sequences(self.X_train_scaled, self.y_train, best_seq_len)
            X_v, y_v   = create_sequences(self.X_val_scaled, self.y_val, best_seq_len)
            train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=32, shuffle=False)
            val_loader   = DataLoader(TimeSeriesDataset(X_v, y_v), batch_size=32, shuffle=False)
            
            if name == "Seq2Seq":
                model = ModelClass(input_size, hidden_size=p["hidden_size"], enc_layers=1, dec_layers=1, dropout=p["dropout"]).to(DEVICE)
            elif name == "TCN":
                model = ModelClass(input_size, num_channels=p["hidden_size"], kernel_size=3, num_blocks=2, dropout=p["dropout"]).to(DEVICE)
            elif name in ["NBEATS"]:
                model = ModelClass(input_size, hidden_size=p["hidden_size"]).to(DEVICE)
            elif name in ["NHITS"]:
                model = ModelClass(input_size, seq_len=best_seq_len, hidden_size=p["hidden_size"]).to(DEVICE)
            elif name == "PatchTST":
                model = ModelClass(input_size, seq_len=best_seq_len, d_model=p["hidden_size"], dropout=p["dropout"]).to(DEVICE)
            elif name == "TFT":
                model = ModelClass(input_size, hidden_size=p["hidden_size"], dropout=p["dropout"]).to(DEVICE)
            else:
                model = ModelClass(input_size, hidden_size=p["hidden_size"], num_layers=p["num_layers"], dropout=p["dropout"]).to(DEVICE)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
            for epoch in range(30):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.unsqueeze(1).to(DEVICE)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            all_preds = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(DEVICE)
                    out = model(bx)
                    all_preds.extend(out.cpu().numpy())
                    
            th = find_optimal_threshold(y_v, np.array(all_preds))
            val_metrics = calculate_metrics(y_v, np.array(all_preds), threshold=th)
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"automl_{name}.pt")
            
            self.update_leaderboard(name, val_metrics, p, time.time() - start_time, threshold=th)

    def train_statistical_models(self):
        print("\n--- Optimizing Statistical Models ---")
        
        # SARIMAX
        start_time = time.time()
        print("Training SARIMAX...")
        try:
            model_sarimax = SARIMAX(self.y_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            fit_sarimax = model_sarimax.fit(disp=False)
            preds_sarimax = fit_sarimax.forecast(steps=len(self.y_val))
            th = find_optimal_threshold(self.y_val, preds_sarimax)
            metrics = calculate_metrics(self.y_val, preds_sarimax, threshold=th)
            joblib.dump(fit_sarimax, CHECKPOINT_DIR / "automl_SARIMAX.joblib")
            self.update_leaderboard("SARIMAX", metrics, {"order": (1,1,1)}, time.time() - start_time, threshold=th)
        except Exception as e:
            print(f"SARIMAX failed: {e}")
            
        # ETS
        start_time = time.time()
        print("Training ETS...")
        try:
            model_ets = ETS(self.y_train, trend='add', seasonal=None)
            fit_ets = model_ets.fit()
            preds_ets = fit_ets.forecast(steps=len(self.y_val))
            th = find_optimal_threshold(self.y_val, preds_ets)
            metrics = calculate_metrics(self.y_val, preds_ets, threshold=th)
            joblib.dump(fit_ets, CHECKPOINT_DIR / "automl_ETS.joblib")
            self.update_leaderboard("ETS", metrics, {"trend": "add"}, time.time() - start_time, threshold=th)
        except Exception as e:
            print(f"ETS failed: {e}")
            
        # Prophet
        if Prophet is not None:
            start_time = time.time()
            print("Training Prophet...")
            try:
                df_prophet = pd.DataFrame({"ds": self.df_train.index, "y": self.y_train})
                m = Prophet()
                m.fit(df_prophet)
                future = pd.DataFrame({"ds": self.df_val.index})
                forecast = m.predict(future)
                preds_prophet = forecast['yhat'].values
                th = find_optimal_threshold(self.y_val, preds_prophet)
                metrics = calculate_metrics(self.y_val, preds_prophet, threshold=th)
                joblib.dump(m, CHECKPOINT_DIR / "automl_Prophet.joblib")
                self.update_leaderboard("Prophet", metrics, {}, time.time() - start_time, threshold=th)
            except Exception as e:
                print(f"Prophet failed: {e}")
        else:
            print("Prophet skipped due to missing dependency.")

    def print_leaderboard(self):
        print("\n" + "="*80)
        print("  AUTOML PIPELINE LEADERBOARD")
        print("="*80)
        print(f"{'Rank':<5} {'Model':<15} {'DirAcc':<10} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'Thresh':<10}")
        print("-" * 80)
        for i, row in enumerate(self.leaderboard):
            print(f"{i+1:<5} {row['Model']:<15} {row['DirAcc']:<9.2f}% {row['R2']:<9.4f} {row['RMSE']:<9.6f} {row['MAE']:<9.6f} {row['Threshold']:<9.4f}")
        print("="*80)

    def cleanup_checkpoints(self):
        print("\n--- Cleaning up checkpoints (Keeping Top 3) ---")
        top_models = [row["Model"] for row in self.leaderboard[:3]]
        
        for file in CHECKPOINT_DIR.iterdir():
            if file.name.startswith("automl_") and (file.name.endswith(".pt") or file.name.endswith(".joblib")):
                model_name = file.stem.replace("automl_", "")
                if model_name not in top_models:
                    file.unlink()
                    print(f"  [-] Deleted non-top checkpoint: {file.name}")
                    
        # Copy the absolute best model to final_output
        import shutil
        if self.best_model_name:
            best_ext = ".pt" if self.best_model_name in ["SimpleRNN", "GRU", "LSTM", "Seq2Seq", "TCN", "NBEATS", "NHITS", "PatchTST", "TFT"] else ".joblib"
            best_file = CHECKPOINT_DIR / f"automl_{self.best_model_name}{best_ext}"
            if best_file.exists():
                shutil.copy2(best_file, FINAL_OUTPUT_DIR / best_file.name)
                print(f"  [+] Copied WINNER ({best_file.name}) to final_output/")
                
        # Save Leaderboard to final output
        pd.DataFrame(self.leaderboard).to_csv(FINAL_OUTPUT_DIR / "leaderboard.csv", index=False)
        print("  [+] Saved leaderboard.csv to final_output/")
        print("Cleanup complete.")

    def run(self):
        print("="*80)
        print("  STARTING AUTOML FORECASTING PIPELINE (Target: Directional Accuracy)")
        print("="*80)
        
        self.prepare_data()
        
        self.train_statistical_models()
        self.train_ml_models()
        self.train_dl_models()
        
        self.print_leaderboard()
        self.cleanup_checkpoints()
        
        print(f"\nFinal Best Model: {self.best_model_name} with DirAcc: {self.best_diracc:.2f}%")
        print("Pipeline execution completed successfully.")

def validate_startup():
    print_gpu_info()
    print(f"Data Path: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}!")
    print(f"Checkpoints Dir: {CHECKPOINT_DIR}")
    print(f"Final Output Dir: {FINAL_OUTPUT_DIR}")
    print("Startup validation successful.\n")

if __name__ == "__main__":
    validate_startup()
    pipeline = ForecastingPipeline()
    pipeline.run()

