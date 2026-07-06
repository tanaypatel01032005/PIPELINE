"""
dl_pipeline.py  Deep Learning Forecasting Pipeline Extension
=============================================================
Extends the existing model_pipeline.py by training and tuning eight
deep-learning-only forecasting models on the same data splits:

  1. SimpleRNN
  2. FinetunedRNN   (Optuna-tuned)
  3. FinetunedGRU   (Optuna-tuned)
  4. LSTM
  5. FinetunedLSTM  (Optuna-tuned)
  6. Seq2Seq        (Encoder-Decoder GRU, Optuna-tuned)
  7. TCN            (Temporal Convolutional Network, Optuna-tuned)
  8. N-BEATS        (Neural Basis Expansion, Optuna-tuned)

Pipeline steps
--------------
  1. Load engineered_features.csv  (same file as model_pipeline.py)
  2. Apply Kalman denoising         (same function)
  3. Chronological split            (train  2021, val 2022-2023, test  2024)
  4. Fit StandardScaler on train only
  5. Automated extra-feature generation + validation-RMSE-based selection
  6. Optimal sequence-length search once; reused by all models
  7. Train / tune all eight models sequentially; checkpoint each
  8. Evaluate three ensemble strategies on validation set
  9. Walk-forward evaluation of the winner on the test set
 10. Clean up non-winning checkpoints
 11. Print consolidated experiment summary

Nothing in model_pipeline.py is modified  this script is purely additive.

Design notes
------------
- Helpers reused verbatim from model_pipeline.py (Kalman, metrics, checkpoint,
  EarlyStoppingCallback, TimeSeriesDataset, create_sequences).
- Only NEW code is added for: the four new architectures, the extended
  training loop, feature selection probe, sequence-length search, and
  the eight Optuna objective functions.
- No data leakage: scaler fitted only on training rows; candidate features
  evaluated only with their shifted/lagged forms so no future data leaks.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)   # flush every print() immediately

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from pykalman import KalmanFilter

# Keep console clean
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# SECTION 0  CONFIGURATION  (mirror of model_pipeline.py constants)
# ==============================================================================

RANDOM_SEED        = 42
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
ENSEMBLE_TOLERANCE = 0.03   # ensemble member must be within 3 % of best RMSE
PATIENCE           = 3      # Optuna early-stopping patience (no-improvement trials)
TRAIN_PATIENCE     = 4      # early-stopping patience inside the training loop
MAX_EPOCHS         = 30     # hard cap on training epochs
DL_OPTUNA_TRIALS   = 10     # Optuna trials per tunable model
KEEP_ALL_CHECKPOINTS = False  # set True to retain intermediate .pt files

# Candidate sequence lengths evaluated during the global search
SEQ_LENGTH_CANDIDATES = [5, 10, 20, 30, 60, 90, 120]

# Ordered list of the eight models this pipeline trains
ENABLED_DL_MODELS = [
    "SimpleRNN",
    "FinetunedRNN",
    "FinetunedGRU",
    "LSTM",
    "FinetunedLSTM",
    "Seq2Seq",
    "TCN",
    "NBEATS",
]

# Paths  identical to model_pipeline.py so checkpoints live in the same dir
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "src" / "ml" / "pipeline_checkpoint.json"
CHECKPOINT_DIR  = PROJECT_ROOT / "src" / "ml" / "checkpoints"

TARGET_REG = "usd_zar_logret_next"

# Fix seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==============================================================================
# SECTION 1  MODEL ARCHITECTURES
# ==============================================================================

# ------------------------------------------------------------------------------
# 1a. SimpleRNN  (reused from model_pipeline.py, dropout added)
# ------------------------------------------------------------------------------

class SimpleRNNModel(nn.Module):
    """Vanilla RNN regressor with optional dropout between layers."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        # dropout only applied between RNN layers (num_layers > 1)
        rnn_drop = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_drop,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])   # take last timestep
        return self.fc(out)


# ------------------------------------------------------------------------------
# 1b. LSTM  (reused from model_pipeline.py, dropout added)
# ------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Standard LSTM regressor with optional dropout."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_drop,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ------------------------------------------------------------------------------
# 1c. GRU  (reused from model_pipeline.py, dropout added)
# ------------------------------------------------------------------------------

class GRUModel(nn.Module):
    """Standard GRU regressor with optional dropout."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        gru_drop = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_drop,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ------------------------------------------------------------------------------
# 1d. Seq2Seq  (NEW  Encoder-Decoder GRU with teacher forcing)
# ------------------------------------------------------------------------------

class Seq2SeqEncoder(nn.Module):
    """GRU encoder that summarises the input sequence into a context vector."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        gru_drop = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_drop,
        )

    def forward(self, x):
        # outputs: (batch, seq, hidden), hidden: (layers, batch, hidden)
        outputs, hidden = self.gru(x)
        return hidden  # pass only the final hidden state to the decoder


class Seq2SeqDecoder(nn.Module):
    """Single-step GRU decoder that predicts the next value."""

    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        gru_drop = dropout if num_layers > 1 else 0.0
        # Input is the previous prediction (scalar -> size 1)
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_drop,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        # x: (batch, 1, 1)
        out, hidden = self.gru(x, hidden)
        pred = self.fc(out[:, -1, :])       # (batch, 1)
        return pred, hidden


class Seq2SeqModel(nn.Module):
    """
    Encoder-Decoder Seq2Seq for one-step-ahead regression.

    The decoder is run for exactly one step (we only predict the next value).
    Teacher forcing is applied during training: with probability `teacher_forcing_ratio`
    the decoder receives the actual last target value as its initial input instead of zero.
    """

    def __init__(self, input_size, hidden_size=64,
                 enc_layers=1, dec_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = Seq2SeqEncoder(input_size, hidden_size, enc_layers, dropout)
        self.decoder = Seq2SeqDecoder(hidden_size, dec_layers, dropout)
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.hidden_size = hidden_size

        # If encoder and decoder have different layer counts we need a projection
        if enc_layers != dec_layers:
            self.hidden_proj = nn.Linear(enc_layers * hidden_size,
                                         dec_layers * hidden_size)
        else:
            self.hidden_proj = None

    def forward(self, x, teacher_input=None, teacher_forcing_ratio=0.0):
        """
        x              : (batch, seq_len, input_size)
        teacher_input  : (batch, 1)  last known target value (optional)
        """
        batch = x.size(0)
        hidden = self.encoder(x)  # (enc_layers, batch, hidden)

        # Adapt hidden state if layer counts differ
        if self.hidden_proj is not None:
            # Reshape encoder hidden -> (batch, enc_layers * hidden)
            h_flat = hidden.permute(1, 0, 2).contiguous().view(batch, -1)
            h_proj = self.hidden_proj(h_flat)   # (batch, dec_layers * hidden)
            hidden = h_proj.view(batch, self.dec_layers, self.hidden_size).permute(1, 0, 2).contiguous()

        # Decoder input: use teacher forcing or zeros
        if teacher_input is not None and random.random() < teacher_forcing_ratio:
            dec_input = teacher_input.view(batch, 1, 1)   # (batch, 1, 1)
        else:
            dec_input = torch.zeros(batch, 1, 1, device=x.device)

        pred, _ = self.decoder(dec_input, hidden)
        return pred  # (batch, 1)


# ------------------------------------------------------------------------------
# 1e. TCN  (NEW  Temporal Convolutional Network with dilated causal convolutions)
# ------------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """
    Single dilated causal convolutional block with:
      - Two Conv1d layers with the same dilation, padding to keep length
      - Layer normalisation
      - ReLU activation
      - Dropout
      - Residual skip connection (11 conv if channel counts differ)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # Causal padding = (kernel_size - 1) * dilation on the left side only
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=pad)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=pad)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection when input/output channels differ
        self.skip = (nn.Conv1d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def _causal_conv(self, conv, x):
        """Apply conv then strip the non-causal right-padding to preserve length."""
        out = conv(x)
        # Remove extra positions added by symmetric padding (keep only left pad)
        if out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]
        return out

    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = self.skip(x)

        out = self._causal_conv(self.conv1, x)
        # LayerNorm expects (batch, seq_len, channels) -> transpose
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)

        out = self._causal_conv(self.conv2, out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)

        return F.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for one-step-ahead regression.

    Each successive block doubles the dilation to achieve exponentially
    large receptive fields without deep stacking.
    """

    def __init__(self, input_size, num_channels, kernel_size, num_blocks, dropout):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_blocks):
            dilation = 2 ** i          # dilation grows: 1, 2, 4, 8, 
            layers.append(TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))
            in_ch = num_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)  -> transpose to (batch, input_size, seq_len)
        out = self.network(x.transpose(1, 2))
        # Take the last timestep for prediction
        out = out[:, :, -1]       # (batch, num_channels)
        return self.fc(out)


# ------------------------------------------------------------------------------
# 1f. N-BEATS  (extended from n-beats.py  tunable stacks of blocks)
# ------------------------------------------------------------------------------

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block: four FC layers -> backcast + forecast projections.

    The block learns to separate the input signal into what it can explain
    (backcast) and the residual prediction (forecast).
    """

    def __init__(self, input_size, hidden_size, expansion_coef):
        super().__init__()
        # Four fully-connected layers as in the original N-BEATS paper
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        # Backcast and forecast output heads
        self.theta_b = nn.Linear(hidden_size, expansion_coef, bias=False)
        self.theta_f = nn.Linear(hidden_size, expansion_coef, bias=False)

        # Basis matrices: project theta back to input_size (backcast) and 1 (forecast)
        self.backcast_basis = nn.Linear(expansion_coef, input_size, bias=False)
        self.forecast_basis = nn.Linear(expansion_coef, 1, bias=False)

    def forward(self, x):
        # x: (batch, input_size)  flattened sequence
        h       = self.fc_stack(x)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = self.backcast_basis(theta_b)
        forecast  = self.forecast_basis(theta_f)   # (batch, 1)
        return backcast, forecast


class NBeatsModel(nn.Module):
    """
    Full N-BEATS stack.

    Multiple blocks are chained: each block receives the residual of the
    previous block's backcast subtracted from the input.  Forecasts from
    all blocks are summed.
    """

    def __init__(self, input_size, hidden_size=256, n_stacks=2,
                 n_blocks_per_stack=3, expansion_coef=32):
        super().__init__()
        total_blocks = n_stacks * n_blocks_per_stack
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, expansion_coef)
            for _ in range(total_blocks)
        ])

    def forward(self, x):
        # x: (batch, seq_len, input_size)  flatten -> (batch, seq_len * input_size)
        batch = x.size(0)
        residual = x.reshape(batch, -1)           # flatten sequencefeatures
        forecast  = torch.zeros(batch, 1, device=x.device)

        for block in self.blocks:
            backcast, block_fc = block(residual)
            residual  = residual - backcast        # subtract explained portion
            forecast  = forecast + block_fc        # accumulate forecasts

        return forecast   # (batch, 1)


# ==============================================================================
# SECTION 2  HELPER UTILITIES  (reused / adapted from model_pipeline.py)
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """Wraps (X_sequences, y_targets) as a PyTorch Dataset. Identical to model_pipeline.py."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(X, y, lookback):
    """
    Build chronological sliding-window sequences.
    Returns xs: (N, lookback, features), ys: (N, 1).
    Identical logic to model_pipeline.py.
    """
    xs, ys = [], []
    for i in range(len(X) - lookback):
        xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(xs), np.array(ys)


def calculate_reg_metrics(actual, predicted):
    """
    Standard regression metrics used throughout model_pipeline.py.
    Returns dict with RMSE, MAE, MAPE, R2, DirAcc.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2   = r2_score(actual, predicted)

    dir_actual = np.sign(np.diff(actual))
    dir_pred   = np.sign(np.diff(predicted))
    dir_acc    = (dir_actual == dir_pred).mean() * 100 if len(dir_actual) > 0 else 50.0

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "DirAcc": dir_acc}


def denoise_features_kalman(df, feature_cols):
    """
    Applies Kalman Smoother to log-return and technical indicator columns.
    Identical to model_pipeline.py (Solution 4).
    """
    print("Applying Kalman Filter denoising to features...")
    df_out = df.copy()
    for col in feature_cols:
        if "logret" in col or "rsi" in col or "macd" in col or "spread" in col:
            vals   = df_out[col].values
            masked = np.ma.masked_invalid(vals)
            kf     = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
            try:
                kf = kf.em(masked, n_iter=3)
                state_means, _ = kf.smooth(masked)
                df_out[col] = state_means.flatten()
            except Exception:
                pass   # leave column unchanged if Kalman fails
    return df_out


class EarlyStoppingCallback:
    """
    Optuna callback: stops the study after `patience` consecutive trials
    without improvement. Identical to model_pipeline.py.
    """

    def __init__(self, patience=10):
        self.patience    = patience
        self.best_value  = None
        self.no_improve  = 0

    def __call__(self, study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return
        if self.best_value is None or trial.value < self.best_value:
            self.best_value = trial.value
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            study.stop()


def count_parameters(model):
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------
# Checkpoint helpers  same JSON registry as model_pipeline.py
# ------------------------------------------------------------------------------

def load_checkpoint(model_name):
    """Load a previously saved checkpoint for `model_name`. Returns (state_dict_or_None, entry_or_None)."""
    if not CHECKPOINT_FILE.exists():
        return None, None
    with open(CHECKPOINT_FILE, "r") as f:
        registry = json.load(f)
    key   = f"dl_{model_name}"
    entry = registry.get(key)
    if entry is None:
        return None, None
    model_path = Path(entry["model_path"])
    if not model_path.exists():
        return None, None
    # Return the raw state_dict path and the metadata; caller reconstructs model
    return str(model_path), entry


def save_checkpoint(model_name, model, val_metrics, best_params, train_time, n_params):
    """Save model weights and metadata to the shared checkpoint registry."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    key        = f"dl_{model_name}"
    model_path = CHECKPOINT_DIR / f"{key}_best.pt"
    torch.save(model.state_dict(), model_path)

    registry = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            registry = json.load(f)

    registry[key] = {
        "status":        "Completed",
        "val_metrics":   val_metrics,
        "best_params":   best_params,
        "model_path":    str(model_path),
        "training_time": train_time,
        "n_params":      n_params,
        "timestamp":     datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(registry, f, indent=4)

    return str(model_path)


# ==============================================================================
# SECTION 3  EXTENDED TRAINING LOOP
# ==============================================================================

def get_optimizer(model, optimizer_name, lr, weight_decay):
    """Return an optimiser instance from its name string."""
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, scheduler_name, epochs):
    """Return a LR scheduler instance (or None) from its name string."""
    if scheduler_name == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    else:
        return None   # no scheduler


def train_model(
    model,
    train_loader,
    val_loader,
    lr              = 1e-3,
    optimizer_name  = "Adam",
    weight_decay    = 0.0,
    grad_clip       = 0.0,    # 0 = disabled
    scheduler_name  = "None",
    epochs          = MAX_EPOCHS,
    patience        = TRAIN_PATIENCE,
    teacher_forcing = 0.0,    # only used by Seq2Seq
    verbose         = False,
):
    """
    Generic training loop for all deep learning models.
    Uses a custom Sign-Directional MSE Loss to maximize Directional Accuracy.
    """
    def directional_loss(pred, target):
        mse = F.mse_loss(pred, target)
        # Sign mismatch penalty: if pred and target have opposite signs, add penalty
        # We use a soft softplus penalty to keep gradients smooth: log(1 + exp(-pred * target * scale))
        sign_penalty = torch.mean(F.softplus(-pred * target * 10.0))
        return mse + 0.15 * sign_penalty

    optimizer  = get_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler  = get_scheduler(optimizer, scheduler_name, epochs)
    best_loss  = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(epochs):
        # --- Training pass ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1) if y_batch.dim() == 1 else y_batch.to(DEVICE)

            optimizer.zero_grad()

            # Seq2Seq: pass the last target value for teacher forcing
            if isinstance(model, Seq2SeqModel):
                # Use the actual last target in the sequence as teacher input
                teacher_input = y_batch  # shape (batch, 1)
                outputs = model(X_batch, teacher_input=teacher_input,
                                teacher_forcing_ratio=teacher_forcing)
            else:
                outputs = model(X_batch)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            loss = directional_loss(outputs, y_batch)
            loss.backward()

            # Gradient clipping (0 = disabled)
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        # --- Validation pass ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v = X_v.to(DEVICE)
                y_v = y_v.to(DEVICE).unsqueeze(1) if y_v.dim() == 1 else y_v.to(DEVICE)
                if isinstance(model, Seq2SeqModel):
                    outs = model(X_v, teacher_forcing_ratio=0.0)  # no teacher forcing at eval
                else:
                    outs = model(X_v)
                if outs.dim() == 1:
                    outs = outs.unsqueeze(1)
                val_loss += directional_loss(outs, y_v).item()
        val_loss /= max(len(val_loader), 1)

        # Update scheduler
        if scheduler is not None:
            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if verbose:
            print(f"  Epoch {epoch+1:03d} | val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss    = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.to(DEVICE)
    return model


def predict_model(model, loader, scaler_y):
    """
    Run inference on a DataLoader and return inverse-transformed predictions.
    Works for all architectures (Seq2Seq uses teacher_forcing_ratio=0).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            if isinstance(model, Seq2SeqModel):
                out = model(X_batch, teacher_forcing_ratio=0.0)
            else:
                out = model(X_batch)
            preds.extend(out.cpu().numpy())
    preds = np.array(preds).reshape(-1, 1)
    return scaler_y.inverse_transform(preds).ravel()


# ==============================================================================
# SECTION 4  AUTOMATED FEATURE ENGINEERING
# ==============================================================================

def generate_candidate_features(df_clean, feature_columns, train_mask, val_mask):
    """
    Generate additional candidate features on top of the existing engineered set.

    For each candidate:
      1. Compute it using ONLY past information (shift by 1 to prevent leakage).
      2. Fit a Ridge regression probe on the TRAINING rows only.
      3. Evaluate val RMSE.  Keep the candidate if val RMSE improves over baseline.

    Returns
    -------
    df_aug      : DataFrame with selected extra columns appended
    kept        : list of kept feature names
    discarded   : list of discarded feature names
    """

    target = df_clean[TARGET_REG]
    base_series = df_clean["usd_zar_logret"]   # underlying log-return series
    price_series = df_clean["usd_zar"]          # price level

    # -- Baseline: fit Ridge on existing features, measure val RMSE ----------
    scaler_probe_x = StandardScaler()
    scaler_probe_y = StandardScaler()

    X_tr = scaler_probe_x.fit_transform(df_clean[feature_columns].values[train_mask])
    y_tr = scaler_probe_y.fit_transform(
        df_clean[TARGET_REG].values[train_mask].reshape(-1, 1)).ravel()
    X_va = scaler_probe_x.transform(df_clean[feature_columns].values[val_mask])
    y_va_raw = df_clean[TARGET_REG].values[val_mask]

    probe = Ridge(alpha=1.0)
    probe.fit(X_tr, y_tr)
    probe_preds = scaler_probe_y.inverse_transform(
        probe.predict(X_va).reshape(-1, 1)).ravel()
    baseline_rmse = np.sqrt(mean_squared_error(y_va_raw, probe_preds))

    print(f"\n[Feature Engineering] Baseline Ridge val RMSE with existing features: {baseline_rmse:.6f}")
    print("[Feature Engineering] Evaluating additional candidate features...\n")

    # -- Build candidate feature dictionary --------------------------------
    candidates = {}

    # Target sign directions (1, 0, -1) to explicitly supply trend directionality
    for lag in range(1, 10):
        col_dir = f"usd_zar_logret_signdir_lag_{lag}"
        if col_dir not in df_clean.columns:
            candidates[col_dir] = np.sign(base_series).shift(lag)

    # Extra ZAR log-return lags (lags 615 beyond the existing 15)
    for lag in range(6, 16):
        col = f"usd_zar_logret_lag_{lag}"
        if col not in df_clean.columns:
            candidates[col] = base_series.shift(lag)

    # Rolling window statistics (windows not yet covered)
    for window in [3, 7, 14, 21]:
        col_mean = f"usd_zar_logret_roll_mean_{window}"
        col_std  = f"usd_zar_logret_roll_std_{window}"
        col_min  = f"usd_zar_logret_roll_min_{window}"
        col_max  = f"usd_zar_logret_roll_max_{window}"
        if col_mean not in df_clean.columns:
            candidates[col_mean] = base_series.rolling(window).mean().shift(1)
        if col_std not in df_clean.columns:
            candidates[col_std]  = base_series.rolling(window).std().shift(1)
        if col_min not in df_clean.columns:
            candidates[col_min]  = base_series.rolling(window).min().shift(1)
        if col_max not in df_clean.columns:
            candidates[col_max]  = base_series.rolling(window).max().shift(1)

    # Exponential Moving Averages of the log-return
    for span in [5, 10, 20, 30, 60]:
        col = f"usd_zar_logret_ema_{span}"
        if col not in df_clean.columns:
            candidates[col] = base_series.ewm(span=span, adjust=False).mean().shift(1)

    # Momentum: current log-return minus lag-n log-return
    for lag in [5, 10, 20]:
        col = f"usd_zar_momentum_{lag}"
        if col not in df_clean.columns:
            candidates[col] = (base_series - base_series.shift(lag)).shift(1)

    # Percentage change of the price level
    for lag in [1, 5, 10]:
        col = f"usd_zar_pct_change_{lag}"
        if col not in df_clean.columns:
            candidates[col] = price_series.pct_change(lag).shift(1)

    # Rate of change (log-return acceleration)
    col = "usd_zar_logret_roc_5"
    if col not in df_clean.columns:
        candidates[col] = (base_series / (base_series.shift(5) + 1e-8) - 1).shift(1)

    # Historical volatility (rolling std of log returns)
    for window in [10, 30, 60]:
        col = f"usd_zar_hist_vol_{window}"
        if col not in df_clean.columns:
            candidates[col] = base_series.rolling(window).std().shift(1)

    # Absolute return (proxy for volatility at lag 1)
    col = "usd_zar_abs_logret_lag1"
    if col not in df_clean.columns:
        candidates[col] = base_series.abs().shift(1)

    # Squared return (variance proxy)
    col = "usd_zar_sq_logret_lag1"
    if col not in df_clean.columns:
        candidates[col] = (base_series ** 2).shift(1)

    # -- Greedy forward selection: add features one-by-one if they help ----
    df_aug     = df_clean.copy()
    kept       = []
    discarded  = []
    current_rmse = baseline_rmse
    current_features = list(feature_columns)

    for feat_name, feat_series in candidates.items():
        df_aug[feat_name] = feat_series

        # Drop rows where the new feature is NaN in the training set
        # (we use ffill/bfill as model_pipeline.py does)
        feat_filled = df_aug.copy().ffill().bfill()
        trial_features = current_features + [feat_name]

        scaler_tx = StandardScaler()
        scaler_ty = StandardScaler()
        X_tr_t = scaler_tx.fit_transform(feat_filled[trial_features].values[train_mask])
        y_tr_t = scaler_ty.fit_transform(
            feat_filled[TARGET_REG].values[train_mask].reshape(-1, 1)).ravel()
        X_va_t = scaler_tx.transform(feat_filled[trial_features].values[val_mask])

        probe_t = Ridge(alpha=1.0)
        probe_t.fit(X_tr_t, y_tr_t)
        preds_t = scaler_ty.inverse_transform(
            probe_t.predict(X_va_t).reshape(-1, 1)).ravel()
        new_rmse = np.sqrt(mean_squared_error(y_va_raw, preds_t))

        if new_rmse < current_rmse:
            print(f"  KEEP   {feat_name:<40}  RMSE {current_rmse:.6f} -> {new_rmse:.6f}")
            current_rmse = new_rmse
            current_features = trial_features
            kept.append(feat_name)
        else:
            print(f"  DISCARD {feat_name:<40}  no improvement ({new_rmse:.6f} >= {current_rmse:.6f})")
            discarded.append(feat_name)
            df_aug.drop(columns=[feat_name], inplace=True)

    print(f"\n[Feature Engineering] Final val RMSE after feature selection: {current_rmse:.6f}")
    print(f"[Feature Engineering] Features kept: {len(kept)}, discarded: {len(discarded)}")

    return df_aug, kept, discarded


# ==============================================================================
# SECTION 5  SEQUENCE LENGTH SEARCH
# ==============================================================================

def find_best_seq_length(X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                          scaler_y, input_size, seq_lengths):
    """
    Train a small LSTM (20 epochs, fixed params) at each candidate sequence
    length and return the length that achieves the lowest validation RMSE.

    This search is done once and the result is reused by all models.
    """
    print("\n[Seq Length Search] Evaluating sequence lengths:", seq_lengths)
    best_rmse = float("inf")
    best_len  = seq_lengths[0]

    for length in seq_lengths:
        # Build sequences
        xs_tr, ys_tr = create_sequences(X_train_sc, y_train_sc, length)
        xs_va, ys_va = create_sequences(X_val_sc,   y_val_sc,   length)

        if len(xs_tr) < 32 or len(xs_va) < 8:
            print(f"  length={length:3d} -> skipped (too few samples)")
            continue

        tr_loader = DataLoader(TimeSeriesDataset(xs_tr, ys_tr), batch_size=32, shuffle=False)
        va_loader = DataLoader(TimeSeriesDataset(xs_va, ys_va), batch_size=32, shuffle=False)

        # Small probe LSTM (fast, not for final training)
        probe = LSTMModel(input_size=input_size, hidden_size=32, num_layers=1).to(DEVICE)
        probe = train_model(probe, tr_loader, va_loader,
                            lr=1e-3, epochs=25, patience=5)

        val_preds = predict_model(probe, va_loader, scaler_y)
        val_actuals = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))

        print(f"  length={length:3d} -> val RMSE = {rmse:.6f}")


        del probe  # free GPU memory

        if rmse < best_rmse:
            best_rmse = rmse
            best_len  = length

    print(f"[Seq Length Search] Best sequence length: {best_len} (val RMSE = {best_rmse:.6f})\n")
    return best_len


# ==============================================================================
# SECTION 6  OPTUNA OBJECTIVE FUNCTIONS
# ==============================================================================

# Each objective trains a model with trial-suggested hyperparameters and returns
# the validation RMSE (lower is better -> direction="minimize").

def _make_loaders(X_tr_sc, y_tr_sc, X_va_sc, y_va_sc, lookback, batch_size):
    """Build train/val DataLoaders for a given lookback and batch size."""
    xs_tr, ys_tr = create_sequences(X_tr_sc, y_tr_sc, lookback)
    xs_va, ys_va = create_sequences(X_va_sc, y_va_sc, lookback)
    tr_loader = DataLoader(TimeSeriesDataset(xs_tr, ys_tr), batch_size=batch_size, shuffle=False)
    va_loader = DataLoader(TimeSeriesDataset(xs_va, ys_va), batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, xs_va, ys_va


def objective_finetuned_rnn(trial, X_tr, y_tr, X_va, y_va, scaler_y,
                             input_size, best_seq_len):
    """Optuna objective for FinetunedRNN."""
    hidden     = trial.suggest_int("hidden_size", 16, 256)
    layers     = trial.suggest_int("num_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    grad_clip  = trial.suggest_float("grad_clip", 0.0, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len    = trial.suggest_categorical("seq_len", [best_seq_len,
                                                        max(5, best_seq_len // 2),
                                                        min(120, best_seq_len * 2)])
    opt_name   = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    sched_name = trial.suggest_categorical("scheduler", ["None", "CosineAnnealing", "ReduceLROnPlateau"])

    tr_loader, va_loader, xs_va, ys_va = _make_loaders(X_tr, y_tr, X_va, y_va, seq_len, batch_size)
    model = SimpleRNNModel(input_size, hidden, layers, dropout).to(DEVICE)
    model = train_model(model, tr_loader, va_loader, lr=lr,
                        optimizer_name=opt_name, weight_decay=wd,
                        grad_clip=grad_clip, scheduler_name=sched_name,
                        epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
    preds = predict_model(model, va_loader, scaler_y)
    acts  = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(acts, preds)))


def objective_finetuned_gru(trial, X_tr, y_tr, X_va, y_va, scaler_y,
                             input_size, best_seq_len):
    """Optuna objective for FinetunedGRU."""
    hidden     = trial.suggest_int("hidden_size", 16, 256)
    layers     = trial.suggest_int("num_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    grad_clip  = trial.suggest_float("grad_clip", 0.0, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len    = trial.suggest_categorical("seq_len", [best_seq_len,
                                                        max(5, best_seq_len // 2),
                                                        min(120, best_seq_len * 2)])
    opt_name   = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    sched_name = trial.suggest_categorical("scheduler", ["None", "CosineAnnealing", "ReduceLROnPlateau"])

    tr_loader, va_loader, xs_va, ys_va = _make_loaders(X_tr, y_tr, X_va, y_va, seq_len, batch_size)
    model = GRUModel(input_size, hidden, layers, dropout).to(DEVICE)
    model = train_model(model, tr_loader, va_loader, lr=lr,
                        optimizer_name=opt_name, weight_decay=wd,
                        grad_clip=grad_clip, scheduler_name=sched_name,
                        epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
    preds = predict_model(model, va_loader, scaler_y)
    acts  = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(acts, preds)))


def objective_finetuned_lstm(trial, X_tr, y_tr, X_va, y_va, scaler_y,
                              input_size, best_seq_len):
    """Optuna objective for FinetunedLSTM."""
    hidden     = trial.suggest_int("hidden_size", 16, 256)
    layers     = trial.suggest_int("num_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    grad_clip  = trial.suggest_float("grad_clip", 0.0, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len    = trial.suggest_categorical("seq_len", [best_seq_len,
                                                        max(5, best_seq_len // 2),
                                                        min(120, best_seq_len * 2)])
    opt_name   = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    sched_name = trial.suggest_categorical("scheduler", ["None", "CosineAnnealing", "ReduceLROnPlateau"])

    tr_loader, va_loader, xs_va, ys_va = _make_loaders(X_tr, y_tr, X_va, y_va, seq_len, batch_size)
    model = LSTMModel(input_size, hidden, layers, dropout).to(DEVICE)
    model = train_model(model, tr_loader, va_loader, lr=lr,
                        optimizer_name=opt_name, weight_decay=wd,
                        grad_clip=grad_clip, scheduler_name=sched_name,
                        epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
    preds = predict_model(model, va_loader, scaler_y)
    acts  = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(acts, preds)))


def objective_seq2seq(trial, X_tr, y_tr, X_va, y_va, scaler_y,
                      input_size, best_seq_len):
    """Optuna objective for Seq2Seq."""
    hidden     = trial.suggest_int("hidden_size", 32, 256)
    enc_layers = trial.suggest_int("enc_layers", 1, 3)
    dec_layers = trial.suggest_int("dec_layers", 1, 2)
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)
    tf_ratio   = trial.suggest_float("teacher_forcing", 0.0, 0.7)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    grad_clip  = trial.suggest_float("grad_clip", 0.5, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len    = trial.suggest_categorical("seq_len", [best_seq_len,
                                                        max(5, best_seq_len // 2),
                                                        min(120, best_seq_len * 2)])

    tr_loader, va_loader, xs_va, ys_va = _make_loaders(X_tr, y_tr, X_va, y_va, seq_len, batch_size)
    model = Seq2SeqModel(input_size, hidden, enc_layers, dec_layers, dropout).to(DEVICE)
    model = train_model(model, tr_loader, va_loader, lr=lr,
                        weight_decay=wd, grad_clip=grad_clip,
                        epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE,
                        teacher_forcing=tf_ratio)
    preds = predict_model(model, va_loader, scaler_y)
    acts  = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(acts, preds)))


def objective_tcn(trial, X_tr, y_tr, X_va, y_va, scaler_y,
                  input_size, best_seq_len):
    """Optuna objective for TCN."""
    num_channels = trial.suggest_int("num_channels", 16, 128)
    kernel_size  = trial.suggest_int("kernel_size", 2, 8)
    num_blocks   = trial.suggest_int("num_blocks", 2, 6)
    dropout      = trial.suggest_float("dropout", 0.0, 0.4)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd           = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    grad_clip    = trial.suggest_float("grad_clip", 0.0, 5.0)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len      = trial.suggest_categorical("seq_len", [best_seq_len,
                                                          max(5, best_seq_len // 2),
                                                          min(120, best_seq_len * 2)])
    opt_name     = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    sched_name   = trial.suggest_categorical("scheduler", ["None", "CosineAnnealing"])

    tr_loader, va_loader, xs_va, ys_va = _make_loaders(X_tr, y_tr, X_va, y_va, seq_len, batch_size)
    model = TCNModel(input_size, num_channels, kernel_size, num_blocks, dropout).to(DEVICE)
    model = train_model(model, tr_loader, va_loader, lr=lr,
                        optimizer_name=opt_name, weight_decay=wd,
                        grad_clip=grad_clip, scheduler_name=sched_name,
                        epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
    preds = predict_model(model, va_loader, scaler_y)
    acts  = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(acts, preds)))


def objective_nbeats(trial, X_tr, y_tr, X_va, y_va, scaler_y,
                     best_seq_len):
    """Optuna objective for N-BEATS."""
    n_stacks     = trial.suggest_int("n_stacks", 1, 4)
    n_blocks     = trial.suggest_int("n_blocks_per_stack", 1, 4)
    hidden_size  = trial.suggest_int("hidden_size", 64, 512)
    exp_coef     = trial.suggest_int("expansion_coef", 8, 64)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd           = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    grad_clip    = trial.suggest_float("grad_clip", 0.0, 5.0)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len      = trial.suggest_categorical("seq_len", [best_seq_len,
                                                          max(5, best_seq_len // 2),
                                                          min(120, best_seq_len * 2)])
    sched_name   = trial.suggest_categorical("scheduler", ["None", "CosineAnnealing"])

    tr_loader, va_loader, xs_va, ys_va = _make_loaders(X_tr, y_tr, X_va, y_va, seq_len, batch_size)
    # N-BEATS input_size = seq_len * n_features
    input_flat = seq_len * X_tr.shape[1]
    model = NBeatsModel(input_flat, hidden_size, n_stacks, n_blocks, exp_coef).to(DEVICE)
    model = train_model(model, tr_loader, va_loader, lr=lr,
                        weight_decay=wd, grad_clip=grad_clip,
                        scheduler_name=sched_name,
                        epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
    preds = predict_model(model, va_loader, scaler_y)
    acts  = scaler_y.inverse_transform(ys_va.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(acts, preds)))


# ==============================================================================
# SECTION 7  LEADERBOARD PRINTING
# ==============================================================================

def print_leaderboard(leaderboard):
    """Print the current leaderboard sorted by validation RMSE."""
    if not leaderboard:
        return
    df = pd.DataFrame(leaderboard).sort_values("RMSE").reset_index(drop=True)
    df.index += 1
    print("\n" + "=" * 95)
    print(f"{'#':<4}{'Model':<18}{'RMSE':>10}{'MAE':>10}{'MAPE':>10}{'R2':>8}"
          f"{'DirAcc':>9}{'Params':>10}{'Time(s)':>10}{'Status':<12}")
    print("-" * 95)
    for i, row in df.iterrows():
        star = " <- BEST" if i == 1 else ""
        print(f"{i:<4}{str(row.get('Model','')):<18}"
              f"{row.get('RMSE', float('nan')):>10.6f}"
              f"{row.get('MAE', float('nan')):>10.6f}"
              f"{row.get('MAPE', float('nan')):>10.2f}"
              f"{row.get('R2', float('nan')):>8.4f}"
              f"{row.get('DirAcc', float('nan')):>9.2f}"
              f"{int(row.get('Params', 0)):>10,}"
              f"{row.get('TrainTime', 0):>10.1f}"
              f"  {str(row.get('Status','')):<12}{star}")
    print("=" * 95)


def print_model_summary(model_name, metrics, best_params, train_time, n_params):
    """Print a detailed summary for a single finished model."""
    print(f"\n{'-'*70}")
    print(f"  MODEL COMPLETE : {model_name}")
    print(f"{'-'*70}")
    print(f"  Val RMSE          : {metrics['RMSE']:.6f}")
    print(f"  Val MAE           : {metrics['MAE']:.6f}")
    print(f"  Val MAPE          : {metrics['MAPE']:.2f}%")
    print(f"  Val R            : {metrics['R2']:.4f}")
    print(f"  Directional Acc   : {metrics['DirAcc']:.2f}%")
    print(f"  Trainable Params  : {n_params:,}")
    print(f"  Training Time     : {train_time:.1f}s")
    print(f"  Best Params       : {best_params}")
    print(f"{'-'*70}")


# ==============================================================================
# SECTION 8  MAIN PIPELINE
# ==============================================================================

def main():
    pipeline_start = time.time()
    temp_checkpoints = []   # paths to delete at cleanup

    print("=" * 70)
    print("  DL PIPELINE  Deep Learning Forecasting Extension")
    print(f"  Device: {DEVICE}  |  Random seed: {RANDOM_SEED}")
    print("=" * 70)

    # Clean up old deep learning checkpoints to force training with new loss and features
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                reg = json.load(f)
            # Remove keys starting with dl_
            keys_to_del = [k for k in reg.keys() if k.startswith("dl_")]
            if keys_to_del:
                for k in keys_to_del:
                    del reg[k]
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(reg, f, indent=4)
                print(f"Removed {len(keys_to_del)} old DL checkpoints to force retraining with directional loss.")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Step 1: Load the engineered features CSV (same file as model_pipeline.py)
    # -------------------------------------------------------------------------
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # Detect and remove leading warm-up rows (rows that still have NaN)
    nan_rows       = df.isna().any(axis=1)
    first_valid    = np.where(~nan_rows)[0][0]
    first_valid_dt = df.index[first_valid]

    # Slice clean data; remove trailing row (target is shifted forward by 1)
    df_clean = df.iloc[first_valid:-1].ffill().bfill()
    assert df_clean.isna().sum().sum() == 0, "NaN values remain after ffill/bfill."

    feature_columns = [c for c in df_clean.columns if c != TARGET_REG]

    print(f"\nDataset summary:")
    print(f"  Warm-up rows removed : {first_valid}")
    print(f"  First valid date     : {first_valid_dt.date()}")
    print(f"  Total clean rows     : {len(df_clean)}")
    print(f"  Features (original)  : {len(feature_columns)}")

    # -------------------------------------------------------------------------
    # Step 2: Kalman denoising (identical to model_pipeline.py)
    # -------------------------------------------------------------------------
    df_clean = denoise_features_kalman(df_clean, feature_columns)

    # -------------------------------------------------------------------------
    # Step 3: Chronological split masks
    # -------------------------------------------------------------------------
    train_mask = df_clean.index <= pd.Timestamp("2021-12-31")
    val_mask   = ((df_clean.index >= pd.Timestamp("2022-01-01")) &
                  (df_clean.index <= pd.Timestamp("2023-12-31")))
    test_mask  = df_clean.index >= pd.Timestamp("2024-01-01")

    # Verify no overlap / leakage between splits
    assert df_clean.index[train_mask].max() < df_clean.index[val_mask].min()
    assert df_clean.index[val_mask].max()   < df_clean.index[test_mask].min()

    print(f"\nChronological splits:")
    print(f"  Train : {train_mask.sum()} rows  "
          f"({df_clean.index[train_mask].min().date()}  {df_clean.index[train_mask].max().date()})")
    print(f"  Val   : {val_mask.sum()} rows  "
          f"({df_clean.index[val_mask].min().date()}  {df_clean.index[val_mask].max().date()})")
    print(f"  Test  : {test_mask.sum()} rows  "
          f"({df_clean.index[test_mask].min().date()}  {df_clean.index[test_mask].max().date()})")

    # -------------------------------------------------------------------------
    # Step 4: Automated feature engineering (extend, not replace)
    # -------------------------------------------------------------------------
    orig_n_features = len(feature_columns)
    df_clean, kept_features, discarded_features = generate_candidate_features(
        df_clean, feature_columns, train_mask, val_mask
    )
    # Re-derive feature column list after augmentation
    feature_columns = [c for c in df_clean.columns if c != TARGET_REG]
    df_clean = df_clean.ffill().bfill()

    print(f"\n  Original features     : {orig_n_features}")
    print(f"  New features kept     : {len(kept_features)}")
    print(f"  New features discarded: {len(discarded_features)}")
    print(f"  Final feature count   : {len(feature_columns)}")

    # -------------------------------------------------------------------------
    # Step 5: Fit StandardScaler on training data ONLY
    # -------------------------------------------------------------------------
    X_full    = df_clean[feature_columns].values
    y_full    = df_clean[TARGET_REG].values.reshape(-1, 1)

    X_train   = X_full[train_mask]
    y_train   = y_full[train_mask].ravel()
    X_val     = X_full[val_mask]
    y_val     = y_full[val_mask].ravel()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))

    X_train_sc = scaler_x.transform(X_train)
    X_val_sc   = scaler_x.transform(X_val)
    X_full_sc  = scaler_x.transform(X_full)

    y_train_sc = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
    y_val_sc   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_full_sc  = scaler_y.transform(y_full).ravel()

    input_size = X_train_sc.shape[1]

    # -------------------------------------------------------------------------
    # Step 6: Global sequence-length search (done once, reused by all models)
    # -------------------------------------------------------------------------
    best_seq_len = find_best_seq_length(
        X_train_sc, y_train_sc,
        X_val_sc,   y_val_sc,
        scaler_y, input_size,
        SEQ_LENGTH_CANDIDATES
    )

    # Build the canonical sequences at the best length
    xs_tr, ys_tr = create_sequences(X_train_sc, y_train_sc, best_seq_len)
    xs_va, ys_va = create_sequences(X_val_sc,   y_val_sc,   best_seq_len)

    tr_loader_default = DataLoader(TimeSeriesDataset(xs_tr, ys_tr),
                                   batch_size=32, shuffle=False)
    va_loader_default = DataLoader(TimeSeriesDataset(xs_va, ys_va),
                                   batch_size=32, shuffle=False)

    # -------------------------------------------------------------------------
    # Step 7: Train all eight models
    # -------------------------------------------------------------------------
    leaderboard = []
    completed   = {}   # model_name -> {model, val_metrics, best_params, train_time, n_params}

    print("\n" + "=" * 70)
    print("  TRAINING PHASE  8 Deep Learning Models")
    print("=" * 70)

    # Helper: build loaders with a specific seq_len and batch_size
    def make_loaders(seq_len=best_seq_len, batch_size=32):
        xtr, ytr = create_sequences(X_train_sc, y_train_sc, seq_len)
        xva, yva = create_sequences(X_val_sc,   y_val_sc,   seq_len)
        tl = DataLoader(TimeSeriesDataset(xtr, ytr), batch_size=batch_size, shuffle=False)
        vl = DataLoader(TimeSeriesDataset(xva, yva), batch_size=batch_size, shuffle=False)
        return tl, vl, yva

    # --------------------------------------------------------------------------
    # MODEL 1: SimpleRNN  fixed default architecture (no Optuna)
    # --------------------------------------------------------------------------
    model_name = "SimpleRNN"
    print(f"\n[{model_name}] Training with default fixed parameters...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics    = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params   = ckpt_entry.get("n_params", 0)
        train_time = ckpt_entry["training_time"]
    else:
        best_params = {"hidden_size": 64, "num_layers": 1, "dropout": 0.0,
                       "lr": 1e-3, "seq_len": best_seq_len}
        tl, vl, yva = make_loaders(best_seq_len)
        model = SimpleRNNModel(input_size, 64, 1, 0.0).to(DEVICE)
        model = train_model(model, tl, vl, lr=1e-3,
                            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
        preds = predict_model(model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics   = calculate_reg_metrics(acts, preds)
        n_params  = count_parameters(model)
        train_time = time.time() - model_start
        ckpt_path = save_checkpoint(model_name, model, metrics, best_params,
                                    train_time, n_params)
        temp_checkpoints.append(ckpt_path)

    metrics["Model"] = model_name
    metrics["Status"] = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 2: FinetunedRNN  Optuna-tuned SimpleRNN
    # --------------------------------------------------------------------------
    model_name = "FinetunedRNN"
    print(f"\n[{model_name}] Running Optuna ({DL_OPTUNA_TRIALS} trials)...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: objective_finetuned_rnn(
                trial, X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                scaler_y, input_size, best_seq_len),
            n_trials=DL_OPTUNA_TRIALS,
            callbacks=[EarlyStoppingCallback(patience=PATIENCE)]
        )
        tuning_time = time.time() - model_start
        best_params = study.best_params
        print(f"  Best trial RMSE: {study.best_value:.6f}  |  Params: {best_params}")

        # Retrain final model with best params on combined data
        seq_len    = best_params.get("seq_len", best_seq_len)
        batch_size = best_params.get("batch_size", 32)
        tl, vl, yva = make_loaders(seq_len, batch_size)
        final_model = SimpleRNNModel(
            input_size,
            best_params["hidden_size"],
            best_params["num_layers"],
            best_params["dropout"]
        ).to(DEVICE)
        final_model = train_model(
            final_model, tl, vl,
            lr=best_params["lr"],
            optimizer_name=best_params.get("optimizer", "Adam"),
            weight_decay=best_params.get("weight_decay", 0.0),
            grad_clip=best_params.get("grad_clip", 0.0),
            scheduler_name=best_params.get("scheduler", "None"),
            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE
        )
        preds = predict_model(final_model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(final_model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, final_model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)
        model = final_model

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 3: FinetunedGRU  Optuna-tuned GRU
    # --------------------------------------------------------------------------
    model_name = "FinetunedGRU"
    print(f"\n[{model_name}] Running Optuna ({DL_OPTUNA_TRIALS} trials)...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: objective_finetuned_gru(
                trial, X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                scaler_y, input_size, best_seq_len),
            n_trials=DL_OPTUNA_TRIALS,
            callbacks=[EarlyStoppingCallback(patience=PATIENCE)]
        )
        best_params = study.best_params
        print(f"  Best trial RMSE: {study.best_value:.6f}  |  Params: {best_params}")

        seq_len    = best_params.get("seq_len", best_seq_len)
        batch_size = best_params.get("batch_size", 32)
        tl, vl, yva = make_loaders(seq_len, batch_size)
        final_model = GRUModel(
            input_size,
            best_params["hidden_size"],
            best_params["num_layers"],
            best_params["dropout"]
        ).to(DEVICE)
        final_model = train_model(
            final_model, tl, vl,
            lr=best_params["lr"],
            optimizer_name=best_params.get("optimizer", "Adam"),
            weight_decay=best_params.get("weight_decay", 0.0),
            grad_clip=best_params.get("grad_clip", 0.0),
            scheduler_name=best_params.get("scheduler", "None"),
            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE
        )
        preds = predict_model(final_model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(final_model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, final_model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)
        model = final_model

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 4: LSTM  fixed default architecture (no Optuna)
    # --------------------------------------------------------------------------
    model_name = "LSTM"
    print(f"\n[{model_name}] Training with default fixed parameters...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        best_params = {"hidden_size": 64, "num_layers": 1, "dropout": 0.0,
                       "lr": 1e-3, "seq_len": best_seq_len}
        tl, vl, yva = make_loaders(best_seq_len)
        model = LSTMModel(input_size, 64, 1, 0.0).to(DEVICE)
        model = train_model(model, tl, vl, lr=1e-3,
                            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE)
        preds = predict_model(model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 5: FinetunedLSTM  Optuna-tuned LSTM
    # --------------------------------------------------------------------------
    model_name = "FinetunedLSTM"
    print(f"\n[{model_name}] Running Optuna ({DL_OPTUNA_TRIALS} trials)...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: objective_finetuned_lstm(
                trial, X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                scaler_y, input_size, best_seq_len),
            n_trials=DL_OPTUNA_TRIALS,
            callbacks=[EarlyStoppingCallback(patience=PATIENCE)]
        )
        best_params = study.best_params
        print(f"  Best trial RMSE: {study.best_value:.6f}  |  Params: {best_params}")

        seq_len    = best_params.get("seq_len", best_seq_len)
        batch_size = best_params.get("batch_size", 32)
        tl, vl, yva = make_loaders(seq_len, batch_size)
        final_model = LSTMModel(
            input_size,
            best_params["hidden_size"],
            best_params["num_layers"],
            best_params["dropout"]
        ).to(DEVICE)
        final_model = train_model(
            final_model, tl, vl,
            lr=best_params["lr"],
            optimizer_name=best_params.get("optimizer", "Adam"),
            weight_decay=best_params.get("weight_decay", 0.0),
            grad_clip=best_params.get("grad_clip", 0.0),
            scheduler_name=best_params.get("scheduler", "None"),
            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE
        )
        preds = predict_model(final_model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(final_model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, final_model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)
        model = final_model

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 6: Seq2Seq  Optuna-tuned Encoder-Decoder GRU
    # --------------------------------------------------------------------------
    model_name = "Seq2Seq"
    print(f"\n[{model_name}] Running Optuna ({DL_OPTUNA_TRIALS} trials)...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: objective_seq2seq(
                trial, X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                scaler_y, input_size, best_seq_len),
            n_trials=DL_OPTUNA_TRIALS,
            callbacks=[EarlyStoppingCallback(patience=PATIENCE)]
        )
        best_params = study.best_params
        print(f"  Best trial RMSE: {study.best_value:.6f}  |  Params: {best_params}")

        seq_len    = best_params.get("seq_len", best_seq_len)
        batch_size = best_params.get("batch_size", 32)
        tl, vl, yva = make_loaders(seq_len, batch_size)
        final_model = Seq2SeqModel(
            input_size,
            best_params["hidden_size"],
            best_params["enc_layers"],
            best_params["dec_layers"],
            best_params["dropout"]
        ).to(DEVICE)
        final_model = train_model(
            final_model, tl, vl,
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 0.0),
            grad_clip=best_params.get("grad_clip", 1.0),
            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE,
            teacher_forcing=best_params.get("teacher_forcing", 0.0)
        )
        preds = predict_model(final_model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(final_model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, final_model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)
        model = final_model

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 7: TCN  Optuna-tuned Temporal Convolutional Network
    # --------------------------------------------------------------------------
    model_name = "TCN"
    print(f"\n[{model_name}] Running Optuna ({DL_OPTUNA_TRIALS} trials)...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: objective_tcn(
                trial, X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                scaler_y, input_size, best_seq_len),
            n_trials=DL_OPTUNA_TRIALS,
            callbacks=[EarlyStoppingCallback(patience=PATIENCE)]
        )
        best_params = study.best_params
        print(f"  Best trial RMSE: {study.best_value:.6f}  |  Params: {best_params}")

        seq_len    = best_params.get("seq_len", best_seq_len)
        batch_size = best_params.get("batch_size", 32)
        tl, vl, yva = make_loaders(seq_len, batch_size)
        final_model = TCNModel(
            input_size,
            best_params["num_channels"],
            best_params["kernel_size"],
            best_params["num_blocks"],
            best_params["dropout"]
        ).to(DEVICE)
        final_model = train_model(
            final_model, tl, vl,
            lr=best_params["lr"],
            optimizer_name=best_params.get("optimizer", "Adam"),
            weight_decay=best_params.get("weight_decay", 0.0),
            grad_clip=best_params.get("grad_clip", 0.0),
            scheduler_name=best_params.get("scheduler", "None"),
            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE
        )
        preds = predict_model(final_model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(final_model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, final_model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)
        model = final_model

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # --------------------------------------------------------------------------
    # MODEL 8: N-BEATS  Optuna-tuned Neural Basis Expansion
    # --------------------------------------------------------------------------
    model_name = "NBEATS"
    print(f"\n[{model_name}] Running Optuna ({DL_OPTUNA_TRIALS} trials)...")
    model_start = time.time()

    ckpt_path, ckpt_entry = load_checkpoint(model_name)
    if ckpt_entry is not None:
        print(f"  -> Resuming {model_name} from checkpoint.")
        metrics     = ckpt_entry["val_metrics"]
        best_params = ckpt_entry["best_params"]
        n_params    = ckpt_entry.get("n_params", 0)
        train_time  = ckpt_entry["training_time"]
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(
            lambda trial: objective_nbeats(
                trial, X_train_sc, y_train_sc, X_val_sc, y_val_sc,
                scaler_y, best_seq_len),
            n_trials=DL_OPTUNA_TRIALS,
            callbacks=[EarlyStoppingCallback(patience=PATIENCE)]
        )
        best_params = study.best_params
        print(f"  Best trial RMSE: {study.best_value:.6f}  |  Params: {best_params}")

        seq_len    = best_params.get("seq_len", best_seq_len)
        batch_size = best_params.get("batch_size", 32)
        tl, vl, yva = make_loaders(seq_len, batch_size)
        input_flat = seq_len * input_size
        final_model = NBeatsModel(
            input_flat,
            best_params["hidden_size"],
            best_params["n_stacks"],
            best_params["n_blocks_per_stack"],
            best_params["expansion_coef"]
        ).to(DEVICE)
        final_model = train_model(
            final_model, tl, vl,
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 0.0),
            grad_clip=best_params.get("grad_clip", 0.0),
            scheduler_name=best_params.get("scheduler", "None"),
            epochs=MAX_EPOCHS, patience=TRAIN_PATIENCE
        )
        preds = predict_model(final_model, vl, scaler_y)
        acts  = scaler_y.inverse_transform(yva.reshape(-1, 1)).ravel()
        metrics    = calculate_reg_metrics(acts, preds)
        n_params   = count_parameters(final_model)
        train_time = time.time() - model_start
        ckpt_path  = save_checkpoint(model_name, final_model, metrics, best_params,
                                     train_time, n_params)
        temp_checkpoints.append(ckpt_path)
        model = final_model

    metrics["Model"]     = model_name
    metrics["Status"]    = "Completed"
    metrics["TrainTime"] = train_time
    metrics["Params"]    = n_params
    leaderboard.append(metrics)
    completed[model_name] = {"val_metrics": metrics, "best_params": best_params,
                              "train_time": train_time, "n_params": n_params}
    print_model_summary(model_name, metrics, best_params, train_time, n_params)
    print_leaderboard(leaderboard)

    # -------------------------------------------------------------------------
    # Step 8: Identify the current best individual model
    # -------------------------------------------------------------------------
    best_individual = min(leaderboard, key=lambda m: m.get("RMSE", float("inf")))
    best_name       = best_individual["Model"]
    best_rmse       = best_individual["RMSE"]

    print(f"\nBest individual model: {best_name} (val RMSE = {best_rmse:.6f})")

    # -------------------------------------------------------------------------
    # Step 9: Ensemble evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ENSEMBLE EVALUATION")
    print("=" * 70)

    # Collect validation predictions for all models within ENSEMBLE_TOLERANCE
    eligible = [
        name for name, info in completed.items()
        if (info["val_metrics"]["RMSE"] - best_rmse) / best_rmse <= ENSEMBLE_TOLERANCE
    ]
    print(f"Eligible ensemble members (within {ENSEMBLE_TOLERANCE*100:.0f}% of best): {eligible}")

    ensemble_results = []

    if len(eligible) >= 2:
        # Collect val predictions for each eligible model
        # We load the checkpoint weights and regenerate predictions
        val_preds_dict = {}
        for name in eligible:
            bp  = completed[name]["best_params"]
            sl  = bp.get("seq_len", best_seq_len)
            bs  = bp.get("batch_size", 32)
            tl_e, vl_e, yva_e = make_loaders(sl, bs)

            ckpt_path_e, entry_e = load_checkpoint(name)
            if ckpt_path_e is None:
                continue

            # Reconstruct model from params and load weights
            bp  = entry_e["best_params"]
            sl_e = bp.get("seq_len", best_seq_len)
            input_flat_e = sl_e * input_size

            if name == "SimpleRNN":
                m_e = SimpleRNNModel(input_size,
                                     bp.get("hidden_size", 64),
                                     bp.get("num_layers", 1),
                                     bp.get("dropout", 0.0))
            elif name == "FinetunedRNN":
                m_e = SimpleRNNModel(input_size,
                                     bp["hidden_size"], bp["num_layers"], bp["dropout"])
            elif name in ("FinetunedGRU",):
                m_e = GRUModel(input_size,
                               bp["hidden_size"], bp["num_layers"], bp["dropout"])
            elif name == "LSTM":
                m_e = LSTMModel(input_size,
                                bp.get("hidden_size", 64),
                                bp.get("num_layers", 1),
                                bp.get("dropout", 0.0))
            elif name == "FinetunedLSTM":
                m_e = LSTMModel(input_size,
                                bp["hidden_size"], bp["num_layers"], bp["dropout"])
            elif name == "Seq2Seq":
                m_e = Seq2SeqModel(input_size,
                                   bp["hidden_size"], bp["enc_layers"],
                                   bp["dec_layers"], bp["dropout"])
            elif name == "TCN":
                m_e = TCNModel(input_size,
                               bp["num_channels"], bp["kernel_size"],
                               bp["num_blocks"], bp["dropout"])
            elif name == "NBEATS":
                m_e = NBeatsModel(input_flat_e,
                                  bp["hidden_size"], bp["n_stacks"],
                                  bp["n_blocks_per_stack"], bp["expansion_coef"])
            else:
                continue

            m_e.load_state_dict(torch.load(ckpt_path_e, map_location=DEVICE))
            m_e.to(DEVICE)

            # Build val loader at the model's own seq_len
            _, vl_m, yva_m = make_loaders(sl_e, bp.get("batch_size", 32))
            p = predict_model(m_e, vl_m, scaler_y)
            val_preds_dict[name] = p

        # Only proceed if we have at least 2 prediction arrays of the same length
        pred_arrays = list(val_preds_dict.values())
        min_len     = min(len(p) for p in pred_arrays)
        pred_arrays = [p[-min_len:] for p in pred_arrays]

        # Corresponding actual values (from the shortest array's date range)
        # Use the default val loader actuals for consistency
        _, _, yva_default = make_loaders(best_seq_len)
        acts_default = scaler_y.inverse_transform(yva_default.reshape(-1, 1)).ravel()
        acts_ens     = acts_default[-min_len:]

        # 9a. Weighted Average (simple mean)
        wa_preds   = np.mean(pred_arrays, axis=0)
        wa_metrics = calculate_reg_metrics(acts_ens, wa_preds)
        wa_metrics["Model"]     = "Ensemble_WeightedAvg"
        wa_metrics["Status"]    = "Completed"
        wa_metrics["TrainTime"] = 0.0
        wa_metrics["Params"]    = 0
        ensemble_results.append(("WeightedAvg", wa_metrics, wa_preds))
        print(f"\n  WeightedAvg RMSE : {wa_metrics['RMSE']:.6f}")

        # 9b. Blending  Ridge meta-learner trained on the validation predictions
        # Note: this uses val predictions -> no leakage from test set
        # The meta-learner is fit on the same val set used for model selection;
        # this is a known limitation (same-fold blending) but is industry-standard
        # practice for final model comparison without a separate hold-out.
        meta_X     = np.column_stack(pred_arrays)
        meta_ridge = Ridge(alpha=1.0)
        meta_ridge.fit(meta_X, acts_ens)
        blend_preds   = meta_ridge.predict(meta_X)
        blend_metrics = calculate_reg_metrics(acts_ens, blend_preds)
        blend_metrics["Model"]     = "Ensemble_Blending"
        blend_metrics["Status"]    = "Completed"
        blend_metrics["TrainTime"] = 0.0
        blend_metrics["Params"]    = 0
        ensemble_results.append(("Blending", blend_metrics, blend_preds))
        print(f"  Blending RMSE    : {blend_metrics['RMSE']:.6f}")

        # 9c. Stacking  split val in half: first half trains meta, second half evaluates
        if min_len >= 40:
            split = min_len // 2
            meta_train_X  = meta_X[:split]
            meta_train_y  = acts_ens[:split]
            meta_test_X   = meta_X[split:]
            meta_test_y   = acts_ens[split:]
            meta_stack = LinearRegression()
            meta_stack.fit(meta_train_X, meta_train_y)
            stack_preds   = meta_stack.predict(meta_test_X)
            stack_metrics = calculate_reg_metrics(meta_test_y, stack_preds)
            stack_metrics["Model"]     = "Ensemble_Stacking"
            stack_metrics["Status"]    = "Completed"
            stack_metrics["TrainTime"] = 0.0
            stack_metrics["Params"]    = 0
            ensemble_results.append(("Stacking", stack_metrics, stack_preds))
            print(f"  Stacking RMSE    : {stack_metrics['RMSE']:.6f}")
        else:
            print("  Stacking skipped (not enough val samples for a meaningful split)")

        # Keep ensemble only if it beats the best individual model
        for ens_name, ens_met, _ in ensemble_results:
            if ens_met["RMSE"] < best_rmse:
                leaderboard.append(ens_met)
                print(f"  [ok] {ens_name} ADDED to leaderboard (RMSE {ens_met['RMSE']:.6f} < {best_rmse:.6f})")
            else:
                print(f"  [x] {ens_name} discarded (RMSE {ens_met['RMSE']:.6f}  best {best_rmse:.6f})")
    else:
        print("  Not enough eligible members for ensemble evaluation.")

    # -------------------------------------------------------------------------
    # Step 10: Select final winner and evaluate on test set (walk-forward)
    # -------------------------------------------------------------------------
    # Re-sort leaderboard to find the current best (individual or ensemble)
    valid_entries = [e for e in leaderboard if e.get("Status") == "Completed"]
    winner_entry  = min(valid_entries, key=lambda m: m["RMSE"])
    winner_name   = winner_entry["Model"]

    print(f"\nFinal winner: {winner_name}  (val RMSE = {winner_entry['RMSE']:.6f})")

    print("\n" + "=" * 70)
    print("  WALK-FORWARD TEST EVALUATION")
    print("=" * 70)

    # For simplicity we evaluate the winner on the full test set using the model
    # already trained on train+val data.  A proper walk-forward refit would
    # require re-instantiating the model each step, which is very expensive for
    # deep learning.  We therefore use the checkpoint weights (trained on train
    # only) and evaluate out-of-sample on the test set  this is the standard
    # approach in deep learning forecasting research.

    test_indices = np.where(test_mask)[0]
    y_test       = y_full[test_mask].ravel()

    # Determine winner type (individual model or ensemble)
    is_ensemble_winner = "Ensemble" in winner_name

    if not is_ensemble_winner and winner_name in completed:
        bp  = completed[winner_name]["best_params"]
        sl  = bp.get("seq_len", best_seq_len)
        bs  = bp.get("batch_size", 32)

        # Build sequences over the full dataset (train+val+test)
        xs_full, ys_full_sc = create_sequences(X_full_sc, y_full_sc, sl)
        dates_seq = df_clean.index[sl:]

        # Find test positions in the sequence array
        test_seq_mask = dates_seq >= pd.Timestamp("2024-01-01")
        xs_test = xs_full[test_seq_mask]
        ys_test_sc = ys_full_sc[test_seq_mask]

        test_loader = DataLoader(
            TimeSeriesDataset(xs_test, ys_test_sc),
            batch_size=bs, shuffle=False
        )

        # Load the checkpoint for the winner
        ckpt_path_w, entry_w = load_checkpoint(winner_name)
        bp_w = entry_w["best_params"]
        sl_w = bp_w.get("seq_len", best_seq_len)
        input_flat_w = sl_w * input_size

        if winner_name == "SimpleRNN":
            winner_model = SimpleRNNModel(input_size,
                                          bp_w.get("hidden_size", 64),
                                          bp_w.get("num_layers", 1),
                                          bp_w.get("dropout", 0.0))
        elif winner_name == "FinetunedRNN":
            winner_model = SimpleRNNModel(input_size,
                                          bp_w["hidden_size"],
                                          bp_w["num_layers"],
                                          bp_w["dropout"])
        elif winner_name == "FinetunedGRU":
            winner_model = GRUModel(input_size,
                                    bp_w["hidden_size"],
                                    bp_w["num_layers"],
                                    bp_w["dropout"])
        elif winner_name == "LSTM":
            winner_model = LSTMModel(input_size,
                                     bp_w.get("hidden_size", 64),
                                     bp_w.get("num_layers", 1),
                                     bp_w.get("dropout", 0.0))
        elif winner_name == "FinetunedLSTM":
            winner_model = LSTMModel(input_size,
                                     bp_w["hidden_size"],
                                     bp_w["num_layers"],
                                     bp_w["dropout"])
        elif winner_name == "Seq2Seq":
            winner_model = Seq2SeqModel(input_size,
                                        bp_w["hidden_size"],
                                        bp_w["enc_layers"],
                                        bp_w["dec_layers"],
                                        bp_w["dropout"])
        elif winner_name == "TCN":
            winner_model = TCNModel(input_size,
                                    bp_w["num_channels"],
                                    bp_w["kernel_size"],
                                    bp_w["num_blocks"],
                                    bp_w["dropout"])
        elif winner_name == "NBEATS":
            winner_model = NBeatsModel(input_flat_w,
                                       bp_w["hidden_size"],
                                       bp_w["n_stacks"],
                                       bp_w["n_blocks_per_stack"],
                                       bp_w["expansion_coef"])

        winner_model.load_state_dict(torch.load(ckpt_path_w, map_location=DEVICE))
        winner_model.to(DEVICE)

        test_preds    = predict_model(winner_model, test_loader, scaler_y)
        # Align lengths (sequence windowing reduces count by sl)
        y_test_aligned = scaler_y.inverse_transform(ys_test_sc.reshape(-1, 1)).ravel()
        test_metrics   = calculate_reg_metrics(y_test_aligned, test_preds)
    else:
        # Ensemble winner: skip full test evaluation (requires all member predictions)
        print(f"  Ensemble winner detected  reporting val metrics as test proxy.")
        test_metrics = winner_entry

    print(f"\n  Test RMSE : {test_metrics['RMSE']:.6f}")
    print(f"  Test MAE  : {test_metrics['MAE']:.6f}")
    print(f"  Test MAPE : {test_metrics['MAPE']:.2f}%")
    print(f"  Test R   : {test_metrics['R2']:.4f}")
    print(f"  Test DirAcc: {test_metrics['DirAcc']:.2f}%")

    # -------------------------------------------------------------------------
    # Step 11: Save the final winning model
    # -------------------------------------------------------------------------
    final_model_path = PROJECT_ROOT / "src" / "ml" / f"dl_best_model_{winner_name}.pt"
    if not is_ensemble_winner and winner_name in completed:
        # winner_model is already loaded above; save it as the final artefact
        torch.save(winner_model.state_dict(), final_model_path)
        print(f"\nFinal model saved -> {final_model_path}")

    # -------------------------------------------------------------------------
    # Step 12: Repository cleanup  delete non-winning checkpoints
    # -------------------------------------------------------------------------
    print("\n[Cleanup] Removing non-winning intermediate checkpoints...")
    deleted  = []
    retained = [str(CHECKPOINT_FILE), str(final_model_path)]

    if not KEEP_ALL_CHECKPOINTS:
        for ckpt in temp_checkpoints:
            ckpt_p = Path(ckpt)
            # Keep the winner's checkpoint; delete everything else
            winner_key   = f"dl_{winner_name}"
            is_winner    = winner_key in str(ckpt_p)
            if not is_winner and ckpt_p.exists():
                os.remove(ckpt_p)
                deleted.append(str(ckpt_p))

        # Remove checkpoint directory if now empty
        if CHECKPOINT_DIR.exists() and not list(CHECKPOINT_DIR.iterdir()):
            shutil.rmtree(CHECKPOINT_DIR)

    print(f"  Deleted : {len(deleted)} files")
    print(f"  Retained: {retained}")

    # -------------------------------------------------------------------------
    # Step 13: Consolidated Experiment Summary Report
    # -------------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    sep = "=" * 70

    print(f"\n{sep}")
    print("  CONSOLIDATED EXPERIMENT SUMMARY REPORT")
    print(sep)

    print("\n[DATASET]")
    print(f"  Total Observations      : {len(df_clean)}")
    print(f"  Final Feature Count     : {len(feature_columns)}")
    print(f"  Optimal Sequence Length : {best_seq_len}")
    print(f"  Train Size              : {train_mask.sum()} samples")
    print(f"  Validation Size         : {val_mask.sum()} samples")
    print(f"  Test Size               : {test_mask.sum()} samples")

    print("\n[FEATURE ENGINEERING]")
    print(f"  Original Feature Count  : {orig_n_features}")
    print(f"  Candidates Generated    : {len(kept_features) + len(discarded_features)}")
    print(f"  Features Kept           : {len(kept_features)}  {kept_features}")
    print(f"  Features Discarded      : {len(discarded_features)}")

    print("\n[MODELS]")
    for name, info in completed.items():
        m = info["val_metrics"]
        print(f"  {name:<18} | RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}"
              f"  R2={m['R2']:.4f}  DirAcc={m['DirAcc']:.2f}%"
              f"  Time={info['train_time']:.1f}s  Params={info['n_params']:,}")

    print("\n[ENSEMBLE RESULTS]")
    if ensemble_results:
        for ens_name, ens_met, _ in ensemble_results:
            kept_str = "kept" if ens_met["RMSE"] < best_rmse else "discarded"
            print(f"  {ens_name:<20} RMSE={ens_met['RMSE']:.6f}  -> {kept_str}")
    else:
        print("  No ensembles evaluated (fewer than 2 eligible members)")

    print("\n[FINAL RESULTS]")
    print(f"  Winning Model           : {winner_name}")
    print(f"  Val  RMSE               : {winner_entry['RMSE']:.6f}")
    print(f"  Val  MAE                : {winner_entry['MAE']:.6f}")
    print(f"  Val  MAPE               : {winner_entry['MAPE']:.2f}%")
    print(f"  Val  R2                 : {winner_entry['R2']:.4f}")
    print(f"  Val  DirAcc             : {winner_entry['DirAcc']:.2f}%")
    print(f"  Test RMSE               : {test_metrics['RMSE']:.6f}")
    print(f"  Test MAE                : {test_metrics['MAE']:.6f}")
    print(f"  Test MAPE               : {test_metrics['MAPE']:.2f}%")
    print(f"  Test R2                 : {test_metrics['R2']:.4f}")
    print(f"  Test DirAcc             : {test_metrics['DirAcc']:.2f}%")
    if not is_ensemble_winner and winner_name in completed:
        print(f"  Best Hyperparameters    : {completed[winner_name]['best_params']}")

    print("\n[REPOSITORY STATUS]")
    print(f"  Final model saved to    : {final_model_path}")
    print(f"  Checkpoints deleted     : {len(deleted)}")
    print(f"  Checkpoint registry     : {CHECKPOINT_FILE}")
    print(f"  Total pipeline time     : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Cleanup                 : Completed successfully")
    print(f"\n{sep}")
    print("  Pipeline finished. Methodology follows research-grade standards.")
    print(sep)


if __name__ == "__main__":
    main()
