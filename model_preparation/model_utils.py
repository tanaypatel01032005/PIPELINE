"""
model_utils.py — Shared utilities for all model training scripts (Stages 5–8).

Functions: load_model_data, load_unscaled_test, compute_metrics, save_metrics_row,
           save_predictions, save_feature_importance, plot_actual_vs_predicted,
           plot_residuals, plot_feature_importance
"""

import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for script execution without a display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for _d in ["data/predictions", "data/results", "plots", "models/scalers"]:
    os.makedirs(_d, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_model_data(
    x_train_path="data/model_input/X_train_scaled.csv",
    x_test_path ="data/model_input/X_test_scaled.csv",
    y_train_path="data/model_input/y_train.csv",
    y_test_path ="data/model_input/y_test.csv",
):
    """Load and validate the four model-input CSVs. Returns (X_train, X_test, y_train, y_test)."""
    for path in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: '{path}'\nRun train_test_split.py and feature_scaling.py first.")

    X_train = pd.read_csv(x_train_path, index_col="Date", parse_dates=True)
    X_test  = pd.read_csv(x_test_path,  index_col="Date", parse_dates=True)
    y_train = pd.read_csv(y_train_path, index_col="Date", parse_dates=True).squeeze("columns")
    y_test  = pd.read_csv(y_test_path,  index_col="Date", parse_dates=True).squeeze("columns")

    assert X_train.isna().sum().sum() == 0, "NaN in X_train"
    assert X_test.isna().sum().sum()  == 0, "NaN in X_test"
    assert y_train.isna().sum()       == 0, "NaN in y_train"
    assert y_test.isna().sum()        == 0, "NaN in y_test"
    assert list(X_train.columns) == list(X_test.columns), "Feature columns differ between X_train and X_test."
    assert X_train.index.max() < X_test.index.min(), "Data leakage: training and test date ranges overlap."

    _log.info(f"Data loaded — X_train {X_train.shape}  X_test {X_test.shape}  y_train {y_train.shape}  y_test {y_test.shape}")
    return X_train, X_test, y_train, y_test


def load_unscaled_test(x_test_path="data/model_input/X_test.csv"):
    """Load unscaled X_test for the persistence model (needs raw log-return values)."""
    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"Unscaled X_test not found: '{x_test_path}'\nRun train_test_split.py first.")
    return pd.read_csv(x_test_path, index_col="Date", parse_dates=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name):
    """
    Compute MAE, RMSE, MSE, R², MAPE, and Directional Accuracy.
    MAPE uses (|actual| + 1e-8) denominator to handle near-zero log returns.
    Dir_Acc = % of predictions with the correct sign (up/down direction).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true,  y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)

    return {
        "Model":       model_name,
        "MAE":         round(mae,     6),
        "RMSE":        round(rmse,    6),
        "MSE":         round(mse,     6),
        "R2":          round(r2,      6),
        "MAPE (%)":    round(mape,    4),
        "Dir_Acc (%)": round(dir_acc, 4),
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def save_metrics_row(metrics, csv_path):
    """Append or replace a model's metrics row in the shared CSV (no duplicates)."""
    new_row = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        combined    = pd.concat([df_existing[df_existing["Model"] != metrics["Model"]], new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(csv_path, index=False)
    _log.info(f"Metrics saved → {csv_path}")


def save_predictions(y_true, y_pred, model_name, save_dir="data/predictions"):
    """Save actual vs predicted values to CSV with Date index."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_predictions.csv")
    pd.DataFrame(
        {"Actual": y_true.values, "Predicted": np.asarray(y_pred).ravel()},
        index=y_true.index,
    ).rename_axis("Date").to_csv(path)
    _log.info(f"Predictions saved → {path}")
    return path


def save_feature_importance(feature_names, importances, model_name, save_dir="data/results"):
    """Save feature importance scores to CSV, sorted descending."""
    os.makedirs(save_dir, exist_ok=True)
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    )
    path = os.path.join(save_dir, f"{model_name}_feature_importance.csv")
    fi_df.to_csv(path, index=False)
    _log.info(f"Feature importance saved → {path}")
    return fi_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_fig(fig, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    _log.info(f"Plot saved → {save_path}")


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path):
    """Time-series line plot: actual vs predicted log returns."""
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(y_true.index, y_true.values, label="Actual",    linewidth=0.7, color="#2c7bb6", alpha=0.9)
    ax.plot(y_true.index, y_pred,        label="Predicted", linewidth=0.7, color="#d7191c", alpha=0.8)
    ax.set_title(f"{model_name} — Actual vs Predicted  |  USD/ZAR Log Return", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("USD/ZAR Log Return")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_residuals(y_true, y_pred, model_name, save_path):
    """Two-panel residual plot: residuals over time (left) + histogram (right)."""
    residuals = y_true.values - np.asarray(y_pred).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: residuals over time — helps detect heteroscedasticity and structural breaks
    axes[0].plot(y_true.index, residuals, linewidth=0.6, color="#7b2d8b", alpha=0.8)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0].set_title(f"{model_name} — Residuals Over Time", fontsize=12)
    axes[0].set_xlabel("Date"); axes[0].set_ylabel("Residual  (Actual − Predicted)")
    axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    # Right: histogram — should be approximately bell-shaped for a good model
    axes[1].hist(residuals, bins=60, color="#1a9641", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_title(f"{model_name} — Residual Distribution", fontsize=12)
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_feature_importance(feature_names, importances, model_name, save_path, top_n=20):
    """Horizontal bar chart of top-N features, sorted least→most important (bottom = highest)."""
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(fi_df) * 0.38)))
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="#4575b4", edgecolor="white", alpha=0.88)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=11)
    plt.tight_layout()
    _save_fig(fig, save_path)