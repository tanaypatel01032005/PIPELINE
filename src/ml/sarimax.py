from pathlib import Path

import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"

df = pd.read_csv(DATA_PATH)

feature_cols = [
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

TARGET = "usd_zar_logret_next"

model_df = df[[TARGET] + feature_cols].dropna().reset_index(drop=True)

train_size = int(len(model_df) * 0.8)

train = model_df.iloc[:train_size]
test = model_df.iloc[train_size:]

y_train = train[TARGET]
y_test = test[TARGET]

X_train = train[feature_cols]
X_test = test[feature_cols]

best_aic = np.inf
best_order = None
best_model = None

for p, d, q in itertools.product(range(4), range(2), range(4)):

    try:

        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        result = model.fit(disp=False)

        if result.aic < best_aic:

            best_aic = result.aic
            best_order = (p, d, q)
            best_model = result

    except Exception:
        continue

forecast = best_model.predict(
    start=len(y_train),
    end=len(y_train) + len(y_test) - 1,
    exog=X_test,
)

forecast = np.array(forecast)
actual = y_test.values

mse = mean_squared_error(actual, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, forecast)
mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100
r2 = r2_score(actual, forecast)

direction_actual = np.sign(np.diff(actual))
direction_pred = np.sign(np.diff(forecast))

directional_accuracy = (
    (direction_actual == direction_pred).mean() * 100
)

results = pd.DataFrame(
    {
        "Actual": actual,
        "Predicted": forecast,
    }
)

results.to_csv(
    PROJECT_ROOT / "data" / "results" / "sarimax_predictions.csv",
    index=False,
)

print(f"Best Order : {best_order}")
print(f"AIC : {best_aic:.2f}")
print(f"MSE : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"MAPE : {mape:.2f}%")
print(f"R2 Score : {r2:.6f}")
print(f"Directional Accuracy : {directional_accuracy:.2f}%")