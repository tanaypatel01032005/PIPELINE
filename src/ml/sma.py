from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"

TARGET = "usd_zar_logret_next"

df = pd.read_csv(DATA_PATH)

series = df[TARGET].dropna().reset_index(drop=True)

train_end = int(len(series) * 0.8)

train = series.iloc[:train_end]
test = series.iloc[train_end:]

validation_size = int(len(train) * 0.2)

train_part = train.iloc[:-validation_size]
validation = train.iloc[-validation_size:]

windows = [3, 5, 7, 10, 15, 20]

best_window = None
best_rmse = np.inf

for window in windows:

    history = train_part.tolist()
    predictions = []

    for actual in validation:

        prediction = np.mean(history[-window:])
        predictions.append(prediction)
        history.append(actual)

    rmse = np.sqrt(mean_squared_error(validation, predictions))

    if rmse < best_rmse:
        best_rmse = rmse
        best_window = window

history = train.tolist()
predictions = []

for actual in test:

    prediction = np.mean(history[-best_window:])
    predictions.append(prediction)
    history.append(actual)

predictions = np.array(predictions)
actual = test.values

mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predictions)
mape = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
r2 = r2_score(actual, predictions)

direction_actual = np.sign(np.diff(actual))
direction_pred = np.sign(np.diff(predictions))

directional_accuracy = (
    (direction_actual == direction_pred).mean() * 100
)

results = pd.DataFrame({
    "Actual": actual,
    "Predicted": predictions
})

results.to_csv(
    PROJECT_ROOT / "data" / "results" / "sma_predictions.csv",
    index=False
)

print(f"Best Window : {best_window}")
print(f"MSE         : {mse:.6f}")
print(f"RMSE        : {rmse:.6f}")
print(f"MAE         : {mae:.6f}")
print(f"MAPE        : {mape:.2f}%")
print(f"R2 Score    : {r2:.6f}")
print(f"Directional Accuracy : {directional_accuracy:.2f}%")