from pathlib import Path

import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_features.csv"
)

TARGET = "usd_zar_logret_next"

LOOKBACK = 30

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.001

HIDDEN_SIZE = 256

N_BLOCKS = 4


df = pd.read_csv(DATA_PATH)

if "Date" in df.columns:
    df = df.drop(columns=["Date"])

df = df.dropna().reset_index(drop=True)

feature_columns = [
    c
    for c in df.columns
    if c != TARGET
]

X = df[feature_columns].values

y = df[TARGET].values.reshape(-1, 1)

train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

x_scaler = MinMaxScaler()

y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train)

X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train)

y_test = y_scaler.transform(y_test)


def create_sequences(
    X,
    y,
    lookback,
):

    xs = []

    ys = []

    for i in range(
        len(X) - lookback
    ):

        xs.append(
            X[i:i + lookback]
        )

        ys.append(
            y[i + lookback]
        )

    return (
        np.array(xs),
        np.array(ys),
    )


X_train_seq, y_train_seq = create_sequences(
    X_train,
    y_train,
    LOOKBACK,
)

X_test_seq, y_test_seq = create_sequences(
    X_test,
    y_test,
    LOOKBACK,
)


class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        X,
        y,
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(
        self,
        idx,
    ):

        return (
            self.X[idx],
            self.y[idx],
        )


train_dataset = TimeSeriesDataset(
    X_train_seq,
    y_train_seq,
)

test_dataset = TimeSeriesDataset(
    X_test_seq,
    y_test_seq,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class NBeatsBlock(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                input_size + 1,
            ),
        )

    def forward(
        self,
        x,
    ):

        theta = self.network(x)

        backcast = theta[:, :-1]

        forecast = theta[:, -1:]

        return (
            backcast,
            forecast,
        )


class NBeats(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_blocks,
    ):

        super().__init__()

        self.blocks = nn.ModuleList(

            [
                NBeatsBlock(
                    input_size,
                    hidden_size,
                )
                for _
                in range(n_blocks)
            ]
        )

    def forward(
        self,
        x,
    ):

        batch = x.size(0)

        residual = x.reshape(
            batch,
            -1,
        )

        forecast = torch.zeros(
            batch,
            1,
            device=x.device,
        )

        for block in self.blocks:

            backcast, block_forecast = block(
                residual
            )

            residual = residual - backcast

            forecast = (
                forecast
                + block_forecast
            )

        return forecast


INPUT_SIZE = (
    X_train_seq.shape[1]
    * X_train_seq.shape[2]
)

model = NBeats(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    n_blocks=N_BLOCKS,
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print(model)

print()

print(
    "Training Samples :",
    len(train_dataset),
)

print(
    "Testing Samples :",
    len(test_dataset),
)

print(
    "Flatten Input Size :",
    INPUT_SIZE,
)

print(
    "Device :",
    device,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

best_loss = float("inf")

patience = 10

counter = 0

best_model = copy.deepcopy(
    model.state_dict()
)

history = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    history.append(epoch_loss)

    print(
        f"Epoch {epoch + 1:03d} | Loss = {epoch_loss:.6f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        counter = 0

        best_model = copy.deepcopy(
            model.state_dict()
        )

    else:

        counter += 1

        if counter >= patience:

            print("Early stopping")

            break


model.load_state_dict(
    best_model
)

model.eval()

predictions = []

actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        predictions.extend(
            outputs.cpu().numpy()
        )

        actuals.extend(
            y_batch.numpy()
        )

predictions = np.array(
    predictions
)

actuals = np.array(
    actuals
)

predictions = y_scaler.inverse_transform(
    predictions
)

actuals = y_scaler.inverse_transform(
    actuals
)

mse = mean_squared_error(
    actuals,
    predictions,
)

rmse = np.sqrt(
    mse
)

mae = mean_absolute_error(
    actuals,
    predictions,
)

mape = np.mean(
    np.abs(
        (actuals - predictions)
        /
        (actuals + 1e-8)
    )
) * 100

r2 = r2_score(
    actuals,
    predictions,
)

direction_actual = np.sign(
    np.diff(
        actuals.flatten()
    )
)

direction_pred = np.sign(
    np.diff(
        predictions.flatten()
    )
)

directional_accuracy = (
    direction_actual
    ==
    direction_pred
).mean() * 100

results = pd.DataFrame(
    {
        "Actual": actuals.flatten(),
        "Predicted": predictions.flatten(),
    }
)

results.to_csv(
    PROJECT_ROOT
    / "data"
    / "results"
    / "nbeats_predictions.csv",
    index=False,
)

metrics = pd.DataFrame(
    {
        "Metric": [
            "MSE",
            "RMSE",
            "MAE",
            "MAPE",
            "R2",
            "Directional Accuracy",
        ],
        "Value": [
            mse,
            rmse,
            mae,
            mape,
            r2,
            directional_accuracy,
        ],
    }
)

metrics.to_csv(
    PROJECT_ROOT
    / "data"
    / "results"
    / "nbeats_metrics.csv",
    index=False,
)

loss_df = pd.DataFrame(
    {
        "Epoch": range(
            1,
            len(history) + 1,
        ),
        "Loss": history,
    }
)

loss_df.to_csv(
    PROJECT_ROOT
    / "data"
    / "results"
    / "nbeats_training_loss.csv",
    index=False,
)

torch.save(
    model.state_dict(),
    PROJECT_ROOT
    / "data"
    / "results"
    / "best_nbeats_model.pth",
)

print()

print("=" * 60)

print("N-BEATS Results")

print("=" * 60)

print(f"MSE : {mse:.6f}")

print(f"RMSE : {rmse:.6f}")

print(f"MAE : {mae:.6f}")

print(f"MAPE : {mape:.2f}%")

print(f"R2 Score : {r2:.6f}")

print(
    f"Directional Accuracy : {directional_accuracy:.2f}%"
)

print()

print("Files Saved")

# print(
#     PROJECT_ROOT
#     / "data"
#     / "results"
#     / "nbeats_predictions.csv"
# )

# print(
#     PROJECT_ROOT
#     / "data"
#     / "results"
#     / "nbeats_metrics.csv"
# )

# print(
#     PROJECT_ROOT
#     / "data"
#     / "results"
#     / "nbeats_training_loss.csv"
# )

# print(
#     PROJECT_ROOT
#     / "data"
#     / "results"
#     / "best_nbeats_model.pth"
# )