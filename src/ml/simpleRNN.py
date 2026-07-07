from pathlib import Path

import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from config import DEVICE
device = DEVICE

PROJECT_ROOT = Path.cwd()

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_features.csv"
)

TARGET = "usd_zar_logret_next"

LOOKBACK = 12

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.001

df = pd.read_csv(DATA_PATH)
required_columns = ["Date", "usd_zar_logret_next"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")
df = df.drop(columns=["Date"])
if "Date" in df.columns:
    df = df.drop(columns=["Date"])

df = df.dropna().reset_index(drop=True)

feature_columns = [
    c for c in df.columns
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


def create_sequences(X, y, lookback):

    xs = []

    ys = []

    for i in range(len(X) - lookback):

        xs.append(
            X[i:i + lookback]
        )

        ys.append(
            y[i + lookback]
        )

    return (
        np.array(xs),
        np.array(ys)
    )


X_train_seq, y_train_seq = create_sequences(
    X_train,
    y_train,
    LOOKBACK
)

X_test_seq, y_test_seq = create_sequences(
    X_test,
    y_test,
    LOOKBACK
)


class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )


train_dataset = TimeSeriesDataset(
    X_train_seq,
    y_train_seq
)

test_dataset = TimeSeriesDataset(
    X_test_seq,
    y_test_seq
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training Samples :", len(train_dataset))
print("Testing Samples  :", len(test_dataset))
print("Features         :", X_train_seq.shape[2])
print("Device           :", device)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class SimpleRNNModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=1,
    ):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x,
    ):

        out, _ = self.rnn(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = SimpleRNNModel(
    input_size=X_train_seq.shape[2]
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_loss = float("inf")

patience = 10

counter = 0

best_model = copy.deepcopy(model.state_dict())

history = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

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

    # print(
    #     f"Epoch {epoch+1:03d} | Loss = {epoch_loss:.6f}"
    # )

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


model.load_state_dict(best_model)

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


predictions = np.array(predictions)

actuals = np.array(actuals)

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

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    actuals,
    predictions,
)

mape = np.mean(
    np.abs(
        (actuals - predictions)
        / (actuals + 1e-8)
    )
) * 100

r2 = r2_score(
    actuals,
    predictions,
)

direction_actual = np.sign(
    np.diff(actuals.flatten())
)

direction_pred = np.sign(
    np.diff(predictions.flatten())
)

directional_accuracy = (
    direction_actual
    == direction_pred
).mean() * 100

results = pd.DataFrame(
    {
        "Actual": actuals.flatten(),
        "Predicted": predictions.flatten(),
    }
)

# results.to_csv(
#     PROJECT_ROOT
#     / "data"
#     / "results"
#     / "rnn_predictions.csv",
#     index=False,
# )

# MODEL_PATH = (
#     PROJECT_ROOT
#     / "data"
#     / "results"
#     / "best_rnn_model.pth"
# )

# torch.save(model.state_dict(), MODEL_PATH)

# print(f"Model saved to: {MODEL_PATH}")

print()

print("=" * 50)

print("Simple RNN Results")

print("=" * 50)

print(f"MSE : {mse:.6f}")

print(f"RMSE : {rmse:.6f}")

print(f"MAE : {mae:.6f}")

print(f"MAPE : {mape:.2f}% (Interpret with caution for log returns)")

print(f"R2 Score : {r2:.6f}")

print(
    f"Directional Accuracy : {directional_accuracy:.2f}%"
)