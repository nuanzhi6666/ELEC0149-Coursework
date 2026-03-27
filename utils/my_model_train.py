import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


# ============================================================
# Utility
# ============================================================
def PrintLabelRatio(df: pd.DataFrame) -> None:
    """
    Print label distribution for classification task.
    """

    labels = df["target"].to_numpy(dtype=np.int64)

    unique = np.unique(labels)
    counts = np.array([(labels == u).sum() for u in unique])
    total = counts.sum()

    label_map = {-1: "Sell", 0: "Hold", 1: "Buy"}

    print("Label distribution:")
    for u, c in zip(unique, counts):
        ratio = c / total
        print(f"{label_map[int(u)]:>5}: {c:5d} ({ratio:.3f})")



# ============================================================
# Model
# ============================================================

class TradingNet(nn.Module):
    """
    LSTM-based trading model supporting both
    classification and regression tasks.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.

    hidden_dim : int
        Hidden dimension of LSTM.

    task_type : str
        "classification" or "regression".

    num_classes : int
        Number of classes for classification task.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        task_type: str = "classification",
        num_classes: int = 3,
    ):
        super().__init__()

        self.task_type = task_type

        # feature projection
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # temporal modeling
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # representation layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        # task-specific head
        if task_type == "classification":
            self.head = nn.Linear(hidden_dim, num_classes)
        elif task_type == "regression":
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError("task_type must be classification or regression")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Classification:
                (batch, num_classes)
            Regression:
                (batch,)
        """

        x = self.input_fc(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # use last timestep representation
        x = x[:, -1, :]
        x = self.fc(x)

        out = self.head(x)

        if self.task_type == "regression":
            return out.squeeze(-1)

        return out


# ============================================================
# Dataset construction
# ============================================================

def BuildSequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct sliding window sequences.

    Each sample uses previous `seq_len` timesteps
    to predict target at time t.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered dataframe.

    feature_cols : list[str]
        Feature column names.

    seq_len : int
        Length of input sequence.

    Returns
    -------
    X : np.ndarray
        Shape (N, seq_len, num_features)

    y : np.ndarray
        Shape (N,)
    """

    data = df[feature_cols].values
    target = df["target"].values

    X, y = [], []

    for i in range(seq_len, len(df)):
        X.append(data[i - seq_len : i])
        y.append(target[i])

    return np.array(X), np.array(y)


def PrepareDataloaders(
    dict_df: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    seq_len: int,
    batch_size: int,
    task_type: str = "classification",
):
    """
    Prepare PyTorch dataloaders for time-series learning.

    Supports both classification and regression.

    Parameters
    ----------
    dict_df : dict
        {"train": df, "val": df, "test": df}

    feature_cols : list[str]
        Input feature columns.

    seq_len : int
        Sliding window length.

    batch_size : int
        Batch size.

    task_type : str
        "classification" or "regression"

    Returns
    -------
    train_loader, val_loader, test_loader
    """

    X_train, y_train = BuildSequences(dict_df["train"], feature_cols, seq_len)
    X_val, y_val     = BuildSequences(dict_df["val"], feature_cols, seq_len)
    X_test, y_test   = BuildSequences(dict_df["test"], feature_cols, seq_len)

    if task_type == "classification":
        label_map = {-1: 0, 0: 1, 1: 2}

        y_train = np.vectorize(label_map.get)(y_train)
        y_val   = np.vectorize(label_map.get)(y_val)
        y_test  = np.vectorize(label_map.get)(y_test)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val   = torch.tensor(y_val, dtype=torch.long)
        y_test  = torch.tensor(y_test, dtype=torch.long)

    else:
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val   = torch.tensor(y_val, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), y_train
    )

    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), y_val
    )

    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), y_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Training / Evaluation
# ============================================================

def TrainOneEpoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def Evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    task_type: str,
    theta: float = 0.002,
):
    """
    Evaluation function.

    Classification:
        returns loss, accuracy

    Regression:
        returns loss, decision accuracy
        (after threshold-based discretization)
    """

    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(dataloader)

    # ===============================
    # classification task
    # ===============================
    if task_type == "classification":
        pred_cls = np.argmax(preds, axis=1)
        acc = (pred_cls == labels).mean()
        return avg_loss, acc

    # ===============================
    # regression → decision accuracy
    # ===============================
    pred_action = ReturnToAction(preds, theta)
    true_action = ReturnToAction(labels, theta)

    acc = (pred_action == true_action).mean()

    return avg_loss, acc


def GetPredictions(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)

            #  classification: logits → class
            if pred.ndim == 2:
                pred = pred.argmax(dim=1)

            preds.append(pred.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(labels), np.concatenate(preds)


def ReturnToAction(r: np.ndarray, theta: float):
    """
    Map return to trading action.

    r >  theta  →  1 (Buy)
    r < -theta  → -1 (Sell)
    else        →  0 (Hold)
    """
    action = np.zeros_like(r, dtype=np.int64)
    action[r >  theta] = 1
    action[r < -theta] = -1
    return action
