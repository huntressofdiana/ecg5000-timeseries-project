"""Comparison of mLSTM, sLSTM, and mixed xLSTM models for time-series classification.

This script applies the xLSTM architecture to the RacketSports multivariate
time-series classification dataset. Each sample contains accelerometer and
gyroscope measurements over a fixed sequence of time steps.

Three xLSTM variants are implemented and can be compared:

    1) mLSTM (matrix LSTM) -- uses matrix-valued associative memory with
       query, key, and value projections. The `qkv_proj_blocksize` controls
       the feature-group size used for the Q/K/V projections, while
       `conv1d_kernel_size` determines the local causal temporal context.

    2) sLSTM (scalar LSTM) -- uses recurrent scalar-memory cells with
       exponential gating and explicit hidden-state recurrence. Unlike mLSTM,
       it does not use Q/K/V projections. The implementation can use either
       the vanilla PyTorch backend or the optimized custom CUDA backend.

    3) xLSTM (mixed) -- combines mLSTM and sLSTM blocks within the same
       xLSTM block stack. In the implementation used here, the first block is
       an mLSTM block and the second block is an sLSTM block, allowing both
       associative memory and recurrent state tracking to be combined.

The same preprocessing, train/validation/test split, optimizer, loss function,
and evaluation metrics are used for all model variants so that their
classification performance and computational cost can be compared directly.
"""


import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)
import pandas as pd

class MLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        num_blocks: int,
        num_heads: int,
        context_length: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, # Local causal temporal context
                    qkv_proj_blocksize=qkv_proj_blocksize, # Feature-group size for Q/K/V projections
                    num_heads=num_heads,
                )
            ),
            context_length=context_length, # Number of time steps
            num_blocks=num_blocks, # Number of mLSTM blocks
            embedding_dim=d_model,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model*context_length,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_projection(x) # (B, T, input_size)

        z = self.encoder(x) # (B, T, d_model)

        z_flat = z.flatten(start_dim=1) # (bs, n_timesteps*d_model)

        output = self.output_projection(z_flat) #(bs, 4)
        return output


class SLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        num_blocks: int,
        num_heads: int,
        context_length: int,
        conv1d_kernel_size: int,
    ):
        super().__init__()

        # Project input features to model dimension
        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        # sLSTM configuration
        xlstm_config = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size, # Local causal temporal context
                    backend="vanilla", # sLSTM implementation: vanilla PyTorch or custom CUDA
                ),
            ),
            context_length=context_length, # Number of time steps
            num_blocks=num_blocks, # Number of sLSTM blocks
            embedding_dim=d_model,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model*context_length,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (B, T, input_size) -> (B, T, d_model)
        x = self.input_projection(x)

        # (B, T, d_model)
        z = self.encoder(x)

        # # Final sequence representation
        # z_last = z[:, -1, :]
        z_flat = z.flatten(start_dim=1) #  # (bs, n_timesteps*d_model)

        output = self.output_projection(z_flat) #(bs, 4)

        # # (B, n_classes)
        # output = self.output_projection(z_last)

        return output


class XLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        num_blocks: int,
        num_heads: int,
        context_length: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,   # Local causal temporal context
                    qkv_proj_blocksize=qkv_proj_blocksize,  # Feature-group size for Q/K/V projections
                    num_heads=num_heads,
                )
            ),

            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,  # Local causal temporal context
                    backend="vanilla",  # sLSTM implementation: vanilla PyTorch or custom CUDA
                ),
            ),

            context_length=context_length,  # Number of time steps
            num_blocks=num_blocks,                   # One mLSTM block + one sLSTM block
            embedding_dim=d_model,
            slstm_at=[1],                   # Block 0: mLSTM, Block 1: sLSTM
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model * context_length,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (B, T, input_size) -> (B, T, d_model)
        x = self.input_projection(x)

        # mLSTM -> sLSTM
        # (B, T, d_model) -> (B, T, d_model)
        z = self.encoder(x)

        # Flatten all temporal representations
        z_flat = z.flatten(start_dim=1)   # (bs, n_timesteps*d_model)

        # Classification logits
        output = self.output_projection(z_flat)  # (B, n_classes)

        return output


def plot_confusion_matrix(
    true_indices: np.ndarray, predicted_indices: np.ndarray
) -> None:
    class_names = [
        "Badminton Clear",
        "Badminton Smash",
        "Squash Forehand",
        "Squash Backhand",
    ]

    cm = confusion_matrix(
        true_indices,
        predicted_indices,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix – RacketSports")
    plt.tight_layout()
    plt.show()

def load_ecg5000(file_path):
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    labels = df.iloc[:, 0].to_numpy().astype(int) - 1
    data = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    # (samples, 140) -> (samples, 140, 1)
    data = data[..., np.newaxis]

    return data, labels


def plot_sample(data: np.ndarray, labels: np.ndarray, sample_idx: int) -> None:
    """Plot sample data and labels.
    Args:
        data (np.ndarray): Array of shape (n_examples, n_timesteps, n_features).
        labels (np.ndarray): Array of shape (n_examples,).
        sample_idx (int): Array of shape (n_examples,).
    """
    sample = data[sample_idx]
    label = labels[sample_idx]

    time = np.arange(sample.shape[0])

    channel_names = [
        "Accelerometer x",
        "Accelerometer y",
        "Accelerometer z",
        "Gyroscope x",
        "Gyroscope y",
        "Gyroscope z",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.ravel()

    for i in range(6):
        axes[i].plot(time, sample[:, i], linewidth=1.8)
        axes[i].set_title(channel_names[i], fontsize=11)
        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"Class: {label}", fontsize=14)
    plt.tight_layout()
    plt.show()


class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    print(torch.cuda.is_available()) # Checks if Nvidia GPU is available
    print("Cuda version:", torch.version.cuda) # Checks which cuda version is installed
    torch.manual_seed(42)

    ### Load the data
    train_data, train_labels = load_ecg5000("data/ECG5000_TRAIN.txt")
    plot_sample(train_data, train_labels, sample_idx=0)

    unique_labels = np.unique(train_labels)
    print(train_data.shape)
    print(unique_labels.shape)

    test_data, test_labels = load_ecg5000("data/ECG5000_TEST.txt")

    #### One Hot Encoding of the labels
    # ohe = OneHotEncoder()
    # train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1)).toarray()
    # test_labels_ohe = ohe.transform(test_labels.reshape(-1, 1)).toarray()

    train_data, val_data, train_labels, val_labels = train_test_split(
    train_data,
    train_labels,
    test_size=0.1,
    random_state=42,
    stratify=train_labels,
)

    ### Validation split
    train_data, val_data, train_labels_ohe, val_labels_ohe = train_test_split(
        train_data,
        train_labels_ohe,
        test_size=0.1,
        random_state=42,
    )

    ### Scaling of the signals
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        train_data.reshape(-1, train_data.shape[-1])
    ).reshape(train_data.shape[0], train_data.shape[1], train_data.shape[-1])
    val_scaled = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(
        val_data.shape[0], val_data.shape[1], val_data.shape[-1]
    )
    test_scaled = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(
        test_data.shape[0], test_data.shape[1], test_data.shape[-1]
    )

    ### Initialize the dataset
    train_dataset = ECGDataset(train_scaled, train_labels)
    val_dataset = ECGDataset(val_scaled, val_labels)
    test_dataset = ECGDataset(test_scaled, test_labels)

    bs = 32
    lr = 0.0001
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ### Initialize model and optimizer
    input_size = train_data.shape[-1]
    n_classes = train_labels_ohe.shape[1]
    d_model = 64
    num_blocks = 1 # set to 2 or more for xLSTM with mLSTM and sLSTM block
    num_heads = 4
    conv1d_kernel_size = 4
    qkv_proj_blocksize = 4 # Only for the mLSTM
    n_timesteps = train_data.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MLSTMClassifier(
    #     input_size=input_size,
    #     n_classes=n_classes,
    #     d_model=d_model,
    #     num_blocks=num_blocks,
    #     num_heads=num_heads,
    #     context_length=n_timesteps,
    #     conv1d_kernel_size=conv1d_kernel_size,
    #     qkv_proj_blocksize=qkv_proj_blocksize, # Only for MLSTM
    # ).to(device)

    model = SLSTMClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        context_length=n_timesteps,
        conv1d_kernel_size=conv1d_kernel_size,
    ).to(device)

    ### Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    epochs = 200

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    time_per_epoch = []

    for epoch in range(epochs):
        start_time = time.time() # Start time per epoch

        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(x_batch)

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        end_time = time.time()
        time_per_epoch.append(end_time - start_time) # Appends the time per epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f}"
            f"| Val loss: {val_loss:.8f}"
            f" | "
            f"Best Val loss: {best_val_loss:.8f}"
            f" | "
            f"Best Val epoch: {best_val_epoch + 1}"
        )

    ### Plot the loss function
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.ylabel("Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.legend()
    plt.show()

    ### Model Testing
    best_model = MLSTMClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        context_length=n_timesteps,
        conv1d_kernel_size=conv1d_kernel_size,
        qkv_proj_blocksize=qkv_proj_blocksize, # Only for MLSTM or XLSTM
    ).to(device)

    best_model.load_state_dict(torch.load("best_model.pth"))

    best_model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            logits = best_model(x_batch)

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    predicted_indices = all_logits.argmax(dim=1).numpy()
    true_indices = all_targets.numpy()

    ### Metrics
    accuracy = accuracy_score(
        true_indices,
        predicted_indices,
    )

    precision = precision_score(true_indices, predicted_indices, average="macro")
    recall = recall_score(true_indices, predicted_indices, average="macro")
    f1_macro = f1_score(true_indices, predicted_indices, average="macro")

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision (Macro): {precision:.2f}")
    print(f"Recall (Macro: {recall:.2f}")
    print(f"F1 (Macro): {f1_macro:.4f}")

    average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    print(f"Average time per epoch: {average_time_per_epoch:.4f}s")

    ### Confusion Matrix
    plot_confusion_matrix(true_indices, predicted_indices)


if __name__ == "__main__":
    main()