"""sLSTM classifier for the ECG5000 univariate time-series dataset.

Same data pipeline as ecg_mlstm_classifier.py, but the encoder is a stack of
sLSTM (scalar LSTM) blocks instead of mLSTM blocks. sLSTM uses recurrent
scalar-memory cells with exponential gating and explicit hidden-state
recurrence -- no Q/K/V projections, unlike mLSTM.

Expected data format: whitespace-separated files where column 0 is the class
label and columns 1..140 are the signal values (the standard ECG5000_TRAIN /
ECG5000_TEST layout from the UCR archive).
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)


class ECGDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        # signals: (N, T, 1) float32, labels: (N,) int64 already 0-indexed
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


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

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        xlstm_config = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,  # Local causal temporal context
                    backend="vanilla",  # sLSTM implementation: vanilla PyTorch or custom CUDA
                ),
            ),
            context_length=context_length,  # Number of time steps (140 for ECG5000)
            num_blocks=num_blocks,  # Number of sLSTM blocks
            embedding_dim=d_model,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        # Classification head
        self.output_projection = nn.Linear(
            d_model * context_length,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)  # (B, T, input_size) -> (B, T, d_model)
        z = self.encoder(x)  # (B, T, d_model)
        z_flat = z.flatten(start_dim=1)  # (B, T * d_model)
        output = self.output_projection(z_flat)  # (B, n_classes)
        return output


def load_ecg5000(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an ECG5000 TRAIN/TEST file.

    Args:
        file_path: Path to a whitespace-separated ECG5000 split file where
            column 0 is the class label and the remaining columns are the
            140 signal values.

    Returns:
        x (np.ndarray): Array of shape (n_examples, n_timesteps).
        y (np.ndarray): Labels, shape (n_examples,), as given in the file
            (ECG5000 uses 1-5).
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    y = df.iloc[:, 0].values.astype(np.int64)
    x = df.iloc[:, 1:].values.astype(np.float32)
    return x, y


def plot_sample(signal: np.ndarray, label: int, sample_idx: int = 0) -> None:
    """Plot a single ECG waveform."""
    plt.figure(figsize=(8, 3))
    plt.plot(signal, linewidth=1.5)
    plt.title(f"ECG5000 sample {sample_idx} - class {label}")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_indices: np.ndarray, predicted_indices: np.ndarray) -> None:
    class_names = ["I", "II", "III", "IV", "V"]

    cm = confusion_matrix(true_indices, predicted_indices)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix – ECG5000 (sLSTM)")
    plt.tight_layout()
    plt.show()


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    ### Load Fold 1 data ------------------------------------------------
    # Predefined 5-fold CV split:
    #   TRAIN_3200 -> training
    #   VAL_800    -> validation
    #   FINAL_TEST_1000 -> held-out final test set
    train_data, train_labels = load_ecg5000(
        "dataProcessed/ECG5000_FOLD_5_TRAIN_3200.txt"
    )
    val_data, val_labels = load_ecg5000(
        "dataProcessed/ECG5000_FOLD_5_VAL_800.txt"
    )
    test_data, test_labels = load_ecg5000(
        "dataProcessed/ECG5000_FINAL_TEST_1000.txt"
    )

    plot_sample(train_data[0], train_labels[0], sample_idx=0)

    print("Train data:", train_data.shape)
    print("Validation data:", val_data.shape)
    print("Test data:", test_data.shape)
    print("Unique labels:", np.unique(train_labels))

    # ECG5000 labels are 1-5; CrossEntropyLoss expects 0-indexed classes.
    train_labels = train_labels - 1
    val_labels = val_labels - 1
    test_labels = test_labels - 1

    ### Scaling of the signals ---------------------------------------------
    # Fit only on training data, reuse the same scaler for val/test.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
    val_scaled = scaler.transform(val_data.reshape(-1, 1)).reshape(val_data.shape)
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)

    # Add the feature dimension: (N, T) -> (N, T, 1)
    train_scaled = train_scaled[:, :, np.newaxis]
    val_scaled = val_scaled[:, :, np.newaxis]
    test_scaled = test_scaled[:, :, np.newaxis]

    ### Initialize the datasets --------------------------------------------
    train_dataset = ECGDataset(train_scaled, train_labels)
    val_dataset = ECGDataset(val_scaled, val_labels)
    test_dataset = ECGDataset(test_scaled, test_labels)

    ### Class weights (ECG5000 is heavily imbalanced) -----------------------
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = len(train_labels) / (len(class_counts) * np.clip(class_counts, 1, None))
    class_weights = np.clip(class_weights, a_min=None, a_max=10.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    bs = 32
    lr = 0.0005
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    ### Initialize model and optimizer --------------------------------------
    input_size = 1  # ECG5000 is a single-channel signal
    n_classes = 5
    d_model = 64
    num_blocks = 2
    num_heads = 4
    conv1d_kernel_size = 4
    context_length = train_data.shape[1]  # 140 time steps for ECG5000

    model = SLSTMClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        context_length=context_length,
        conv1d_kernel_size=conv1d_kernel_size,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    epochs = 30

    train_losses = []
    val_losses = []
    time_per_epoch = []
    best_val_loss = float("inf")
    best_val_epoch = 0

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time
        time_per_epoch.append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), "best_slstm_ecg.pth")

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Best val loss: {best_val_loss:.6f} | "
            f"Best val epoch: {best_val_epoch + 1} | "
            f"Time: {epoch_time:.2f}s"
        )

    average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    print(f"Average time per epoch: {average_time_per_epoch:.2f}s")

    ### Plot the loss function ---------------------------------------------
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.ylabel("Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.title("sLSTM Training / Validation Loss – ECG5000")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ### Model testing --------------------------------------------------------
    best_model = SLSTMClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        context_length=context_length,
        conv1d_kernel_size=conv1d_kernel_size,
    ).to(device)

    best_model.load_state_dict(torch.load("best_slstm_ecg.pth", map_location=device))
    best_model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = best_model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    ### Metrics -----------------------------------------------------------
    accuracy = accuracy_score(all_true, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_true, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")

    ### Confusion matrix -----------------------------------------------------
    plot_confusion_matrix(all_true, all_preds)


if __name__ == "__main__":
    main()