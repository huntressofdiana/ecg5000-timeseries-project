import time

import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# xLSTM imports
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)


# ============================================================
# DATASET
# ============================================================

class ECGDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
    ):
        self.signals = torch.tensor(
            signals,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        # Number of complete ECG signals
        return len(self.signals)

    def __getitem__(self, idx):

        # One complete ECG waveform
        # Original shape = (140,)
        x = self.signals[idx]

        # Add feature dimension
        # (140,) --> (140, 1)
        x = x.unsqueeze(-1)

        # ECG classes are originally 1-5
        # CrossEntropyLoss requires 0-4
        y = self.labels[idx] - 1

        return x, y


# ============================================================
# mLSTM MODEL
# ============================================================

class MLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_blocks: int,
        num_heads: int,
        num_classes: int,
        context_length: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
    ):
        super().__init__()

        # Project the single ECG feature into d_model dimensions
        #
        # (B, 140, 1)
        #       ↓
        # (B, 140, d_model)

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )


        # ----------------------------------------------------
        # Configure mLSTM
        # ----------------------------------------------------

        xlstm_config = xLSTMBlockStackConfig(

            # Only mLSTM blocks are being used
            mlstm_block=mLSTMBlockConfig(

                mlstm=mLSTMLayerConfig(

                    # Local temporal context
                    conv1d_kernel_size=conv1d_kernel_size,

                    # Feature grouping for Q/K/V projections
                    qkv_proj_blocksize=qkv_proj_blocksize,

                    # Number of mLSTM heads
                    num_heads=num_heads,
                )
            ),

            # Number of ECG time steps
            context_length=context_length,

            # Number of stacked mLSTM blocks
            num_blocks=num_blocks,

            # Internal representation size
            embedding_dim=d_model,
        )


        # mLSTM encoder
        self.encoder = xLSTMBlockStack(
            xlstm_config
        )


        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        # mLSTM output:
        #
        # (B, context_length, d_model)
        #
        # Flatten:
        #
        # (B, context_length * d_model)
        #
        # Then classify into the 5 ECG classes.

        self.output_projection = nn.Linear(
            d_model * context_length,
            num_classes,
        )


    def forward(self, x):

        # Input:
        # (B, 140, 1)

        x = self.input_projection(x)

        # After projection:
        # (B, 140, d_model)

        z = self.encoder(x)

        # mLSTM output:
        # (B, 140, d_model)

        z_flat = z.flatten(
            start_dim=1
        )

        # (B, 140 * d_model)

        prediction = self.output_projection(
            z_flat
        )

        # (B, 5)

        return prediction


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    true_indices: np.ndarray,
    predicted_indices: np.ndarray,
) -> None:

    class_names = [
        "I",
        "II",
        "III",
        "IV",
        "V",
    ]

    cm = confusion_matrix(
        true_indices,
        predicted_indices,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    plt.title(
        "Confusion Matrix – ECG5000 mLSTM"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    torch.manual_seed(42)
    np.random.seed(42)


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    train_path = "dataProcessed/ECG5000_FOLD_1_TRAIN_3200.txt"

    val_path = "dataProcessed/ECG5000_FOLD_1_800.txt"

    test_path = "dataProcessed/ECG5000_FINAL_TEST_1000.txt"


    train_data = np.loadtxt(
        train_path
    )

    val_data = np.loadtxt(
        val_path
    )

    test_data = np.loadtxt(
        test_path
    )


    # --------------------------------------------------------
    # SEPARATE LABELS AND ECG SIGNALS
    # --------------------------------------------------------

    # First column = class label

    y_train = train_data[:, 0].astype(int)

    y_val = val_data[:, 0].astype(int)

    y_test = test_data[:, 0].astype(int)


    # Remaining 140 columns = ECG waveform

    X_train = train_data[:, 1:]

    X_val = val_data[:, 1:]

    X_test = test_data[:, 1:]


    print(
        "\nTraining:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )


    print(
        "\nTraining labels:",
        y_train.shape
    )

    print(
        "Validation labels:",
        y_val.shape
    )

    print(
        "Test labels:",
        y_test.shape
    )


    # ========================================================
    # STANDARD SCALING
    # ========================================================

    scaler = StandardScaler()


    # Fit ONLY using training data

    X_train = scaler.fit_transform(
        X_train
    )


    # Apply the same scaler to validation data

    X_val = scaler.transform(
        X_val
    )


    # Apply the same scaler to test data

    X_test = scaler.transform(
        X_test
    )


    # ========================================================
    # CREATE DATASETS
    # ========================================================

    train_dataset = ECGDataset(
        signals=X_train,
        labels=y_train,
    )

    val_dataset = ECGDataset(
        signals=X_val,
        labels=y_val,
    )

    test_dataset = ECGDataset(
        signals=X_test,
        labels=y_test,
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_counts = np.bincount(
        y_train.astype(int)
    )[1:]


    class_weights = np.sqrt(
        len(y_train)
        /
        (
            len(class_counts)
            * class_counts
        )
    )


    class_weights = np.clip(
        class_weights,
        a_min=None,
        a_max=10.0,
    )


    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    ).to(device)


    print(
        "\nClass counts:",
        class_counts
    )

    print(
        "Class weights:",
        class_weights
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    batch_size = 32


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


    # --------------------------------------------------------
    # CHECK DATA SHAPES
    # --------------------------------------------------------

    x, y = train_dataset[0]

    print(
        "\nIndividual ECG shape:",
        x.shape
    )

    print(
        "Individual ECG label:",
        y
    )


    x_batch, y_batch = next(
        iter(train_loader)
    )


    print(
        "Batch input shape:",
        x_batch.shape
    )

    print(
        "Batch label shape:",
        y_batch.shape
    )


    # ========================================================
    # mLSTM HYPERPARAMETERS
    # ========================================================

    # ECG has one feature at each timestep
    input_size = 1


    # Internal representation dimension
    d_model = 64


    # Number of stacked mLSTM blocks
    num_blocks = 1


    # Number of mLSTM heads
    num_heads = 4


    # ECG5000 has five classes
    num_classes = 5


    # ECG5000 has 140 time steps
    context_length = X_train.shape[1]


    # Local temporal context
    conv1d_kernel_size = 4


    # Feature grouping for Q/K/V projections
    qkv_proj_blocksize = 4


    # ========================================================
    # INITIALISE MODEL
    # ========================================================

    model = MLSTMClassifier(
        input_size=input_size,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        num_classes=num_classes,
        context_length=context_length,
        conv1d_kernel_size=conv1d_kernel_size,
        qkv_proj_blocksize=qkv_proj_blocksize,
    ).to(device)


    # --------------------------------------------------------
    # NUMBER OF PARAMETERS
    # --------------------------------------------------------

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        f"\nModel has {num_params:,} "
        f"trainable parameters."
    )


    # --------------------------------------------------------
    # CHECK MODEL OUTPUT
    # --------------------------------------------------------

    x_batch = x_batch.to(device)

    outputs = model(
        x_batch
    )


    print(
        "Model output shape:",
        outputs.shape
    )

    # Expected:
    # (batch_size, 5)


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    y_batch = y_batch.to(device)


    loss = criterion(
        outputs,
        y_batch,
    )


    print(
        "Initial loss:",
        loss.item()
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    learning_rate = 0.0005


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    epochs = 30


    train_losses = []

    val_losses = []

    time_per_epoch = []


    best_val_loss = float("inf")

    best_val_epoch = 0


    for epoch in range(epochs):

        # ----------------------------------------------------
        # START EPOCH TIMER
        # ----------------------------------------------------

        start_time = time.time()


        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        train_loss = 0.0


        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)


            optimizer.zero_grad()


            outputs = model(
                x_batch
            )


            loss = criterion(
                outputs,
                y_batch,
            )


            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )


            optimizer.step()


            train_loss += loss.item()


        train_loss /= len(
            train_loader
        )


        train_losses.append(
            train_loss
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0


        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)

                y_batch = y_batch.to(device)


                outputs = model(
                    x_batch
                )


                loss = criterion(
                    outputs,
                    y_batch,
                )


                val_loss += loss.item()


        val_loss /= len(
            val_loader
        )


        val_losses.append(
            val_loss
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_epoch = epoch


            torch.save(
                model.state_dict(),
                "best_mlstm_ecg.pth",
            )


        # ====================================================
        # EPOCH TIME
        # ====================================================

        end_time = time.time()


        epoch_time = (
            end_time
            - start_time
        )


        time_per_epoch.append(
            epoch_time
        )


        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Best Val Loss: {best_val_loss:.6f} | "
            f"Best Epoch: {best_val_epoch + 1} | "
            f"Time: {epoch_time:.2f}s"
        )


    # ========================================================
    # AVERAGE TIME PER EPOCH
    # ========================================================

    average_time_per_epoch = (
        sum(time_per_epoch)
        / len(time_per_epoch)
    )


    print(
        f"\nAverage time per epoch: "
        f"{average_time_per_epoch:.2f} seconds"
    )


    # ========================================================
    # PLOT TRAINING / VALIDATION LOSS
    # ========================================================

    plt.plot(
        train_losses,
        label="Training Loss",
    )


    plt.plot(
        val_losses,
        label="Validation Loss",
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.title(
        "mLSTM Training and Validation Loss"
    )


    plt.legend()

    plt.show()


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    best_model = MLSTMClassifier(
        input_size=input_size,
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        num_classes=num_classes,
        context_length=context_length,
        conv1d_kernel_size=conv1d_kernel_size,
        qkv_proj_blocksize=qkv_proj_blocksize,
    ).to(device)


    best_model.load_state_dict(
        torch.load(
            "best_mlstm_ecg.pth",
            map_location=device,
        )
    )


    # ========================================================
    # TESTING
    # ========================================================

    best_model.eval()


    all_true = []

    all_preds = []


    with torch.no_grad():

        for x_batch, y_batch in test_loader:

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)


            outputs = best_model(
                x_batch
            )


            predicted_classes = torch.argmax(
                outputs,
                dim=1,
            )


            all_true.extend(
                y_batch.cpu().numpy()
            )


            all_preds.extend(
                predicted_classes.cpu().numpy()
            )


    all_true = np.array(
        all_true
    )


    all_preds = np.array(
        all_preds
    )


    # ========================================================
    # EVALUATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_true,
        all_preds,
    )


    precision = precision_score(
        all_true,
        all_preds,
        average="macro",
        zero_division=0,
    )


    recall = recall_score(
        all_true,
        all_preds,
        average="macro",
        zero_division=0,
    )


    f1_macro = f1_score(
        all_true,
        all_preds,
        average="macro",
        zero_division=0,
    )


    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )


    print(
        f"Macro Precision: {precision:.4f}"
    )


    print(
        f"Macro Recall: {recall:.4f}"
    )


    print(
        f"Macro F1: {f1_macro:.4f}"
    )


    print(
        f"Average Time Per Epoch: "
        f"{average_time_per_epoch:.2f} seconds"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    plot_confusion_matrix(
        all_true,
        all_preds,
    )


if __name__ == "__main__":
    main()