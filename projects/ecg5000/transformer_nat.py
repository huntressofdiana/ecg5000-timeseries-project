import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


import time

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


### Dataset
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
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]

        return x, y


### Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
    ):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return x


### Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        dim_ff: int,
        n_classes: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        n_tokens: int,
    ):
        super().__init__()

        # Convert ECG value at each timestep
        # from input_size = 1
        # to d_model dimensions
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        # Add information about where each ECG value
        # occurs in the 140 timestep sequence
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=n_tokens
        )

        # One Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )

        # Stack Transformer encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Flatten all 140 contextualised timestep representations
        # before classification
        self.output_projection = nn.Linear(
            d_model * n_tokens,
            n_classes
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # x:
        # (batch_size, 140, 1)

        x_embed = self.input_projection(x)

        # (batch_size, 140, d_model)

        x_embed_pe = self.positional_encoding(
            x_embed
        )

        # Transformer encoder
        z = self.encoder(
            x_embed_pe
        )

        # z:
        # (batch_size, 140, d_model)

        # Flatten timestep and feature dimensions
        z_flat = z.flatten(
            start_dim=1
        )

        # (batch_size, 140 * d_model)

        logits = self.output_projection(
            z_flat
        )

        # (batch_size, 5)

        return logits


### Confusion Matrix
def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
):
    class_names = [
        "Class 1",
        "Class 2",
        "Class 3",
        "Class 4",
        "Class 5",
    ]

    cm = confusion_matrix(
        true_labels,
        predicted_labels
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

    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("ECG5000 Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():

    torch.manual_seed(42)
    np.random.seed(42)


    ### Load Data

    train_path = "dataProcessed/ECG5000_CV_4000.txt"

    test_path = "dataProcessed/ECG5000_FINAL_TEST_1000.txt"

    train_data = np.loadtxt(
        train_path
    )

    test_data = np.loadtxt(
        test_path
    )


    ### Separate labels and ECG signals

    # First column = class label
    train_labels = train_data[:, 0].astype(int)

    # Remaining 140 columns = ECG waveform
    train_signals = train_data[:, 1:]

    test_labels = test_data[:, 0].astype(int)

    test_signals = test_data[:, 1:]


    ### Convert labels from 1-5 to 0-4

    # CrossEntropyLoss expects class indices
    # from 0 to number_of_classes - 1

    train_labels = train_labels - 1
    test_labels = test_labels - 1


    print("Training signal shape:")
    print(train_signals.shape)

    print("Test signal shape:")
    print(test_signals.shape)

    print("Training labels:")
    print(np.unique(train_labels, return_counts=True))


    ### Train / Validation Split

    train_signals, val_signals, train_labels, val_labels = train_test_split(
        train_signals,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )

    # Gives approximately:
    #
    # 3200 training samples
    # 800 validation samples
    # 1000 untouched test samples


    ### Standard Scaling

    scaler = StandardScaler()

    # ECG is univariate.
    #
    # Reshape all training ECG points into one column:
    #
    # (3200, 140)
    #       ↓
    # (3200*140, 1)

    train_scaled = scaler.fit_transform(
        train_signals.reshape(-1, 1)
    ).reshape(train_signals.shape)

    # IMPORTANT:
    # Validation and test use the SAME scaler
    # fitted using training data.

    val_scaled = scaler.transform(
        val_signals.reshape(-1, 1)
    ).reshape(val_signals.shape)

    test_scaled = scaler.transform(
        test_signals.reshape(-1, 1)
    ).reshape(test_signals.shape)


    ### Add feature dimension

    # Transformer expects:
    #
    # (samples, sequence_length, features)
    #
    # Currently:
    # (3200, 140)
    #
    # We need:
    # (3200, 140, 1)

    train_scaled = train_scaled[:, :, np.newaxis]

    val_scaled = val_scaled[:, :, np.newaxis]

    test_scaled = test_scaled[:, :, np.newaxis]


    print("\nTransformer input shapes:")

    print(
        "Train:",
        train_scaled.shape
    )

    print(
        "Validation:",
        val_scaled.shape
    )

    print(
        "Test:",
        test_scaled.shape
    )


    ### Dataset

    train_dataset = ECGDataset(
        signals=train_scaled,
        labels=train_labels,
    )

    val_dataset = ECGDataset(
        signals=val_scaled,
        labels=val_labels,
    )

    test_dataset = ECGDataset(
        signals=test_scaled,
        labels=test_labels,
    )


    ### DataLoaders

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


    ### Transformer Hyperparameters

    input_size = 1

    # Each ECG timestep becomes a 64 dimensional embedding
    d_model = 64

    # Feed-forward network inside Transformer
    dim_ff = 128

    # Number of attention heads
    n_heads = 4

    # Number of stacked Transformer encoder layers
    num_layers = 1

    dropout = 0.1

    # ECG5000 has 140 timesteps
    n_tokens = train_scaled.shape[1]

    # ECG5000 has 5 classes
    n_classes = 5

    learning_rate = 0.0001


    ### Device

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUsing device:")
    print(device)


    ### Initialise Model

    model = TransformerClassifier(
        input_size=input_size,
        d_model=d_model,
        dim_ff=dim_ff,
        n_classes=n_classes,
        num_layers=num_layers,
        n_heads=n_heads,
        dropout=dropout,
        n_tokens=n_tokens,
    ).to(device)


    ### Number of parameters

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nModel has {num_params} trainable parameters."
    )


    ### Optimiser

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    ### Loss Function

    criterion = nn.CrossEntropyLoss()


    ### Training

    epochs = 30


    train_losses = []
    val_losses = []
    epoch_times = []


    best_val_loss = float("inf")
    best_val_epoch = 0



    for epoch in range(epochs):

        ### Training Phase
        epoch_start_time = time.time()

        model.train()

        train_loss = 0.0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(
                x_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()


        train_loss /= len(train_loader)

        train_losses.append(
            train_loss
        )


        ### Validation Phase

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)

                y_batch = y_batch.to(device)

                predictions = model(
                    x_batch
                )

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss += loss.item()


        val_loss /= len(val_loader)

        val_losses.append(
            val_loss
        )


        ### Save Best Model

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_epoch = epoch

            torch.save(
                model.state_dict(),
                "best_transformer_model.pth"
            )
        ## epoch time
        epoch_time = time.time()-epoch_start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Best Val loss: {best_val_loss:.6f} | "
            f"Best epoch: {best_val_epoch + 1} | "
            f"Time: {epoch_time: .2f} s"
        )


    average_epoch_time = np.mean(epoch_times)

    print(
        f"\nAverage time per epoch: {average_epoch_time:.2f} seconds"
    )

    ### Plot Training and Validation Loss

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Cross Entropy Loss")

    plt.title(
        "Transformer Training and Validation Loss"
    )

    plt.legend()

    plt.grid()

    plt.show()


    ### Load Best Model

    best_model = TransformerClassifier(
        input_size=input_size,
        d_model=d_model,
        dim_ff=dim_ff,
        n_classes=n_classes,
        num_layers=num_layers,
        n_heads=n_heads,
        dropout=dropout,
        n_tokens=n_tokens,
    ).to(device)


    best_model.load_state_dict(
        torch.load(
            "best_transformer_model.pth",
            map_location=device
        )
    )

    best_model.eval()


    ### Test Model

    all_predictions = []
    all_targets = []


    with torch.no_grad():

        for x_batch, y_batch in test_loader:

            x_batch = x_batch.to(device)

            logits = best_model(
                x_batch
            )

            predicted_classes = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predicted_classes.cpu().numpy()
            )

            all_targets.extend(
                y_batch.numpy()
            )


    all_predictions = np.array(
        all_predictions
    )

    all_targets = np.array(
        all_targets
    )


    ### Evaluation Metrics

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro"
    )

    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions
    )


    print("\nTest Results")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            all_targets,
            all_predictions,
            target_names=[
                "Class 1",
                "Class 2",
                "Class 3",
                "Class 4",
                "Class 5",
            ],
            digits=4,
        )
    )


    ### Confusion Matrix

    plot_confusion_matrix(
        all_targets,
        all_predictions
    )


if __name__ == "__main__":
    main()