import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Dataset
# ============================================================

class ECGDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
    ):
        # signals originally:
        # (number_of_ECGs, 140)
        #
        # LSTM needs:
        # (number_of_ECGs, 140, 1)

        self.signals = torch.tensor(
            signals,
            dtype=torch.float32,
        ).unsqueeze(-1)

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        # Number of separate ECG recordings
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]

        return x, y


# ============================================================
# LSTM Classification Model
# ============================================================

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,

            # PyTorch LSTM dropout only works when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Instead of predicting ONE numerical value,
        # predict 5 class scores
        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        # x shape:
        # (batch_size, 140, 1)

        lstm_output, _ = self.lstm(x)

        # Take output from final time step
        #
        # Shape:
        # (batch_size, hidden_size)
        last_output = lstm_output[:, -1, :]

        last_output = self.dropout(last_output)

        # Produce one score for each ECG class
        #
        # Shape:
        # (batch_size, 5)
        logits = self.fc(last_output)

        return logits


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load ECG5000
    # --------------------------------------------------------

    train_path = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\data\ECG5000_TRAIN.txt'

    test_path = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\data\ECG5000_TEST.txt'

    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)

    print("Raw training data:", train_data.shape)
    print("Raw test data:", test_data.shape)

    # --------------------------------------------------------
    # Separate labels and ECG signals
    # --------------------------------------------------------

    # First column = ECG class
    y_train = train_data[:, 0].astype(int)
    y_test = test_data[:, 0].astype(int)

    # Remaining 140 columns = ECG waveform
    X_train = train_data[:, 1:].astype(np.float32)
    X_test = test_data[:, 1:].astype(np.float32)

    print("\nBefore processing:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nOriginal classes:")
    print(np.unique(y_train))

    # --------------------------------------------------------
    # Convert labels
    # --------------------------------------------------------
    #
    # ECG5000 labels:
    #
    # 1, 2, 3, 4, 5
    #
    # CrossEntropyLoss requires:
    #
    # 0, 1, 2, 3, 4
    #

    y_train = y_train - 1
    y_test = y_test - 1

    print("\nLabels used by PyTorch:")
    print(np.unique(y_train))

    # --------------------------------------------------------
    # Scale ECG signals
    # --------------------------------------------------------

    scaler = StandardScaler()

    # IMPORTANT:
    # fit only using training data
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(
        X_train
    ).astype(np.float32)

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    # --------------------------------------------------------
    # Create PyTorch datasets
    # --------------------------------------------------------

    train_dataset = ECGDataset(
        signals=X_train_scaled,
        labels=y_train,
    )

    test_dataset = ECGDataset(
        signals=X_test_scaled,
        labels=y_test,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Look at one batch
    x_batch, y_batch = next(iter(train_loader))

    print("\nBatch shapes:")
    print("Input shape:", x_batch.shape)
    print("Target shape:", y_batch.shape)

    # Should be approximately:
    #
    # Input shape:  torch.Size([32, 140, 1])
    # Target shape: torch.Size([32])

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    # At each time step we have ONE feature:
    # ECG amplitude
    input_size = 1

    hidden_size = 64

    num_layers = 1

    # ECG5000 has five classes
    num_classes = 5

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------
    #
    # We are doing classification now.
    #
    # NOT:
    # nn.MSELoss()
    #
    # Instead:
    #

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    learning_rate = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # ========================================================
    # Training
    # ========================================================

    epochs = 50

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(x_batch)

            # Calculate classification loss
            loss = criterion(
                logits,
                y_batch,
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

            # ----------------------------------------------
            # Training accuracy
            # ----------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

        average_loss = total_loss / len(train_loader)

        accuracy = correct / total

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {average_loss:.6f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    # ========================================================
    # Plot training loss
    # ========================================================

    plt.figure(figsize=(8, 4))

    plt.plot(train_losses)

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss")

    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # ========================================================
    # Plot training accuracy
    # ========================================================

    plt.figure(figsize=(8, 4))

    plt.plot(train_accuracies)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # ========================================================
    # Test / Evaluation
    # ========================================================

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for x_batch, y_batch in test_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            logits = model(x_batch)

            # Choose class with largest score
            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu().numpy()
            )

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
    )

    print("\n==========================")
    print("TEST RESULTS")
    print("==========================")

    print(
        f"Accuracy:          "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1 score:    "
        f"{macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

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
            zero_division=0,
        )
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    # Convert predictions back from:
    # 0-4
    #
    # to ECG labels:
    # 1-5

    true_labels = all_targets + 1
    predicted_labels = all_predictions + 1

    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predicted_labels,
        labels=[1, 2, 3, 4, 5],
    )

    plt.title("ECG5000 Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ========================================================
    # Show some predictions
    # ========================================================

    print("\nFirst 20 test predictions:")

    for i in range(20):

        print(
            f"ECG {i:2d} | "
            f"True: {true_labels[i]} | "
            f"Predicted: {predicted_labels[i]}"
        )


if __name__ == "__main__":
    main()