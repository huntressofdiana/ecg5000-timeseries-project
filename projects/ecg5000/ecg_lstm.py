from pyexpat import model

import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
)

train = pd.read_csv(
    "data/ECG5000_TRAIN.txt",
    sep=r"\s+",
    header=None
)

## lstm --> needs it in (seq lenth, no. of features)

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
        return len(self.signals) # --> no. of ecg signals

    def __getitem__(self, idx):
        x = self.signals[idx] # --> 1 complete ecg waveform (so 140 time steps)
        x = x.unsqueeze(-1) # feature dimension --> constrict to (140,1)

        # ECG class
        y = self.labels[idx] - 1

        return x, y

### LSTM Model
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
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        last_output = lstm_output[:, -1, :] # LSTM output at the final context time step

        prediction = self.fc(last_output)

        return prediction

def plot_confusion_matrix(
    true_indices: np.ndarray, predicted_indices: np.ndarray
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

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix – ECG5000")
    plt.tight_layout()
    plt.show()

def main():

    ### Load training data
    train = pd.read_csv(
        "data/ECG5000_TRAIN.txt",
        sep=r"\s+",
        header=None,
    )
    X = train.iloc[:, 1:].values
    y = train.iloc[:, 0].values

    ### Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Standard scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    # Learn mean and standard deviation from TRAINING data only
    X_train = scaler.fit_transform(X_train)

    # Apply the same scaling to validation data
    X_val = scaler.transform(X_val)

    # Keep everything as float32 for PyTorch
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)

    print("X train shape:", X_train.shape) # tells us that there are 500 ecgs, but 140 time steps per ecg
    print("y train shape:", y_train.shape) # just 1 class per ecg

    ### Create the dataset
    train_dataset = ECGDataset(
        signals=X_train,
        labels=y_train,
    )

    val_dataset = ECGDataset(
    signals=X_val,
    labels=y_val,
)

    ### Calculate class weights
    class_counts = np.bincount(y_train.astype(int))[1:]

    class_weights = np.sqrt(len(y_train) / (len(class_counts) * class_counts))
    class_weights = np.clip(class_weights, a_min=None, a_max=10.0)

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    ### Load test data
    test = pd.read_csv(
        "data/ECG5000_TEST.txt",
        sep=r"\s+",
        header=None,
    )

    y_test = test.iloc[:, 0].values
    X_test = test.iloc[:, 1:].values

    print("X test shape:", X_test.shape)
    print("y test shape:", y_test.shape)

    # Use SAME scaler learned from training data
    X_test = scaler.transform(X_test).astype(np.float32)

    ### Create test dataset
    test_dataset = ECGDataset(
        signals=X_test,
        labels=y_test,
    )
    
    test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    )

    # print("Number of ECGs:", len(train_dataset))
    x, y = train_dataset[0]

    print("ECG shape:", x.shape)
    print("Label:", y)

    ### Create DataLoader
    batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)   

    x_batch, y_batch = next(iter(train_loader))
    print("Batch input shape:", x_batch.shape)
    print("Batch label shape:", y_batch.shape)

    ### Initialize model

    input_size = 1
    hidden_size = 128 
    num_layers = 2
    num_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)

    outputs = model(x_batch)

    print("Model output shape:", outputs.shape)

    class_weights = class_weights.to(device)

    ### LOSS FUNCTION --> CROSS ENTROPY LOSS 
    criterion = nn.CrossEntropyLoss(
    weight=class_weights
    )
    loss = criterion(
    outputs,
    y_batch,    
    )

    print("Loss:", loss.item())

    ### Optimizer
    learning_rate = 0.0005

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    ### Training loop
    epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch)

            loss = criterion(
                outputs,
                y_batch,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        ### Validation
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                outputs = model(x_batch)

                loss = criterion(
                    outputs,
                    y_batch,
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
    )

        ### Evaluate on test set
    model.eval()

    all_true = []
    all_preds = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            outputs = model(x_batch)

            predicted_classes = torch.argmax(outputs, dim=1)

            all_true.extend(y_batch.numpy())
            all_preds.extend(predicted_classes.numpy())

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

   # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_accuracy = (all_true == all_preds).mean()

    balanced_acc = balanced_accuracy_score(
        all_true,
        all_preds,
    )

    macro_f1 = f1_score(
        all_true,
        all_preds,
        average="macro",
        zero_division=0,
    )

    print("\n============================")
    print("TEST RESULTS")
    print("============================")

    print(
        f"Accuracy:          "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_acc * 100:.2f}%"
    )

    print(
        f"Macro F1 Score:    "
        f"{macro_f1:.4f}"
    )

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.show()

    plot_confusion_matrix(all_true, all_preds)

if __name__ == "__main__":
    main()