from pyexpat import model

import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

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

def main():

    ### Load training data
    train = pd.read_csv(
        "data/ECG5000_TRAIN.txt",
        sep=r"\s+",
        header=None,
    )

    y_train = train.iloc[:, 0].values
    X_train = train.iloc[:, 1:].values

    print("X train shape:", X_train.shape) # tells us that there are 500 ecgs, but 140 time steps per ecg
    print("y train shape:", y_train.shape) # just 1 class per ecg

    ### Create the dataset
    train_dataset = ECGDataset(
        signals=X_train,
        labels=y_train,
    )

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

    x_batch, y_batch = next(iter(train_loader))
    print("Batch input shape:", x_batch.shape)
    print("Batch label shape:", y_batch.shape)

    ### Initialize model

    input_size = 1
    hidden_size = 64
    num_layers = 1
    num_classes = 5

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.0,
    )

    outputs = model(x_batch)

    print("Model output shape:", outputs.shape)

    ### LOSS FUNCTION --> CROSS ENTROPY LOSS 
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
    outputs,
    y_batch,    
    )

    print("Loss:", loss.item())

    ### Optimizer
    learning_rate = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    ### Training loop
    epochs = 200
    train_losses = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(x_batch)

            loss = criterion(
                outputs,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.6f}"
        )

        ### Evaluate on test set
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            outputs = model(x_batch)

            predicted_classes = torch.argmax(
                outputs,
                dim=1,
            )

            correct += (
                predicted_classes == y_batch
            ).sum().item()

            total += y_batch.size(0)
        test_accuracy = correct / total

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    ### Plot training loss
    plt.plot(train_losses)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("LSTM Training Loss")
    plt.show()

if __name__ == "__main__":
    main()