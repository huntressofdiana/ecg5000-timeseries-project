import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Each time step has 1 value: ECG amplitude
        # Hidden state has 32 values
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            batch_first=True,
        ) 

        # Convert final LSTM output into 5 class scores
        self.fc = nn.Linear(
            32,
            5,
        )

    def forward(self, x):

        # x shape:
        # (batch_size, 140, 1)

        lstm_output, _ = self.lstm(x)

        # Take the output at the LAST time step
        last_output = lstm_output[:, -1, :]

        # Convert it into 5 class scores
        output = self.fc(last_output)

        return output


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    train_path = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\data\ECG5000_TRAIN.txt'

    test_path = r'C:\Users\Yovel\Documents\pytorch_project\DL_TimeSeries_SummerSchool\projects\ecg5000\data\ECG5000_TEST.txt'

    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)


    # --------------------------------------------------------
    # 2. Separate ECG signals and labels
    # --------------------------------------------------------

    # First column = label
    y_train = train_data[:, 0].astype(int)
    y_test = test_data[:, 0].astype(int)

    # Remaining 140 columns = ECG signal
    X_train = train_data[:, 1:]
    X_test = test_data[:, 1:]


    # --------------------------------------------------------
    # 3. Change labels from 1-5 to 0-4
    # --------------------------------------------------------

    y_train = y_train - 1
    y_test = y_test - 1


    # --------------------------------------------------------
    # 4. Convert NumPy arrays into PyTorch tensors
    # --------------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32, #convert input into 32 bit floating point numbers
    ).unsqueeze(-1)

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32,
    ).unsqueeze(-1)

    y_train = torch.tensor(
        y_train,
        dtype=torch.long, #converts labels into 64 bit integers
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long,
    )


    print("Training input:", X_train.shape)
    print("Training labels:", y_train.shape)

    # X_train should be:
    # (500, 140, 1)


    # --------------------------------------------------------
    # 5. Create DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train,  
        #binds the pairs together into a single dataset object
    )


    # Group data into mini batches of 32 samples 
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True, #randomises the order of the samples
    )

    


    # --------------------------------------------------------
    # 6. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)


    # --------------------------------------------------------
    # 7. Create model
    # --------------------------------------------------------

    model = LSTMClassifier().to(device)
    # Creates an instance of your defined neural network model



    # --------------------------------------------------------
    # 8. Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # 9. Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )


    # --------------------------------------------------------
    # 10. Train
    # --------------------------------------------------------

    epochs = 100

    for epoch in range(epochs):

        model.train() #set mode

        total_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Run ECGs through LSTM (forward pass)
            outputs = model(X_batch)

            # Compare predictions with correct labels
            loss = criterion(
                outputs,
                y_batch,
            )

            # Calculate gradients
            loss.backward()

            # Update model weights
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )


    # --------------------------------------------------------
    # 11. Test
    # --------------------------------------------------------

    model.eval()

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():

        outputs = model(X_test)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        correct = (
            predictions == y_test
        ).sum().item()

        accuracy = correct / len(y_test)


    print()
    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()