import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module): # Class: basically a blueprint, so this is defining my own type of NN

    def __init__(self, hidden_size=32): # def: definition of a special
        super().__init__() # Special function that runs when the model is first created
        #ie. construct model --> init --> create the layers


        self.hidden_size = hidden_size

        # Hyperparameters THAT WE CHOOSE
        # Each time step has 1 value: ECG amplitude
        # Hidden state has 32 values
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size, # internal representation contains 32 numbers (internal values/learnt features)
            batch_first=True, 
        ) 

        # Convert final LSTM output into 5 class scores
        self.fc = nn.Linear(
            self.hidden_size, # 32 hidden layers but need to convert into 5 layers
            5,
        )  # so it is ECG (140 X 1) --> LSTM --> 32 learned features --> linear layer --> 5 class scores

    def forward(self, x): #what happens when data goes through the network?

        # x shape:
        # (batch_size, 140, 1) (32 ecgs, 140 time steps each, each timestep 1 number)

        lstm_output, _ = self.lstm(x) #sends all ECGs through LSTM signal 

        # Take the output at the LAST time step (for each ecg in the batch)
        last_output = lstm_output[:, -1, :] #-1 because it can count backwards ie. -5, -4, -3, -2, -1

        # Convert it into 5 class scores (from 32 features --> 5 class scores)
        output = self.fc(last_output)

        return output


# ============================================================
# MAIN
# ============================================================

def main():

    # Hyperparameters that we can change
    HIDDEN_SIZE = 128

    # These go inside the training loop and optimizer directly because it changes how the model learns not its structure. 
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 30

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
    # (500, 140, 1) #entire training set has 500 ECG heartbeats


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
        batch_size=BATCH_SIZE,
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

    model = LSTMClassifier(hidden_size=HIDDEN_SIZE).to(device)
    # Creates an instance of your defined neural network model



    # --------------------------------------------------------
    # 8. Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # 9. Optimizer
    # --------------------------------------------------------

    # Optimiser is responsible for changing the NN weights 
    optimizer = torch.optim.Adam(
        model.parameters(), #gives Adam all the learnable numbers (ie. weights in the network)
        lr=LEARNING_RATE, #changes how aggressively optimiser changes the model
    )


    # --------------------------------------------------------
    # 10. Train
    # --------------------------------------------------------
    X_test_device = X_test.to(device)
    y_test_device = y_test.to(device)


    train_losses = []
    test_losses = []
    
    for epoch in range(EPOCHS):

        model.train() #set mode to trainning mode

        total_loss = 0

        for X_batch, y_batch in train_loader: # loops through all the batches

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Run ECGs through LSTM (forward pass)
            outputs = model(X_batch)

            # Compare predictions with correct labels// calculate the loss
            loss = criterion(
                outputs,
                y_batch,
            )

            # Calculate gradients so we can see how the loss will change
            loss.backward() 

            # Update model weights
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
 
        with torch.no_grad():
            test_outputs = model(X_test_device)
            test_loss = criterion(test_outputs, y_test_device).item()
 
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
 
        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f}"
        )

    best_epoch_index = test_losses.index(min(test_losses))  # 0-based
    best_epoch_number = best_epoch_index + 1                # human-readable
 
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss")
    plt.axvline(
        best_epoch_number,
        color="gray",
        linestyle="--",
        label=f"Best epoch ({best_epoch_number})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Test Loss by Epoch")
    plt.legend()
    plt.show()
 
    print(f"Lowest test loss was at epoch {best_epoch_number} "
            f"(test loss = {test_losses[best_epoch_index]:.4f})")
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


    y_test_np = y_test.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    cm = confusion_matrix(y_test_np, predictions_np)
 
    balanced_acc = balanced_accuracy_score(y_test_np, predictions_np)
 
    macro_f1 = f1_score(y_test_np, predictions_np, average="macro")
 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — ECG5000 Test Set")
    plt.show()
 
    print()
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")
    print(f"Macro F1 Score:    {macro_f1:.4f}")


if __name__ == "__main__":
    main()



    # model
    #   initlaise the model with the hyperparameters
    # data prep: 
    #   construct a data loader to load in the values and convert to pytorch tensors
    # training 
    #   loops through the epoches to optimise
        #   forward run
        #   Wcalculate gradients (backwards)
        #    run optimiser to update the wieghts
    # output is linear (hidden layer) to 5 ouputs
    # and then compare to the actual solution and print out all the sucess metrics import from skibidi library