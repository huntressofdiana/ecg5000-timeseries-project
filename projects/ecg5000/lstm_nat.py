import matplotlib.pyplot as plt
import numpy as np
import torch

#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split


from torch import nn
from torch.utils.data import DataLoader, Dataset

### Dataset
class ECGDataset(Dataset):
    def __init__(
        self,
        signal: np.ndarray,
        labels: np.ndarray,  # why is there a label instead of a context length
    ):
        self.signal = torch.tensor(
            signal,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        # Defines how many ECG samples exist
        return len(self.signal)

    def __getitem__(self, idx):
        # Complete ECG heartbeat
        x = self.signal[idx]

        # ECG class
        y = self.labels[idx]

        return x, y


### LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x):
        # x initially:
        # (batch_size, 140)

        # Add feature dimension
        x = x.unsqueeze(-1)

        # x is now:
        # (batch_size, 140, 1)

        lstm_output, _ = self.lstm(x)

        # LSTM output at the final ECG time step
        last_output = lstm_output[:, -1, :]

        # Apply dropout
        last_output = self.dropout(last_output)

        # Predict ECG class
        prediction = self.fc(last_output)

        return prediction


def main():

    ### Load the data

    # # train_path = r"ECG5000_TRAIN.txt"
    # train_path = "data/ECG5000_TRAIN.txt"
    # # test_path = r"ECG5000_TEST.txt"
    # test_path = "data/ECG5000_TEST.txt"

    cv_path = "dataProcessed/ECG5000_CV_4000.txt"
    test_path = "dataProcessed/ECG5000_FINAL_TEST_1000.txt"

    cv_data = np.loadtxt(cv_path)
    test_data = np.loadtxt(test_path)

    print("CV data shape:", cv_data.shape)
    print("Test shape:", test_data.shape)

    ### Split signal and labels

    # First column = class label
    # train_labels = train_data[:, 0].astype(int)
    cv_labels = cv_data[:, 0].astype(int)
    test_labels = test_data[:, 0].astype(int)

    # Remaining columns = ECG waveform
    # train_signal = train_data[:, 1:]
    cv_signal = cv_data[:, 1:]
    test_signal = test_data[:, 1:]


    ### Change labels from 1-5 to 0-4

    # train_labels = train_labels - 1
    cv_labels = cv_labels - 1
    test_labels = test_labels - 1

    # stratified 3200/800 split
    ### Split 4000 samples into training and validation

    train_signal, val_signal, train_labels, val_labels = train_test_split(
        cv_signal,
        cv_labels,
        test_size=800, #0.2 ??? change to 0.3
        random_state=42,
        stratify=cv_labels,
    )


    # print("Training signal shape:", train_signal.shape)
    # print("Test signal shape:", test_signal.shape)

    # print("Training labels shape:", train_labels.shape)
    # print("Classes:", np.unique(train_labels))

    print("Training signal shape:", train_signal.shape)
    print("Validation signal shape:", val_signal.shape)
    print("Final test signal shape:", test_signal.shape)

    print("Training labels shape:", train_labels.shape)
    print("Validation labels shape:", val_labels.shape)
    print("Final test labels shape:", test_labels.shape)

    print("\nTraining class counts:")
    print(np.bincount(train_labels)) # or np.unique

    print("\nValidation class counts:")
    print(np.bincount(val_labels))

    print("\nFinal test class counts:")
    print(np.bincount(test_labels))




    ### Scale the data

    scaler = StandardScaler()

    scaler.fit(train_signal)  # Fit only on training ECGs

    train_signal_scaled = scaler.transform(
        train_signal
    ).astype(np.float32)

    val_signal_scaled = scaler.transform(
        val_signal
    ).astype(np.float32)

    test_signal_scaled = scaler.transform(
        test_signal
    ).astype(np.float32)


    ### Create the dataset

    train_dataset = ECGDataset(
        signal=train_signal_scaled,
        labels=train_labels,
    )

    val_dataset = ECGDataset(
        signal=val_signal_scaled,
        labels=val_labels,
    )

    test_dataset = ECGDataset(
        signal=test_signal_scaled,
        labels=test_labels,
    )


    bs = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
    )


    x_batch, y_batch = next(iter(train_loader))

    print("Input shape:", x_batch.shape)
    print("Target shape:", y_batch.shape)


    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)


    ### Initialize model and optimizer

    # One ECG amplitude at each time step
    input_size = 1

    # Five ECG classes
    output_size = 5

    hidden_size = 64
    num_layers = 2
    dropout = 0.1


    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    ).to(device)


    print(model)


    # Loss function
    # criterion = nn.CrossEntropyLoss()

    ### Calculate class weights

    class_counts = np.bincount(train_labels)

    print("Class counts:", class_counts)

    class_weights = len(train_labels) / (
        len(class_counts) * class_counts
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    ).to(device)

    print("Class weights:", class_weights)


    ### Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # Optimizer
    lr = 0.001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )


    ### Training loop

    epochs = 100  #change


    #no of epochs allows w/o improvement
    patience = 10

    best_val_f1 = 0.0
    best_epoch = 0

    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    val_f1_scores = []

    ### Training Loop

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        train_correct = 0
        train_total = 0


        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)


            optimizer.zero_grad()


            ### Make prediction

            predictions = model(x_batch)


            ### Calculate loss

            loss = criterion(
                predictions,
                y_batch,
            )


            ### Backpropagation

            loss.backward()

            optimizer.step()


            train_loss += loss.item()


            ### Calculate accuracy

            predicted_class = torch.argmax(
                predictions,
                dim=1,
            )

            train_correct += (
                predicted_class == y_batch
            ).sum().item()

            train_total += y_batch.size(0)


        train_loss /= len(train_loader)

        train_accuracy = train_correct  / train_total 


        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        ##################################################
        ### VALIDATION
        ##################################################

        model.eval()

        val_loss = 0.0

        val_correct = 0
        val_total = 0

        val_predictions = []
        val_targets = []


        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)


                predictions = model(x_batch)


                ### Validation loss

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_loss += loss.item()


                ### Predicted class

                predicted_class = torch.argmax(
                    predictions,
                    dim=1,
                )


                ### Validation accuracy

                val_correct += (
                    predicted_class == y_batch
                ).sum().item()

                val_total += y_batch.size(0)


                ### Store predictions for macro F1

                val_predictions.extend(
                    predicted_class.cpu().numpy()
                )

                val_targets.extend(
                    y_batch.cpu().numpy()
                )


        ### Average validation loss

        val_loss /= len(val_loader)

        val_accuracy = (
            val_correct / val_total
        )


        val_predictions = np.array(
            val_predictions
        )

        val_targets = np.array(
            val_targets
        )


        ### Validation macro F1

        val_f1 = f1_score(
            val_targets,
            val_predictions,
            average="macro",
            zero_division=0,
        )


        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)



        ##################################################
        ### EARLY STOPPING
        ##################################################

        # Check whether validation macro F1 improved

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch + 1

            epochs_without_improvement = 0


            ### Save the best model

            torch.save(
                model.state_dict(),
                "best_lstm_model.pth",
            )


        else:

            epochs_without_improvement += 1





        ## Print results
        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val acc: {val_accuracy:.4f} | "
            f"Val Macro F1: {val_f1:.4f} | "
            f"Best F1: {best_val_f1:.4f}"
        )


        ##################################################
        ### STOP TRAINING
        ##################################################

        if epochs_without_improvement >= patience:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch + 1}"
            )

            print(
                f"Best model was from epoch "
                f"{best_epoch}"
            )

            break





    ### Plot loss curve
    plt.figure()

    plt.plot(
        train_losses,
        label="Train Loss",
    )

    plt.plot(
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.grid(True)
    plt.legend()

    plt.show()



    ### Plot accuracy curve
    plt.figure()

    plt.plot(
        train_accuracies,
        label="Train Accuracy",
    )

    plt.plot(
        val_accuracies,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.grid(True)
    plt.legend()

    plt.show()


    #plot validation macro f1
    plt.figure()

    plt.plot(
        val_f1_scores,
        label="Validation Macro F1",
    )

    plt.axvline(
        x=best_epoch - 1,
        linestyle="--",
        label="Best Model",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")

    plt.title(
        "Validation Macro F1"
    )

    plt.grid(True)
    plt.legend()

    plt.show()

    #LOAD BEST MODEL
    best_model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    ).to(device)


    best_model.load_state_dict(
        torch.load(
            "best_lstm_model.pth",
            map_location=device,
        )
    )


    best_model.eval()

    print(
        f"\nLoaded best model from epoch "
        f"{best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_val_f1:.4f}"
    )





    ### Test the model

    #model.eval()

    test_predictions = []
    test_targets = []


    with torch.no_grad():

        for x_batch, y_batch in test_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)


            predictions = best_model(x_batch)
            # predictions = model(x_batch)


            predicted_class = torch.argmax(
                predictions,
                dim=1,
            )


            test_predictions.extend(
                predicted_class.cpu().numpy()
            )

            test_targets.extend(
                y_batch.cpu().numpy()
            )


    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)


    ### Metrics

    accuracy = accuracy_score(
        test_targets,
        test_predictions,
    )

    macro_f1 = f1_score(
    test_targets,
    test_predictions,
    average="macro",
    zero_division=0
    )

    macro_precision = precision_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print(
        f"\nFINAL TEST RESULTS"
    )
    print(f"\nTest accuracy: {accuracy:.4f}")

    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1 score: {macro_f1:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            test_targets,
            test_predictions,
            zero_division=0
        )
    )


    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            test_targets,
            test_predictions,
        )
    )


    ##################################################
    ### CONFUSION MATRIX
    ##################################################

    cm = confusion_matrix(
        test_targets,
        test_predictions,
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Class 1",
            "Class 2",
            "Class 3",
            "Class 4",
            "Class 5",
        ],
    )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )


    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        "ECG5000 LSTM Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()