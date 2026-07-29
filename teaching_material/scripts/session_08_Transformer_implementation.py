import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score


class BaseTransformerClassifier(nn.Module):
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

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=n_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model * n_tokens, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_embed = self.input_projection(x)
        x_embed_pe = self.positional_encoding(x_embed)

        z = self.encoder(x_embed_pe)

        # Flatten the output of the encoder before passing it to the output projection layer
        z_flat = z.flatten(start_dim=1)

        logits = self.output_projection(z_flat)

        return logits


class CLSTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        n_classes,
        d_model,
        n_heads,
        num_layers,
        dim_ff,
        dropout,
        n_tokens,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=n_tokens + 1
        )
        self.cls_token = nn.Parameter(torch.randn(1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, F_in)
        mask: (B, T) bool (True = real, False = pad)
        """
        x = self.input_proj(x)
        bs = x.shape[0]

        cls_tokens = self.cls_token.expand(bs, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.positional_encoding(x)

        x = self.encoder(x)
        summary = x[:, 0, :]

        logits = self.output_projection(summary)

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class RacketSportsDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():

    torch.manual_seed(42)

    ### Load the data
    train = np.load(
        r"C:\Users\micha\Desktop\Summer_School\datasets\lectures\RacketSports_TRAIN.npz"
    )

    train_data = train["data"]  # (Samples, Time steps, Features)
    train_labels = train[
        "labels"
    ]  # [Badminton Clear, Badminton Smash, Squash Forehand, Squash Backhand]

    unique_labels = np.unique(train_labels)
    print(train_data.shape)
    print(unique_labels.shape)

    test = np.load(
        r"C:\Users\micha\Desktop\Summer_School\datasets\lectures\RacketSports_TRAIN.npz"
    )
    test_data = test["data"]  # (Samples, Time steps, Features)
    test_labels = test["labels"]  # [Standing, Walking, Running, Badminton]

    #### One Hot Encoding of the labels
    ohe = OneHotEncoder()
    train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1)).toarray()
    test_labels_ohe = ohe.transform(test_labels.reshape(-1, 1)).toarray()

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
    train_dataset = RacketSportsDataset(data=train_scaled, labels=train_labels_ohe)
    val_dataset = RacketSportsDataset(data=val_scaled, labels=val_labels_ohe)
    test_dataset = RacketSportsDataset(data=test_scaled, labels=test_labels_ohe)

    bs = 32
    lr = 0.0001
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ### Initialize model and optimizer
    input_size = train_data.shape[-1]
    n_classes = train_labels_ohe.shape[1]
    d_model = 64
    dim_ff = 128
    n_heads = 4
    dropout = 0.1
    n_layers = 1
    n_tokens = train_data.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaseTransformerClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        dim_ff=dim_ff,
        n_heads=n_heads,
        dropout=dropout,
        num_layers=n_layers,
        n_tokens=n_tokens,
    ).to(device)

    ### Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    epochs = 500

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0

    for epoch in range(epochs):
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

        if val_loss < best_val_loss:
            best_val_loss = train_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f}"
            f" | "
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

    all_losses = np.concatenate([train_losses, val_losses])

    # np.save(r"C:\Users\micha\Desktop\Summer_School\train_loss_base.npy", all_losses)
    ### Evaluation

    best_model = BaseTransformerClassifier(
        input_size=input_size,
        n_classes=n_classes,
        d_model=d_model,
        dim_ff=dim_ff,
        n_heads=n_heads,
        dropout=dropout,
        num_layers=n_layers,
        n_tokens=n_tokens,
    ).to(device)
    best_model.load_state_dict(torch.load("best_model.pth"))

    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    predicted_indices = all_logits.argmax(dim=1)
    true_indices = all_targets.argmax(dim=1)

    accuracy = accuracy_score(
        true_indices.numpy(),
        predicted_indices.numpy(),
    )

    f1_micro = f1_score(
        true_indices.numpy(), predicted_indices.numpy(), average="micro"
    )

    print(f"Accuracy: {accuracy*100:.4f}%")

    print(f"F1 (Micro): {f1_micro:.4f}")


if __name__ == "__main__":
    main()
