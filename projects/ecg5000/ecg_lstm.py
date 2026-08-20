import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

train = pd.read_csv(
    "data/ECG5000_TRAIN.txt",
    sep=r"\s+",
    header=None
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
        return len(self.signals) # --> no. of ecg signals

    def __getitem__(self, idx):
        x = self.signals[idx] # --> 1 complete ecg waveform (so 140 time steps)
        x = x.unsqueeze(-1) # feature dimension --> constrict to (140,1)

        # ECG class
        y = self.labels[idx]

        return x, y
