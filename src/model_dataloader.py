import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyHeartDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.label_map = {"Absent": 0, "Present": 1, "Unknown": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = np.load(row["segment_path"])  # Could be (128, T) or (3, 128, T)

        # Ensure shape is (C, 128, T)
        if x.ndim == 2:
            # Case: single-channel (128, T) → add channel dim → (1, 128, T)
            x = np.expand_dims(x, axis=0)
        elif x.ndim == 3 and x.shape[0] not in [1, 3]:
            raise ValueError(f"Unexpected shape {x.shape}: expected (1, 128, T) or (3, 128, T)")

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.label_map[row["murmur_label"]], dtype=torch.long)
        return x, y

def get_loaders(train_csv, val_csv, batch_size=32):
    train_data = MyHeartDataset(train_csv)
    val_data = MyHeartDataset(val_csv)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
