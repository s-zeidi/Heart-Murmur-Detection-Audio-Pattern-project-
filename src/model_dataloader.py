# model_dataloader.py

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
        x = np.load(row["segment_path"])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, 128, T]
        y = torch.tensor(self.label_map[row["murmur_label"]], dtype=torch.long)
        return x, y

def get_loaders(train_csv, val_csv, batch_size=32):
    train_data = MyHeartDataset(train_csv)
    val_data = MyHeartDataset(val_csv)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
