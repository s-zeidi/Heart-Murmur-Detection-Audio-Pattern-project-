import torch
import torch.nn as nn
import torch.optim as optim
from model_dataloader import get_loaders
from sklearn.utils.class_weight import compute_class_weight
import os
from tqdm import tqdm
import pandas as pd
import numpy as np


class LSTMSoundClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=3):
        super(LSTMSoundClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Input: [B, 1, 128, T] → [B, 128, T] → [B, T, 128]
        x = x.squeeze(1).permute(0, 2, 1)
        output, _ = self.lstm(x)                  # [B, T, H*2]
        last_output = output[:, -1, :]            # [B, H*2]
        last_output = self.norm(last_output)      # Normalize
        last_output = self.dropout(last_output)
        return self.fc(last_output)               # [B, 3]


def compute_class_weights(csv_path, label_column="murmur_label"):
    df = pd.read_csv(csv_path)
    labels = df[label_column].values
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def train_lstm_model(train_csv, val_csv, model_dir, batch_size=32, epochs=15):
    # === Load Data ===
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    # === Select Device ===
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    # === Initialize Model ===
    model = LSTMSoundClassifier().to(device)

    # === Compute Class Weights ===
    weights = compute_class_weights(train_csv).to(device)
    print(f"📊 Computed class weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === Ensure Save Directory Exists ===
    os.makedirs(model_dir, exist_ok=True)

    # === Training Loop ===
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        print(f"\n🔁 Epoch {epoch}/{epochs}")
        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"✅ Avg Train Loss: {avg_train_loss:.4f}")

        # === Validation ===
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"📉 Avg Val Loss: {avg_val_loss:.4f}")

        # === Save Model for Each Epoch ===
        epoch_model_path = os.path.join(model_dir, f"lstm_epoch{epoch}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"💾 Saved: {epoch_model_path}")

    print("🏁 Training complete.")
    return model
