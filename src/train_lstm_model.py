import torch
import torch.nn as nn
import torch.optim as optim
from model_dataloader import get_loaders
import os
from tqdm import tqdm


class LSTMSoundClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=3):
        super(LSTMSoundClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, 128, T] -> [B, T, 128]
        output, _ = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)


def train_lstm_model(train_csv, val_csv, model_path, batch_size=32, epochs=10):
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = LSTMSoundClassifier().to(device)

    # Class weights: [Absent, Present, Unknown] => higher weight on Unknown
    weights = torch.tensor([1.0, 3.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} — Average Train Loss: {avg_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"📉 Epoch {epoch+1} — Average Val Loss: {avg_val_loss:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")

    return model