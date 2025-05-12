import torch
import torch.nn as nn
import torch.optim as optim
from model_dataloader import get_loaders
from cnn_attention_model import CNNAttentionClassifier
from focal_loss import FocalLoss
from tqdm import tqdm
import os

def train_attention_model(train_csv, val_csv, model_path, batch_size=32, epochs=10):
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = CNNAttentionClassifier(num_classes=3).to(device)

    # Focal Loss with class weights for [Absent, Present, Unknown]
    alpha = torch.tensor([1.0, 3.0, 5.0]).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} — Average Train Loss: {avg_loss:.4f}")

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
    print(f"✅ Attention model saved to: {model_path}")

    return model