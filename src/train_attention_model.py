import torch
import torch.nn as nn
import torch.optim as optim
from model_dataloader import get_loaders
from cnn_attention_model import CNNAttentionClassifier
from focal_loss import FocalLoss
from tqdm import tqdm
import os

def train_attention_model(train_csv, val_csv, base_model_path, batch_size=32, epochs=10):
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CNNAttentionClassifier(num_classes=3).to(device)

    # Focal Loss with class weights
    alpha = torch.tensor([1.0, 3.0, 5.0]).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(os.path.dirname(base_model_path), exist_ok=True)

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
        print(f"✅ Epoch {epoch + 1} — Average Train Loss: {avg_loss:.4f}")
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"📉 Epoch {epoch + 1} — Average Val Loss: {avg_val_loss:.4f}")

        # 🔒 Save model for this epoch
        epoch_model_path = base_model_path.replace(".pth", f"_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"📦 Saved model to: {epoch_model_path}")

    return model
