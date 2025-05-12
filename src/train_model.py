# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from model_dataloader import get_loaders
from tqdm import tqdm

def train_and_save_model(train_csv, val_csv, model_path, batch_size=32, epochs=5):
    # Data loaders
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    # Device setup
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load ResNet18
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
        for X, y in tqdm(train_loader, desc=f"Training", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} — Average Train Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")

    return model