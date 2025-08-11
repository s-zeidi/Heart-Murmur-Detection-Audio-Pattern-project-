import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet34
from model_dataloader import get_loaders
from tqdm import tqdm

#v3 now is available
def train_resnet_model(model_type, train_csv, val_csv, model_path, batch_size=32, epochs=5):
    # Data loaders
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    # Device setup
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 Using device: {device}")

    # Dynamically get number of input channels from a sample
    sample_input, _ = next(iter(train_loader))
    in_channels = sample_input.shape[1]  # 1 or 3
    print(f"🧾 Detected input channels: {in_channels}")

    # Load specified ResNet model
    if model_type == "resnet18":
        model = resnet18(weights=None)
    elif model_type == "resnet34":
        model = resnet34(weights=None)
    else:
        raise ValueError("Invalid model_type. Use 'resnet18' or 'resnet34'.")

    # Adjust input layer and final classifier
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

        # Save model at current epoch
        epoch_path = model_path.replace(".pth", f"_epoch{epoch+1}.pth")
        os.makedirs(os.path.dirname(epoch_path), exist_ok=True)
        torch.save(model.state_dict(), epoch_path)
        print(f"💾 Saved model to: {epoch_path}")

    return model
