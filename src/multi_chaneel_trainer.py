import os
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, accuracy_score
from model_dataloader import get_loaders
import torchvision.models as models
#s
class HeartSoundCNN(nn.Module):
    def __init__(self, model_type="resnet18", num_classes=3):
        super(HeartSoundCNN, self).__init__()

        if model_type == "resnet18":
            self.model = models.resnet18(weights=None)
        elif model_type == "resnet34":
            self.model = models.resnet34(weights=None)
        else:
            raise ValueError("Invalid model_type. Choose 'resnet18' or 'resnet34'.")

        # Adjust input channels to 3 (multi-resolution)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adjust final classifier to output num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def train_heart_model(
    train_csv,
    val_csv,
    model_path,
    model_type="resnet18",  # supports 'resnet18' or 'resnet34'
    batch_size=32,
    lr=1e-4,
    epochs=15,
    num_classes=3
):
    # ✅ Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"📡 Using device: {device}")

    # Load data
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    # Initialize model
    model = HeartSoundCNN(model_type=model_type, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"📈 Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

        # Save model at current epoch
        epoch_model_path = model_path.replace(".pth", f"_epoch{epoch}.pth")
        os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
        torch.save(model.state_dict(), epoch_model_path)
        print(f"💾 Model saved → {epoch_model_path}")

    return model
