import torch.nn as nn
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
            raise ValueError("Invalid model_type: choose 'resnet18' or 'resnet34'")

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
