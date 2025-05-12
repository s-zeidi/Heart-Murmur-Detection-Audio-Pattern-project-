import torch
import torch.nn as nn
from torchvision.models import resnet34


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        out = torch.bmm(attn_weights, V)
        return out.mean(dim=1)


class CNNAttentionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNAttentionClassifier, self).__init__()
        base_model = resnet34(weights=None)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*(list(base_model.children())[:-2]))

        self.attention = SelfAttention(input_dim=512)  # ResNet34 also ends in 512-dim features
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        features = self.cnn(x)  # [B, 512, H, W]
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, T, D]
        attended = self.attention(features)  # [B, D]
        out = self.dropout(attended)
        return self.fc(out)
