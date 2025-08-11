import torch
import torch.nn as nn
from torchvision.models import resnet34

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** 0.5
        self.proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        attn_output = torch.bmm(attn_weights, V)
        pooled = attn_output.mean(dim=1)
        residual = x.mean(dim=1)
        return self.proj(pooled) + residual  # residual connection

class CNNAttentionClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNAttentionClassifier, self).__init__()
        base_model = resnet34(weights="IMAGENET1K_V1")  # ✅ pretrained
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*(list(base_model.children())[:-2]))

        self.attention = SelfAttention(input_dim=512, num_heads=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # [B, 512, H, W]
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, T, D]
        attended = self.attention(features)  # [B, D]
        out = self.dropout(attended)
        return self.fc(out)
