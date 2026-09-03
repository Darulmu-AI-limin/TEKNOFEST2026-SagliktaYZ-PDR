import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM1D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HybridECGModel(nn.Module):
    def __init__(self, num_classes=5):
        super(HybridECGModel, self).__init__()
        
        # ResNet1D for feature extraction per lead
        # Input shape: [Batch * 12, 1, 2500] (we process each lead as 1 channel)
        self.resnet = nn.Sequential(
            ResNetBlock1D(1, 32, stride=2),
            nn.MaxPool1d(2),
            ResNetBlock1D(32, 64, stride=2),
            nn.MaxPool1d(2),
            ResNetBlock1D(64, 128, stride=2),
            nn.MaxPool1d(2),
            ResNetBlock1D(128, 256, stride=2),
            nn.MaxPool1d(2)
        )
        
        # CBAM Attention (applies to [Batch * 12, 256, T])
        self.cbam = CBAM1D(256)
        
        # Transformer Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=4, 
            dim_feedforward=1024, 
            dropout=0.1, 
            norm_first=True, # Pre-LN Transformer
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: [Batch, 12, 2500]
        B, num_leads, seq_len = x.shape
        
        # Reshape to process each lead independently through ResNet
        x = x.view(B * num_leads, 1, seq_len)
        
        # ResNet + MaxPool
        x = self.resnet(x) # shape: [B * 12, 256, T]
        
        # CBAM
        x = self.cbam(x)
        
        # Global Average Pooling over time dimension T
        x = torch.mean(x, dim=2) # shape: [B * 12, 256]
        
        # Reshape back to [Batch, 12, 256]
        x = x.view(B, num_leads, 256)
        
        # Add CLS token -> [Batch, 13, 256]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer Encoder (spatial relation between leads)
        x = self.transformer_encoder(x)
        
        # Extract CLS token output -> [Batch, 256]
        cls_out = x[:, 0, :]
        
        # Classification Head (Sigmoid is applied later or inside loss)
        logits = self.classifier(cls_out)
        
        # Sigmoid can be returned for inference, or logits for BCEWithLogitsLoss / ASL
        return logits
