"""CRNN architecture retained from the thesis baseline."""

import torch
import torch.nn as nn
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    """VGG16 blocks 1-3, producing (B, 256, 4, 16)."""

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg16 = models.vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg16.features.children())[:17])
        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, inputs):
        return self.features(inputs)


class AdaptationLayers(nn.Module):
    """Trainable convolutional adaptation and height pooling."""

    def __init__(self, in_channels=256, hidden_channels=512, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.height_pool = nn.AdaptiveMaxPool2d((1, None))

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        features = self.height_pool(features).squeeze(2)
        return features.permute(0, 2, 1)


class BidirectionalLSTM(nn.Module):
    """Two-layer bidirectional LSTM used by the baseline."""

    def __init__(self, input_size=512, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        return self.dropout(outputs)


class CRNNHybrid(nn.Module):
    """VGG16 blocks 1-3 + adaptation + BiLSTM + CTC logits."""

    def __init__(
        self,
        num_classes,
        pretrained_vgg=True,
        freeze_vgg=True,
        hidden_size=256,
        lstm_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.feature_extractor = VGG16FeatureExtractor(pretrained_vgg, freeze_vgg)
        self.adaptation = AdaptationLayers(256, hidden_size * 2, dropout)
        self.bilstm = BidirectionalLSTM(hidden_size * 2, hidden_size, lstm_layers, dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        adapted = self.adaptation(features)
        sequence = self.bilstm(adapted)
        return self.fc(sequence)

    @staticmethod
    def get_sequence_length():
        return 16

    def count_parameters(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": 100.0 * trainable / total,
        }


def create_model(charset, pretrained=True, freeze_vgg=True):
    return CRNNHybrid(
        num_classes=len(charset) + 1,
        pretrained_vgg=pretrained,
        freeze_vgg=freeze_vgg,
        hidden_size=256,
        lstm_layers=2,
        dropout=0.3,
    )
