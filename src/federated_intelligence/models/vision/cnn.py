"""CNN for FEMNIST (1x28x28 grayscale, 62 classes)."""

from __future__ import annotations

from torch import nn


class FemnistCNN(nn.Module):
    """Conv-ReLU-Pool x2 -> FC -> 62 logits (the LEAF reference architecture, slimmed)."""

    def __init__(self, num_classes: int = 62, channels: tuple[int, int] = (16, 32), fc: int = 128):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc),
            nn.ReLU(),
            nn.Linear(fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
