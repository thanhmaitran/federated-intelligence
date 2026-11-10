"""Tabular models: logistic regression and a small MLP (both emit class logits)."""

from __future__ import annotations

from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = num_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
