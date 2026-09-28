"""Small RGB CNN used for the first baseline."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Map batches of 32 by 32 RGB images to 43 raw class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 43),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.layers(images)
        return logits
