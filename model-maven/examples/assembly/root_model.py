"""User-owned root model; the generator does not edit this file."""

from torch import nn


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(4, classes)

    def forward(self, inputs):
        return self.head(self.encoder(inputs))


def build_model(encoder: nn.Module, classes: int = 3) -> nn.Module:
    return Classifier(encoder=encoder, classes=classes)
