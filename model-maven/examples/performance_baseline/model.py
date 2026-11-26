import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
    def forward(self, x):
        return x + self.layers(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*(Residual(1024) for _ in range(4)))
        self.head = nn.Linear(1024, 100)
    def forward(self, x):
        return self.head(self.blocks(x))

def model():
    result = Model()
    result.load_state_dict(torch.load('weights.pt', map_location='cpu', weights_only=True))
    return result

def batch():
    return torch.randn(16, 1024)
