"""Create local random CPU weights; no download or pretrained data."""
import torch
from model import Model

torch.manual_seed(1)
network = Model()
torch.save(network.state_dict(), 'weights.pt')
print('parameters:', sum(parameter.numel() for parameter in network.parameters()))
