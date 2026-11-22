import torch
from torch import nn


class CrossModalFusion(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        # Deliberately shared by both modalities; the probe should observe two calls.
        self.shared_projection = nn.Linear(width, width)
        self.combine = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())

    def forward(self, video_features, text_features):
        video = self.shared_projection(video_features)
        text = self.shared_projection(text_features)
        return self.combine(torch.cat([video, text], dim=-1)), video, text
