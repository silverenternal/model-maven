import torch
from torch import nn


class VideoEncoder(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.frame_projection = nn.Linear(3 * 8 * 8, width)

    def forward(self, video, lengths):
        batch, frames, channels, height, width = video.shape
        flattened = video.reshape(batch, frames, channels * height * width)
        features = torch.tanh(self.frame_projection(flattened))
        valid = torch.arange(frames, device=video.device)[None, :] < lengths[:, None]
        pooled = (features * valid.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / lengths.clamp_min(1).unsqueeze(-1)
        return features, pooled, valid


class TextEncoder(nn.Module):
    def __init__(self, vocabulary=32, width=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)

    def forward(self, tokens, mask):
        features = self.embedding(tokens)
        weights = mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return pooled
