"""A multimodal fixture for diagnosing and repairing a changed encoder contract."""

import torch
from torch import nn

from .encoders import TextEncoder


class VideoEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_projection = nn.Linear(3 * 8 * 8, 48)

    def forward(self, video, lengths):
        batch, frames, channels, height, width = video.shape
        flattened = video.reshape(batch, frames, channels * height * width)
        frame_features = torch.tanh(self.frame_projection(flattened))
        valid = torch.arange(frames, device=video.device)[None, :] < lengths[:, None]
        pooled = (frame_features * valid.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / lengths.clamp_min(1).unsqueeze(-1)
        return frame_features, pooled, valid


class ContractFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # This v1 boundary expects 32 features; VideoEncoderV2 emits 48.
        self.video_projection = nn.Linear(32, 16)
        self.text_projection = nn.Linear(16, 16)
        self.combine = nn.Sequential(nn.Linear(32, 16), nn.GELU())

    def forward(self, video_features, text_features):
        video = self.video_projection(video_features)
        text = self.text_projection(text_features)
        return self.combine(torch.cat([video, text], dim=-1)), video, text


class ContractRepairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = VideoEncoderV2()
        self.text_encoder = TextEncoder(width=16)
        self.fusion = ContractFusion()
        self.classification_head = nn.Linear(16, 4)
        self.temporal_head = nn.Linear(48, 4)

    def forward(self, video, tokens, video_lengths, text_mask):
        frame_features, video_features, valid_frames = self.video_encoder(video, video_lengths)
        text_features = self.text_encoder(tokens, text_mask)
        fused, video_projected, text_projected = self.fusion(video_features, text_features)
        return {
            "logits": self.classification_head(fused),
            "frame_logits": self.temporal_head(frame_features),
            "video_embedding": video_projected,
            "text_embedding": text_projected,
            "valid_frames": valid_frames,
        }


def build_model():
    return ContractRepairModel()
