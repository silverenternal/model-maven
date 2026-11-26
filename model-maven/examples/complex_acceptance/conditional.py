"""An editable two-input model with a user-selected dynamic branch.

The two batch factories use the same deterministic synthetic tensors and differ
only in ``refine``. They are examples, not representative user data.
"""

import torch
from torch import nn

from .model import VideoTextModel


class ConditionalVideoTextModel(VideoTextModel):
    def __init__(self, classes=4, width=16):
        super().__init__(classes=classes, width=width)
        self.refinement = nn.Sequential(nn.Linear(width, width), nn.GELU())

    def forward(self, video, tokens, video_lengths, text_mask, refine=False):
        frame_features, video_features, valid_frames = self.video_encoder(video, video_lengths)
        text_features = self.text_encoder(tokens, text_mask)
        fused, video_projected, text_projected = self.fusion(video_features, text_features)
        if refine:
            fused = self.refinement(fused)
        return {
            "logits": self.classification_head(fused),
            "frame_logits": self.temporal_head(frame_features),
            "video_embedding": video_projected,
            "text_embedding": text_projected,
            "valid_frames": valid_frames,
        }


def build_conditional_model(classes=4, width=16):
    return ConditionalVideoTextModel(classes=classes, width=width)


def _batch(refine):
    generator = torch.Generator().manual_seed(7)
    return {
        "args": [
            torch.randn(2, 3, 3, 8, 8, generator=generator),
            torch.randint(0, 32, (2, 4), generator=generator),
            torch.tensor([3, 2]),
            torch.tensor([[True, True, True, True], [True, True, True, False]]),
        ],
        "kwargs": {"refine": refine},
        "loss_inputs": {"targets": torch.tensor([0, 3])},
    }


def direct_batch():
    return _batch(False)


def refined_batch():
    return _batch(True)
