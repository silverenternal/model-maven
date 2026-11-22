from torch import nn

from .encoders import TextEncoder, VideoEncoder
from .fusion import CrossModalFusion


class VideoTextModel(nn.Module):
    def __init__(self, classes=4, width=16):
        super().__init__()
        self.video_encoder = VideoEncoder(width)
        self.text_encoder = TextEncoder(width=width)
        self.fusion = CrossModalFusion(width)
        self.classification_head = nn.Linear(width, classes)
        self.temporal_head = nn.Linear(width, classes)

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


def build_model(classes=4, width=16):
    return VideoTextModel(classes=classes, width=width)
