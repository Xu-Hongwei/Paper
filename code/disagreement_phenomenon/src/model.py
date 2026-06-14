from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        elif x.ndim != 2:
            raise ValueError(f"Expected [B, D] or [B, T, D], got {tuple(x.shape)}")
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        audio_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, hidden_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, dropout)

        self.text_head = nn.Linear(hidden_dim, num_classes)
        self.vision_head = nn.Linear(hidden_dim, num_classes)
        self.audio_head = nn.Linear(hidden_dim, num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h_t = self.text_encoder(batch["text"])
        h_v = self.vision_encoder(batch["vision"])
        h_a = self.audio_encoder(batch["audio"])
        return {"text": h_t, "vision": h_v, "audio": h_a}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        enc = self.encode(batch)
        fused = torch.cat([enc["text"], enc["vision"], enc["audio"]], dim=-1)
        return {
            "h_t": enc["text"],
            "h_v": enc["vision"],
            "h_a": enc["audio"],
            "logits_t": self.text_head(enc["text"]),
            "logits_v": self.vision_head(enc["vision"]),
            "logits_a": self.audio_head(enc["audio"]),
            "logits_f": self.fusion_head(fused),
        }


def unconditional_alignment_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    h_t = outputs["h_t"]
    h_v = outputs["h_v"]
    h_a = outputs["h_a"]
    loss_tv = 1.0 - F.cosine_similarity(h_t, h_v, dim=-1)
    loss_ta = 1.0 - F.cosine_similarity(h_t, h_a, dim=-1)
    loss_va = 1.0 - F.cosine_similarity(h_v, h_a, dim=-1)
    return (loss_tv + loss_ta + loss_va).mean()
