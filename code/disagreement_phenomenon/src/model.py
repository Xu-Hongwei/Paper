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
        direct_add_alpha: float = 0.0,
        direct_add_pair_mode: str = "text_anchor",
    ) -> None:
        super().__init__()
        if direct_add_pair_mode not in {"text_anchor", "full_pair"}:
            raise ValueError("direct_add_pair_mode must be 'text_anchor' or 'full_pair'.")
        self.direct_add_alpha = direct_add_alpha
        self.direct_add_pair_mode = direct_add_pair_mode
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
        fuse_text = enc["text"]
        fuse_vision = enc["vision"]
        fuse_audio = enc["audio"]
        if self.direct_add_alpha > 0:
            if self.direct_add_pair_mode == "text_anchor":
                fuse_vision = fuse_vision + self.direct_add_alpha * enc["text"]
                fuse_audio = fuse_audio + self.direct_add_alpha * enc["text"]
            else:
                aligned = (enc["text"] + enc["vision"] + enc["audio"]) / 3.0
                fuse_text = fuse_text + self.direct_add_alpha * aligned
                fuse_vision = fuse_vision + self.direct_add_alpha * aligned
                fuse_audio = fuse_audio + self.direct_add_alpha * aligned
        fused = torch.cat([fuse_text, fuse_vision, fuse_audio], dim=-1)
        return {
            "h_t": enc["text"],
            "h_v": enc["vision"],
            "h_a": enc["audio"],
            "logits_t": self.text_head(enc["text"]),
            "logits_v": self.vision_head(enc["vision"]),
            "logits_a": self.audio_head(enc["audio"]),
            "logits_f": self.fusion_head(fused),
        }


def unconditional_alignment_loss(
    outputs: dict[str, torch.Tensor],
    *,
    pair_mode: str = "text_anchor",
) -> torch.Tensor:
    h_t = outputs["h_t"]
    h_v = outputs["h_v"]
    h_a = outputs["h_a"]
    loss_tv = 1.0 - F.cosine_similarity(h_t, h_v, dim=-1)
    loss_ta = 1.0 - F.cosine_similarity(h_t, h_a, dim=-1)
    pairs = [loss_ta, loss_tv]
    if pair_mode == "full_pair":
        pairs.append(1.0 - F.cosine_similarity(h_v, h_a, dim=-1))
    elif pair_mode != "text_anchor":
        raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")
    return torch.stack(pairs, dim=0).mean()


def _bidirectional_infonce(
    left: torch.Tensor,
    right: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if left.shape[0] != right.shape[0]:
        raise ValueError("InfoNCE pairs must have the same batch size.")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError(
            "InfoNCE pairs must have the same feature dimension, "
            f"got {left.shape[-1]} and {right.shape[-1]}."
        )
    if left.shape[0] < 2:
        return left.new_zeros(())
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.T / temperature
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def unconditional_infonce_loss(
    outputs: dict[str, torch.Tensor],
    *,
    temperature: float = 0.1,
    pair_mode: str = "text_anchor",
) -> torch.Tensor:
    h_t = outputs["h_t"]
    h_v = outputs["h_v"]
    h_a = outputs["h_a"]
    pairs = [
        _bidirectional_infonce(h_t, h_a, temperature),
        _bidirectional_infonce(h_t, h_v, temperature),
    ]
    if pair_mode == "full_pair":
        pairs.append(_bidirectional_infonce(h_a, h_v, temperature))
    elif pair_mode != "text_anchor":
        raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")
    return torch.stack(pairs).mean()
