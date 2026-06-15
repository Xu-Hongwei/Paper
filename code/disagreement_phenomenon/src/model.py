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
    ) -> None:
        super().__init__()
        self.direct_add_alpha = direct_add_alpha
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, hidden_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, dropout)

        self.text_head = nn.Linear(hidden_dim, num_classes)
        self.vision_head = nn.Linear(hidden_dim, num_classes)
        self.audio_head = nn.Linear(hidden_dim, num_classes)
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
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
            "class_prototypes": self.class_prototypes,
        }


def unconditional_alignment_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    h_t = outputs["h_t"]
    h_v = outputs["h_v"]
    h_a = outputs["h_a"]
    loss_tv = 1.0 - F.cosine_similarity(h_t, h_v, dim=-1)
    loss_ta = 1.0 - F.cosine_similarity(h_t, h_a, dim=-1)
    loss_va = 1.0 - F.cosine_similarity(h_v, h_a, dim=-1)
    return (loss_tv + loss_ta + loss_va).mean()


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl_div(p, midpoint, eps=eps) + 0.5 * _kl_div(q, midpoint, eps=eps)


def _confidence(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prob = prob.clamp_min(eps)
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)
    normalizer = torch.log(
        torch.tensor(float(prob.shape[-1]), device=prob.device, dtype=prob.dtype)
    )
    return 1.0 - entropy / normalizer


def _label_support(prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return prob.gather(1, labels.view(-1, 1)).squeeze(1)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sum(value * weight) / weight.sum().clamp_min(eps)


def label_aware_copa_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    tau_agreement: float = 0.1,
    proto_weight: float = 1.0,
    agr_weight: float = 1.0,
    comp_weight: float = 0.5,
    comp_margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if tau_agreement <= 0:
        raise ValueError("tau_agreement must be positive.")

    prob_t = torch.softmax(outputs["logits_t"], dim=-1)
    prob_v = torch.softmax(outputs["logits_v"], dim=-1)
    prob_a = torch.softmax(outputs["logits_a"], dim=-1)

    # Use the gates as reliability estimates, not as objectives to optimize directly.
    with torch.no_grad():
        c_t = _confidence(prob_t)
        c_v = _confidence(prob_v)
        c_a = _confidence(prob_a)
        s_t = _label_support(prob_t, labels)
        s_v = _label_support(prob_v, labels)
        s_a = _label_support(prob_a, labels)
        r_t = c_t * s_t
        r_v = c_v * s_v
        r_a = c_a * s_a

        a_tv = torch.exp(-_jsd(prob_t, prob_v) / tau_agreement)
        a_ta = torch.exp(-_jsd(prob_t, prob_a) / tau_agreement)
        a_va = torch.exp(-_jsd(prob_v, prob_a) / tau_agreement)

        q_tv = c_t * c_v
        q_ta = c_t * c_a
        q_va = c_v * c_a
        b_tv = torch.maximum(s_t, s_v)
        b_ta = torch.maximum(s_t, s_a)
        b_va = torch.maximum(s_v, s_a)
        g_tv_agr = q_tv * s_t * s_v * a_tv
        g_ta_agr = q_ta * s_t * s_a * a_ta
        g_va_agr = q_va * s_v * s_a * a_va
        g_tv_comp = q_tv * b_tv * (1.0 - a_tv)
        g_ta_comp = q_ta * b_ta * (1.0 - a_ta)
        g_va_comp = q_va * b_va * (1.0 - a_va)

    h_t = outputs["h_t"]
    h_v = outputs["h_v"]
    h_a = outputs["h_a"]
    prototypes = outputs["class_prototypes"][labels]

    proto_t = _weighted_mean(1.0 - F.cosine_similarity(h_t, prototypes, dim=-1), r_t)
    proto_v = _weighted_mean(1.0 - F.cosine_similarity(h_v, prototypes, dim=-1), r_v)
    proto_a = _weighted_mean(1.0 - F.cosine_similarity(h_a, prototypes, dim=-1), r_a)
    proto_loss = (proto_t + proto_v + proto_a) / 3.0

    cos_tv = F.cosine_similarity(h_t, h_v, dim=-1)
    cos_ta = F.cosine_similarity(h_t, h_a, dim=-1)
    cos_va = F.cosine_similarity(h_v, h_a, dim=-1)
    agr_loss = (
        _weighted_mean(1.0 - cos_tv, g_tv_agr)
        + _weighted_mean(1.0 - cos_ta, g_ta_agr)
        + _weighted_mean(1.0 - cos_va, g_va_agr)
    ) / 3.0
    comp_loss = (
        _weighted_mean(F.relu(cos_tv - comp_margin), g_tv_comp)
        + _weighted_mean(F.relu(cos_ta - comp_margin), g_ta_comp)
        + _weighted_mean(F.relu(cos_va - comp_margin), g_va_comp)
    ) / 3.0

    total = proto_weight * proto_loss + agr_weight * agr_loss + comp_weight * comp_loss
    stats = {
        "copa_proto": proto_loss.detach(),
        "copa_agr": agr_loss.detach(),
        "copa_comp": comp_loss.detach(),
        "copa_gate_agr": ((g_tv_agr + g_ta_agr + g_va_agr) / 3.0).mean(),
        "copa_gate_comp": ((g_tv_comp + g_ta_comp + g_va_comp) / 3.0).mean(),
    }
    return total, stats
