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

        self.text_common = nn.Linear(hidden_dim, hidden_dim)
        self.vision_common = nn.Linear(hidden_dim, hidden_dim)
        self.audio_common = nn.Linear(hidden_dim, hidden_dim)
        self.text_residual = nn.Linear(hidden_dim, hidden_dim)
        self.vision_residual = nn.Linear(hidden_dim, hidden_dim)
        self.audio_residual = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, num_classes)
        self.vision_head = nn.Linear(hidden_dim, num_classes)
        self.audio_head = nn.Linear(hidden_dim, num_classes)
        self.residual_prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
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
        z_t_c = self.text_common(enc["text"])
        z_v_c = self.vision_common(enc["vision"])
        z_a_c = self.audio_common(enc["audio"])
        z_t_r = self.text_residual(enc["text"])
        z_v_r = self.vision_residual(enc["vision"])
        z_a_r = self.audio_residual(enc["audio"])
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
            "z_t_c": z_t_c,
            "z_v_c": z_v_c,
            "z_a_c": z_a_c,
            "z_t_r": z_t_r,
            "z_v_r": z_v_r,
            "z_a_r": z_a_r,
            "logits_t": self.text_head(z_t_c),
            "logits_v": self.vision_head(z_v_c),
            "logits_a": self.audio_head(z_a_c),
            "logits_f": self.fusion_head(fused),
            "residual_prototypes": self.residual_prototypes,
            "class_prototypes": self.residual_prototypes,
        }


def unconditional_alignment_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    h_t = outputs["h_t"]
    h_v = outputs["h_v"]
    h_a = outputs["h_a"]
    loss_tv = 1.0 - F.cosine_similarity(h_t, h_v, dim=-1)
    loss_ta = 1.0 - F.cosine_similarity(h_t, h_a, dim=-1)
    loss_va = 1.0 - F.cosine_similarity(h_v, h_a, dim=-1)
    return (loss_tv + loss_ta + loss_va).mean()


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


def _kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl_div(p, midpoint, eps=eps) + 0.5 * _kl_div(q, midpoint, eps=eps)


def _parse_kernel_bandwidth(bandwidth: str | float) -> float | str:
    if isinstance(bandwidth, str):
        if bandwidth == "median":
            return bandwidth
        return float(bandwidth)
    return float(bandwidth)


def _resolve_kernel_bandwidth(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidth: str | float,
    eps: float = 1e-8,
) -> torch.Tensor:
    parsed = _parse_kernel_bandwidth(bandwidth)
    if parsed == "median":
        features = F.normalize(torch.cat([left, right], dim=0), dim=-1)
        if features.shape[0] < 2:
            return features.new_tensor(1.0)
        distances = torch.pdist(features)
        distances = distances[distances > eps]
        if distances.numel() == 0:
            return features.new_tensor(1.0)
        return distances.median().clamp_min(eps)
    if parsed <= 0:
        raise ValueError("kernel bandwidth must be positive.")
    return left.new_tensor(float(parsed))


def _rbf_point_disagreement(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidth: str | float = "median",
) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(
            "Kernel disagreement pairs must have the same shape, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    sigma = _resolve_kernel_bandwidth(left, right, bandwidth)
    squared = torch.sum(torch.square(left - right), dim=-1)
    return 1.0 - torch.exp(-squared / (2.0 * sigma * sigma))


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


def _gated_common_infonce(
    left: torch.Tensor,
    right: torch.Tensor,
    labels: torch.Tensor,
    gate: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if left.shape != right.shape:
        raise ValueError(
            "Gated InfoNCE pairs must have the same shape, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.shape[0] < 2:
        return left.new_zeros(())

    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.T / temperature
    exp_logits = torch.exp(logits)
    eye = torch.eye(left.shape[0], device=left.device, dtype=torch.bool)
    different_class = labels.view(-1, 1) != labels.view(1, -1)
    denom_mask = different_class | eye
    denom = (exp_logits * denom_mask.to(exp_logits.dtype)).sum(dim=1).clamp_min(eps)
    pos = exp_logits.diagonal().clamp_min(eps)
    loss_lr = -torch.log(pos / denom)

    exp_logits_t = torch.exp(logits.T)
    denom_t = (exp_logits_t * denom_mask.to(exp_logits_t.dtype)).sum(dim=1).clamp_min(eps)
    pos_t = exp_logits_t.diagonal().clamp_min(eps)
    loss_rl = -torch.log(pos_t / denom_t)
    return 0.5 * (_weighted_mean(loss_lr, gate) + _weighted_mean(loss_rl, gate))


def _residual_prototype_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    gate: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if left.shape != right.shape:
        raise ValueError(
            "Residual NCE pairs must have the same shape, "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    residual = torch.abs(left - right)
    logits = F.normalize(residual, dim=-1) @ F.normalize(prototypes, dim=-1).T
    loss = F.cross_entropy(logits / temperature, labels, reduction="none")
    return _weighted_mean(loss, gate)


def _common_residual_orthogonality_loss(
    common: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    common = F.normalize(common, dim=-1)
    residual = F.normalize(residual, dim=-1)
    return torch.square(torch.sum(common * residual, dim=-1)).mean()


def label_aware_copa_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    tau_agreement: float = 0.1,
    proto_weight: float = 1.0,
    agr_weight: float = 1.0,
    comp_weight: float = 0.5,
    comp_margin: float = 0.2,
    orth_weight: float = 0.01,
    gate_metric: str = "prob_jsd",
    kernel_bandwidth: str | float = "median",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if tau_agreement <= 0:
        raise ValueError("tau_agreement must be positive.")

    prob_t = torch.softmax(outputs["logits_t"], dim=-1)
    prob_v = torch.softmax(outputs["logits_v"], dim=-1)
    prob_a = torch.softmax(outputs["logits_a"], dim=-1)
    z_t_c = outputs["z_t_c"]
    z_v_c = outputs["z_v_c"]
    z_a_c = outputs["z_a_c"]
    z_t_r = outputs["z_t_r"]
    z_v_r = outputs["z_v_r"]
    z_a_r = outputs["z_a_r"]

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

        if gate_metric == "prob_jsd":
            d_tv = _jsd(prob_t, prob_v)
            d_ta = _jsd(prob_t, prob_a)
        elif gate_metric == "kernel_mmd":
            d_tv = _rbf_point_disagreement(z_t_c, z_v_c, kernel_bandwidth)
            d_ta = _rbf_point_disagreement(z_t_c, z_a_c, kernel_bandwidth)
        else:
            raise ValueError("gate_metric must be 'prob_jsd' or 'kernel_mmd'.")
        a_tv = torch.exp(-d_tv / tau_agreement)
        a_ta = torch.exp(-d_ta / tau_agreement)

        q_tv = c_t * c_v
        q_ta = c_t * c_a
        b_tv = torch.maximum(s_t, s_v)
        b_ta = torch.maximum(s_t, s_a)
        g_tv_agr = q_tv * s_t * s_v * a_tv
        g_ta_agr = q_ta * s_t * s_a * a_ta
        g_tv_comp = q_tv * b_tv * (1.0 - a_tv)
        g_ta_comp = q_ta * b_ta * (1.0 - a_ta)

    residual_prototypes = outputs["residual_prototypes"]

    agr_tv = _gated_common_infonce(z_t_c, z_v_c, labels, g_tv_agr, tau_agreement)
    agr_ta = _gated_common_infonce(z_t_c, z_a_c, labels, g_ta_agr, tau_agreement)
    agr_loss = 0.5 * (agr_tv + agr_ta)
    dis_tv = _residual_prototype_nce(
        z_t_r,
        z_v_r,
        residual_prototypes,
        labels,
        g_tv_comp,
        tau_agreement,
    )
    dis_ta = _residual_prototype_nce(
        z_t_r,
        z_a_r,
        residual_prototypes,
        labels,
        g_ta_comp,
        tau_agreement,
    )
    dis_loss = 0.5 * (dis_tv + dis_ta)
    orth_loss = (
        _common_residual_orthogonality_loss(z_t_c, z_t_r)
        + _common_residual_orthogonality_loss(z_v_c, z_v_r)
        + _common_residual_orthogonality_loss(z_a_c, z_a_r)
    ) / 3.0

    total = (
        agr_weight * agr_loss
        + comp_weight * proto_weight * dis_loss
        + orth_weight * orth_loss
    )
    stats = {
        "copa_proto": dis_loss.detach(),
        "copa_agr": agr_loss.detach(),
        "copa_comp": dis_loss.detach(),
        "copa_orth": orth_loss.detach(),
        "copa_gate_agr": ((g_tv_agr + g_ta_agr) / 2.0).mean(),
        "copa_gate_comp": ((g_tv_comp + g_ta_comp) / 2.0).mean(),
    }
    return total, stats
