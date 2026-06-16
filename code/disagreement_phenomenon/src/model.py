from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ModalityEncoder(nn.Module):
    """单模态编码器：两层 MLP，将原始特征映射到共享隐空间。

    支持 [B, D] 和 [B, T, D] 两种输入格式，后者会沿时间维取平均。
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        """初始化单模态编码器。

        Args:
            input_dim: 输入特征维度。
            hidden_dim: 隐层维度。
            dropout: Dropout 比率。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征，shape [B, D] 或 [B, T, D]。

        Returns:
            编码后的隐层特征，shape [B, hidden_dim]。

        Raises:
            ValueError: 输入维度不为 2 或 3 时抛出。
        """
        if x.ndim == 3:
            x = x.mean(dim=1)
        elif x.ndim != 2:
            raise ValueError(f"Expected [B, D] or [B, T, D], got {tuple(x.shape)}")
        return self.net(x)


class ProjectionHead(nn.Module):
    """投影头：两层 MLP，将隐层特征映射到 InfoNCE 对比学习的投影空间。"""

    def __init__(self, hidden_dim: int, proj_dim: int, dropout: float) -> None:
        """初始化投影头。

        Args:
            hidden_dim: 输入隐层维度。
            proj_dim: 投影输出维度。
            dropout: Dropout 比率。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 隐层特征，shape [B, hidden_dim]。

        Returns:
            投影特征，shape [B, proj_dim]。
        """
        return self.net(x)


class MultimodalClassifier(nn.Module):
    """多模态分类器：三模态独立编码 + 可选直接对齐注入 + 融合分类头。

    架构：
    1. 三个单模态编码器分别提取 h_t, h_v, h_a
    2. 可选：direct_add 将其他模态信息直接注入当前模态表示
    3. 三个单模态分类头 + 一个拼接融合分类头
    4. 可选的 InfoNCE 投影头用于对比学习
    """

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
        use_nce_projection: bool = True,
        nce_proj_dim: int = 128,
    ) -> None:
        """初始化多模态分类器。

        Args:
            text_dim: 文本模态输入维度。
            vision_dim: 视觉模态输入维度。
            audio_dim: 音频模态输入维度。
            hidden_dim: 共享隐层维度。
            dropout: Dropout 比率。
            num_classes: 分类类别数。
            direct_add_alpha: 直接对齐注入强度，0 表示不注入。
            direct_add_pair_mode: 注入模式——
                "text_anchor" 以文本为锚点、仅注入到视觉和音频；
                "full_pair" 三模态平均后注入到全部三模态；
                "balanced" 三模态经 LayerNorm 后平均再注入。
            use_nce_projection: 是否使用 InfoNCE 投影头。
            nce_proj_dim: InfoNCE 投影维度。

        Raises:
            ValueError: direct_add_pair_mode 不合法时抛出。
        """
        super().__init__()
        if direct_add_pair_mode not in {"text_anchor", "full_pair", "balanced"}:
            raise ValueError(
                "direct_add_pair_mode must be 'text_anchor', 'full_pair', or 'balanced'."
            )
        self.direct_add_alpha = direct_add_alpha
        self.direct_add_pair_mode = direct_add_pair_mode
        self.use_nce_projection = use_nce_projection
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, hidden_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim, dropout)
        self.add_norm_t = nn.LayerNorm(hidden_dim)
        self.add_norm_v = nn.LayerNorm(hidden_dim)
        self.add_norm_a = nn.LayerNorm(hidden_dim)
        proj_dim = nce_proj_dim if nce_proj_dim > 0 else hidden_dim
        if use_nce_projection:
            self.nce_proj_t = ProjectionHead(hidden_dim, proj_dim, dropout)
            self.nce_proj_v = ProjectionHead(hidden_dim, proj_dim, dropout)
            self.nce_proj_a = ProjectionHead(hidden_dim, proj_dim, dropout)
        else:
            self.nce_proj_t = nn.Identity()
            self.nce_proj_v = nn.Identity()
            self.nce_proj_a = nn.Identity()

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
        """仅编码三模态特征，不计算分类 logits。

        Args:
            batch: 包含 "text", "vision", "audio" 键的字典，
                   每个值 shape [B, D] 或 [B, T, D]。

        Returns:
            dict，包含 "text", "vision", "audio" 三个键，
            每个值为编码后的隐层特征 shape [B, hidden_dim]。
        """
        h_t = self.text_encoder(batch["text"])
        h_v = self.vision_encoder(batch["vision"])
        h_a = self.audio_encoder(batch["audio"])
        return {"text": h_t, "vision": h_v, "audio": h_a}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """前向传播：编码 → 可选直接对齐注入 → 分类 / 投影。

        Args:
            batch: 包含 "text", "vision", "audio" 键的字典。

        Returns:
            dict，包含：
            - h_t/h_v/h_a: 隐层特征 [B, hidden_dim]
            - z_t/z_v/z_a: InfoNCE 投影特征 [B, proj_dim]
            - logits_t/logits_v/logits_a: 单模态分类 logits [B, num_classes]
            - logits_f: 融合分类 logits [B, num_classes]
        """
        enc = self.encode(batch)
        fuse_text = enc["text"]
        fuse_vision = enc["vision"]
        fuse_audio = enc["audio"]
        if self.direct_add_alpha > 0:
            if self.direct_add_pair_mode == "text_anchor":
                fuse_vision = fuse_vision + self.direct_add_alpha * enc["text"]
                fuse_audio = fuse_audio + self.direct_add_alpha * enc["text"]
            elif self.direct_add_pair_mode == "balanced":
                aligned = (
                    self.add_norm_t(enc["text"])
                    + self.add_norm_v(enc["vision"])
                    + self.add_norm_a(enc["audio"])
                ) / 3.0
                fuse_text = fuse_text + self.direct_add_alpha * aligned
                fuse_vision = fuse_vision + self.direct_add_alpha * aligned
                fuse_audio = fuse_audio + self.direct_add_alpha * aligned
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
            "z_t": self.nce_proj_t(enc["text"]),
            "z_v": self.nce_proj_v(enc["vision"]),
            "z_a": self.nce_proj_a(enc["audio"]),
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
    """无条件对齐损失：最小化成对模态隐层特征之间的余弦距离。

    使用 1 - cosine_similarity 作为逐样本损失，不依赖标签。

    Args:
        outputs: 模型输出字典，需包含 "h_t", "h_v", "h_a"。
        pair_mode: "text_anchor" 仅对齐 (t,v) 和 (t,a)；
                   "full_pair" 额外对齐 (v,a)。

    Returns:
        标量对齐损失。

    Raises:
        ValueError: pair_mode 不合法时抛出。
    """
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
    """双向 InfoNCE 损失：left→right 和 right→left 的交叉熵平均值。

    正样本为同批次相同索引的配对样本，负样本为批次内其他所有样本。

    Args:
        left: 第一个模态的特征，shape [B, D]。
        right: 第二个模态的特征，shape [B, D]。
        temperature: 温度参数，控制分布的锐度。

    Returns:
        标量 InfoNCE 损失。若 batch_size < 2 则返回 0。

    Raises:
        ValueError: temperature <= 0、batch_size 不一致或特征维度不一致时抛出。
    """
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
    """无条件 InfoNCE 损失：在各模态对之间做双向对比学习。

    优先使用投影特征 (z_*)，若不存在则回退到隐层特征 (h_*)。

    Args:
        outputs: 模型输出字典，优先取 "z_t"/"z_v"/"z_a"，回退到 "h_t"/"h_v"/"h_a"。
        temperature: InfoNCE 温度参数。
        pair_mode: "text_anchor" 仅对 (t,a) 和 (t,v) 计算；
                   "full_pair" 额外对 (a,v) 计算。

    Returns:
        标量 InfoNCE 损失。

    Raises:
        ValueError: pair_mode 不合法时抛出。
    """
    h_t = outputs.get("z_t", outputs["h_t"])
    h_v = outputs.get("z_v", outputs["h_v"])
    h_a = outputs.get("z_a", outputs["h_a"])
    pairs = [
        _bidirectional_infonce(h_t, h_a, temperature),
        _bidirectional_infonce(h_t, h_v, temperature),
    ]
    if pair_mode == "full_pair":
        pairs.append(_bidirectional_infonce(h_a, h_v, temperature))
    elif pair_mode != "text_anchor":
        raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")
    return torch.stack(pairs).mean()
