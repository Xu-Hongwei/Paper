from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import classification_metrics
from .model import (
    MultimodalClassifier,
    unconditional_alignment_loss,
    unconditional_infonce_loss,
)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """将 batch 字典中所有 tensor 移动到指定设备。

    Args:
        batch: 键为字符串、值为 tensor 的字典。
        device: 目标设备。

    Returns:
        移动后的字典。
    """
    return {key: value.to(device) for key, value in batch.items()}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    eta_unimodal: float = 0.0,
    lambda_align: float = 0.0,
    align_pair_mode: str = "text_anchor",
    lambda_nce: float = 0.0,
    nce_temperature: float = 0.1,
    nce_pair_mode: str = "text_anchor",
    lambda_dynamic_weight: float = 0.0,
    dynamic_weight_epsilon: float = 1e-4,
) -> torch.Tensor:
    """计算多任务训练损失。

    损失 = 融合分类 CE + η * 单模态 CE 平均 + λ_align * 对齐损失 + λ_nce * InfoNCE 损失。

    Args:
        outputs: 模型前向输出字典，需含 logits_f, logits_t, logits_v, logits_a 及隐层特征。
        labels: 分类标签，shape [B]。
        eta_unimodal: 单模态分类损失权重（0 表示不使用）。
        lambda_align: 无条件对齐损失权重（0 表示不使用）。
        align_pair_mode: 对齐损失的模态对模式。
        lambda_nce: InfoNCE 对比损失权重（0 表示不使用）。
        nce_temperature: InfoNCE 温度。
        nce_pair_mode: InfoNCE 的模态对模式。
        lambda_dynamic_weight: EMOE-style 动态权重监督损失权重。
        dynamic_weight_epsilon: 误差反比权重的数值稳定项。

    Returns:
        标量损失值。
    """
    loss = F.cross_entropy(outputs["logits_f"], labels)
    if eta_unimodal > 0:
        unimodal_loss = (
            F.cross_entropy(outputs["logits_t"], labels)
            + F.cross_entropy(outputs["logits_v"], labels)
            + F.cross_entropy(outputs["logits_a"], labels)
        ) / 3.0
        loss = loss + eta_unimodal * unimodal_loss
    if lambda_align > 0:
        loss = loss + lambda_align * unconditional_alignment_loss(
            outputs,
            pair_mode=align_pair_mode,
        )
    if lambda_nce > 0:
        loss = loss + lambda_nce * unconditional_infonce_loss(
            outputs,
            temperature=nce_temperature,
            pair_mode=nce_pair_mode,
        )
    if lambda_dynamic_weight > 0:
        if "channel_weight" not in outputs:
            raise ValueError("lambda_dynamic_weight requires outputs['channel_weight'].")
        ce_t = F.cross_entropy(outputs["logits_t"], labels, reduction="none")
        ce_v = F.cross_entropy(outputs["logits_v"], labels, reduction="none")
        ce_a = F.cross_entropy(outputs["logits_a"], labels, reduction="none")
        inv_error = torch.stack(
            [
                1.0 / (ce_t + dynamic_weight_epsilon),
                1.0 / (ce_v + dynamic_weight_epsilon),
                1.0 / (ce_a + dynamic_weight_epsilon),
            ],
            dim=-1,
        )
        target_weight = inv_error / inv_error.sum(dim=-1, keepdim=True).clamp_min(
            dynamic_weight_epsilon
        )
        loss = loss + lambda_dynamic_weight * F.mse_loss(
            outputs["channel_weight"],
            target_weight.detach(),
        )
    return loss


@torch.no_grad()
def predict(
    model: MultimodalClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """在给定数据上做推理，返回标签、各模态概率及隐层特征。

    Args:
        model: 多模态分类器。
        loader: 数据加载器。
        device: 计算设备。

    Returns:
        dict，包含：
        - y_true: 真实分类标签 [N]
        - y_reg: 回归标签 [N]
        - y_pred: 融合预测标签 [N]
        - index: 样本索引 [N]
        - prob_t/prob_v/prob_a/prob_f: 各模态及融合 softmax 概率 [N, K]
        - h_t/h_v/h_a: 各模态隐层特征 [N, D]
    """
    model.eval()
    y_true: list[np.ndarray] = []
    y_reg: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    probs_t: list[np.ndarray] = []
    probs_v: list[np.ndarray] = []
    probs_a: list[np.ndarray] = []
    probs_f: list[np.ndarray] = []
    feats_t: list[np.ndarray] = []
    feats_v: list[np.ndarray] = []
    feats_a: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch)
        pf = torch.softmax(outputs["logits_f"], dim=-1)
        y_true.append(batch["label_cls"].detach().cpu().numpy())
        y_reg.append(batch["label_reg"].detach().cpu().numpy())
        y_pred.append(pf.argmax(dim=-1).detach().cpu().numpy())
        indices.append(batch["index"].detach().cpu().numpy())
        probs_t.append(torch.softmax(outputs["logits_t"], dim=-1).detach().cpu().numpy())
        probs_v.append(torch.softmax(outputs["logits_v"], dim=-1).detach().cpu().numpy())
        probs_a.append(torch.softmax(outputs["logits_a"], dim=-1).detach().cpu().numpy())
        probs_f.append(pf.detach().cpu().numpy())
        feats_t.append(outputs["h_t"].detach().cpu().numpy())
        feats_v.append(outputs["h_v"].detach().cpu().numpy())
        feats_a.append(outputs["h_a"].detach().cpu().numpy())
        if "channel_weight" in outputs:
            weights.append(outputs["channel_weight"].detach().cpu().numpy())

    result = {
        "y_true": np.concatenate(y_true),
        "y_reg": np.concatenate(y_reg),
        "y_pred": np.concatenate(y_pred),
        "index": np.concatenate(indices),
        "prob_t": np.concatenate(probs_t),
        "prob_v": np.concatenate(probs_v),
        "prob_a": np.concatenate(probs_a),
        "prob_f": np.concatenate(probs_f),
        "h_t": np.concatenate(feats_t),
        "h_v": np.concatenate(feats_v),
        "h_a": np.concatenate(feats_a),
    }
    if weights:
        weight_array = np.concatenate(weights)
        result["w_text"] = weight_array[:, 0]
        result["w_vision"] = weight_array[:, 1]
        result["w_audio"] = weight_array[:, 2]
    return result


def evaluate(
    model: MultimodalClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """评估模型在给定数据上的分类指标。

    Args:
        model: 多模态分类器。
        loader: 数据加载器。
        device: 计算设备。

    Returns:
        dict，包含 "acc", "macro_f1" 等分类指标。
    """
    pred = predict(model, loader, device)
    return classification_metrics(pred["y_true"], pred["y_pred"])


def train_model(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    eta_unimodal: float = 0.0,
    lambda_align: float = 0.0,
    align_pair_mode: str = "text_anchor",
    lambda_nce: float = 0.0,
    nce_temperature: float = 0.1,
    nce_pair_mode: str = "text_anchor",
    lambda_dynamic_weight: float = 0.0,
    dynamic_weight_epsilon: float = 1e-4,
    patience: int = 8,
    desc: str = "train",
    show_progress: bool = True,
) -> tuple[MultimodalClassifier, dict[str, float]]:
    """训练多模态分类器，使用 AdamW + early stopping。

    训练循环：前向 → 多任务损失 → 梯度裁剪 → 参数更新。
    每轮在验证集评估 macro_f1，若 patience 轮无提升则早停。

    Args:
        model: 待训练的多模态分类器。
        train_loader: 训练数据加载器。
        valid_loader: 验证数据加载器。
        device: 计算设备。
        epochs: 最大训练轮数。
        lr: 学习率。
        weight_decay: AdamW 权重衰减。
        eta_unimodal: 单模态分类损失权重。
        lambda_align: 对齐损失权重。
        align_pair_mode: 对齐模态对模式。
        lambda_nce: InfoNCE 损失权重。
        nce_temperature: InfoNCE 温度。
        nce_pair_mode: InfoNCE 模态对模式。
        lambda_dynamic_weight: EMOE-style 动态权重监督损失权重。
        dynamic_weight_epsilon: 动态权重监督中的误差反比稳定项。
        patience: 早停耐心值。
        desc: tqdm 进度条描述。
        show_progress: 是否显示进度条。

    Returns:
        (最佳模型, 最佳验证指标) 的元组。最佳指标包含 "epoch", "acc", "macro_f1"。
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_state = deepcopy(model.state_dict())
    best_metrics: dict[str, float] = {}
    stale_epochs = 0

    progress = tqdm(range(1, epochs + 1), desc=desc, leave=False, disable=not show_progress)
    for epoch in progress:
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            labels = batch["label_cls"]
            loss = compute_loss(
                outputs,
                labels,
                eta_unimodal=eta_unimodal,
                lambda_align=lambda_align,
                align_pair_mode=align_pair_mode,
                lambda_nce=lambda_nce,
                nce_temperature=nce_temperature,
                nce_pair_mode=nce_pair_mode,
                lambda_dynamic_weight=lambda_dynamic_weight,
                dynamic_weight_epsilon=dynamic_weight_epsilon,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size

        metrics = evaluate(model, valid_loader, device)
        score = metrics["macro_f1"]
        progress.set_postfix(
            loss=f"{total_loss / max(seen, 1):.4f}",
            val_f1=f"{metrics['macro_f1']:.4f}",
            val_acc=f"{metrics['acc']:.4f}",
        )

        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_metrics = {"epoch": float(epoch), **metrics}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_metrics
