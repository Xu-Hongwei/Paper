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
    label_aware_copa_loss,
    unconditional_alignment_loss,
    unconditional_infonce_loss,
)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    eta_unimodal: float = 0.0,
    lambda_align: float = 0.0,
    lambda_nce: float = 0.0,
    nce_temperature: float = 0.1,
    nce_pair_mode: str = "text_anchor",
    lambda_copa: float = 0.0,
    tau_agreement: float = 0.1,
    copa_proto_weight: float = 1.0,
    copa_agr_weight: float = 1.0,
    copa_comp_weight: float = 0.5,
    copa_comp_margin: float = 0.2,
    copa_orth_weight: float = 0.01,
    copa_gate_type: str = "label_support",
    copa_gate_metric: str = "prob_jsd",
    copa_kernel_bandwidth: str | float = "median",
) -> torch.Tensor:
    loss = F.cross_entropy(outputs["logits_f"], labels)
    if eta_unimodal > 0:
        loss = loss + eta_unimodal * (
            F.cross_entropy(outputs["logits_t"], labels)
            + F.cross_entropy(outputs["logits_v"], labels)
            + F.cross_entropy(outputs["logits_a"], labels)
        )
    if lambda_align > 0:
        loss = loss + lambda_align * unconditional_alignment_loss(outputs)
    if lambda_nce > 0:
        loss = loss + lambda_nce * unconditional_infonce_loss(
            outputs,
            temperature=nce_temperature,
            pair_mode=nce_pair_mode,
        )
    if lambda_copa > 0:
        copa_loss, _ = label_aware_copa_loss(
            outputs,
            labels,
            tau_agreement=tau_agreement,
            proto_weight=copa_proto_weight,
            agr_weight=copa_agr_weight,
            comp_weight=copa_comp_weight,
            comp_margin=copa_comp_margin,
            orth_weight=copa_orth_weight,
            gate_type=copa_gate_type,
            gate_metric=copa_gate_metric,
            kernel_bandwidth=copa_kernel_bandwidth,
        )
        loss = loss + lambda_copa * copa_loss
    return loss


@torch.no_grad()
def predict(
    model: MultimodalClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
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

    return {
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


def evaluate(
    model: MultimodalClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
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
    lambda_nce: float = 0.0,
    nce_temperature: float = 0.1,
    nce_pair_mode: str = "text_anchor",
    lambda_copa: float = 0.0,
    tau_agreement: float = 0.1,
    copa_proto_weight: float = 1.0,
    copa_agr_weight: float = 1.0,
    copa_comp_weight: float = 0.5,
    copa_comp_margin: float = 0.2,
    copa_orth_weight: float = 0.01,
    copa_gate_type: str = "label_support",
    copa_gate_metric: str = "prob_jsd",
    copa_kernel_bandwidth: str | float = "median",
    patience: int = 8,
    desc: str = "train",
    show_progress: bool = True,
) -> tuple[MultimodalClassifier, dict[str, float]]:
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
                lambda_nce=lambda_nce,
                nce_temperature=nce_temperature,
                nce_pair_mode=nce_pair_mode,
                lambda_copa=lambda_copa,
                tau_agreement=tau_agreement,
                copa_proto_weight=copa_proto_weight,
                copa_agr_weight=copa_agr_weight,
                copa_comp_weight=copa_comp_weight,
                copa_comp_margin=copa_comp_margin,
                copa_orth_weight=copa_orth_weight,
                copa_gate_type=copa_gate_type,
                copa_gate_metric=copa_gate_metric,
                copa_kernel_bandwidth=copa_kernel_bandwidth,
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
