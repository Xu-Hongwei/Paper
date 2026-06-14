from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import classification_metrics
from .model import MultimodalClassifier, unconditional_alignment_loss


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    eta_unimodal: float = 0.0,
    lambda_align: float = 0.0,
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

    return {
        "y_true": np.concatenate(y_true),
        "y_reg": np.concatenate(y_reg),
        "y_pred": np.concatenate(y_pred),
        "index": np.concatenate(indices),
        "prob_t": np.concatenate(probs_t),
        "prob_v": np.concatenate(probs_v),
        "prob_a": np.concatenate(probs_a),
        "prob_f": np.concatenate(probs_f),
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
