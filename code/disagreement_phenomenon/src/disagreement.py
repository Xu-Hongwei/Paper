from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics


GROUP_ORDER = ("Low-D", "Mid-D", "High-D")
HIGH_D_RELIABILITY_GROUP_ORDER = ("High-D+Low-R", "High-D+High-R")


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m, eps=eps) + 0.5 * _kl_div(q, m, eps=eps)


def sample_disagreement(prob_t: np.ndarray, prob_v: np.ndarray, prob_a: np.ndarray) -> np.ndarray:
    d_tv = jsd(prob_t, prob_v)
    d_ta = jsd(prob_t, prob_a)
    d_va = jsd(prob_v, prob_a)
    return (d_tv + d_ta + d_va) / 3.0


def normalized_reliability(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    prob = np.clip(prob, eps, 1.0)
    entropy = -np.sum(prob * np.log(prob), axis=-1)
    return 1.0 - entropy / np.log(prob.shape[-1])


def sample_reliability(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
) -> dict[str, np.ndarray]:
    r_text = normalized_reliability(prob_t)
    r_vision = normalized_reliability(prob_v)
    r_audio = normalized_reliability(prob_a)
    r_sample = (r_text + r_vision + r_audio) / 3.0
    return {
        "R_text": r_text,
        "R_vision": r_vision,
        "R_audio": r_audio,
        "R_sample": r_sample,
    }


def validation_thresholds(valid_disagreement: np.ndarray) -> tuple[float, float]:
    q33, q66 = np.quantile(valid_disagreement, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def assign_groups(disagreement: np.ndarray, q33: float, q66: float) -> np.ndarray:
    groups = np.empty(disagreement.shape[0], dtype=object)
    groups[disagreement <= q33] = "Low-D"
    groups[(disagreement > q33) & (disagreement <= q66)] = "Mid-D"
    groups[disagreement > q66] = "High-D"
    return groups


def assign_high_d_reliability_groups(
    groups: np.ndarray,
    r_sample: np.ndarray,
) -> np.ndarray:
    reliability_groups = np.full(groups.shape[0], "", dtype=object)
    high_indices = np.where(groups == "High-D")[0]
    if high_indices.shape[0] == 0:
        return reliability_groups
    sorted_high = high_indices[np.argsort(r_sample[high_indices])]
    split = max(1, sorted_high.shape[0] // 2)
    reliability_groups[sorted_high[:split]] = "High-D+Low-R"
    reliability_groups[sorted_high[split:]] = "High-D+High-R"
    return reliability_groups


def build_group_frame(
    test_pred: dict[str, np.ndarray],
    disagreement: np.ndarray,
    groups: np.ndarray,
    reliability: dict[str, np.ndarray] | None = None,
    reliability_groups: np.ndarray | None = None,
) -> pd.DataFrame:
    payload = {
        "index": test_pred["index"],
        "label_reg": test_pred["y_reg"],
        "label_cls": test_pred["y_true"],
        "prediction": test_pred["y_pred"],
        "D_sample": disagreement,
        "group": groups,
    }
    if reliability is not None:
        payload.update(reliability)
    if reliability_groups is not None:
        payload["high_d_reliability_group"] = reliability_groups
    return pd.DataFrame(payload)


def grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_order: tuple[str, ...] = GROUP_ORDER,
    include_overall: bool = True,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for group in group_order:
        mask = groups == group
        if mask.sum() == 0:
            results[group] = {"acc": float("nan"), "macro_f1": float("nan"), "n": 0.0}
            continue
        metrics = classification_metrics(y_true[mask], y_pred[mask])
        metrics["n"] = float(mask.sum())
        results[group] = metrics
    if include_overall:
        overall = classification_metrics(y_true, y_pred)
        overall["n"] = float(y_true.shape[0])
        results["Overall"] = overall
    return results


def rows_for_method(
    method: str,
    metrics: dict[str, dict[str, float]],
    *,
    lambda_align: float | None = None,
    group_order: tuple[str, ...] = GROUP_ORDER,
    include_overall: bool = True,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    groups = (*group_order, "Overall") if include_overall else group_order
    for group in groups:
        row: dict[str, float | str] = {
            "method": method,
            "group": group,
            "n": metrics[group]["n"],
            "acc": metrics[group]["acc"],
            "macro_f1": metrics[group]["macro_f1"],
        }
        if lambda_align is not None:
            row["lambda_align"] = lambda_align
        rows.append(row)
    return rows
