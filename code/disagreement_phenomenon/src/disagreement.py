from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics


GROUP_ORDER = ("Low-D", "Mid-D", "High-D")


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


def validation_thresholds(valid_disagreement: np.ndarray) -> tuple[float, float]:
    q33, q66 = np.quantile(valid_disagreement, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def assign_groups(disagreement: np.ndarray, q33: float, q66: float) -> np.ndarray:
    groups = np.empty(disagreement.shape[0], dtype=object)
    groups[disagreement <= q33] = "Low-D"
    groups[(disagreement > q33) & (disagreement <= q66)] = "Mid-D"
    groups[disagreement > q66] = "High-D"
    return groups


def build_group_frame(
    test_pred: dict[str, np.ndarray],
    disagreement: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": test_pred["index"],
            "label_reg": test_pred["y_reg"],
            "label_cls": test_pred["y_true"],
            "prediction": test_pred["y_pred"],
            "D_sample": disagreement,
            "group": groups,
        }
    )


def grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for group in GROUP_ORDER:
        mask = groups == group
        if mask.sum() == 0:
            results[group] = {"acc": float("nan"), "macro_f1": float("nan"), "n": 0.0}
            continue
        metrics = classification_metrics(y_true[mask], y_pred[mask])
        metrics["n"] = float(mask.sum())
        results[group] = metrics
    overall = classification_metrics(y_true, y_pred)
    overall["n"] = float(y_true.shape[0])
    results["Overall"] = overall
    return results


def rows_for_method(
    method: str,
    metrics: dict[str, dict[str, float]],
    *,
    lambda_align: float | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group in (*GROUP_ORDER, "Overall"):
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
