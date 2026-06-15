from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics


GROUP_ORDER = ("Low-D", "Mid-D", "High-D")
HIGH_D_RELIABILITY_GROUP_ORDER = ("High-D+Low-R", "High-D+High-R")
RELATION_STATE_GROUP_ORDER = ("RA", "UA", "Mid-D", "RD", "ND")
RELATION_STATE_DESCRIPTIONS = {
    "RA": "Low-D+High-R",
    "UA": "Low-D+Low-R",
    "Mid-D": "Mid-D",
    "RD": "High-D+High-R",
    "ND": "High-D+Low-R",
}


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


def pairwise_disagreement(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "D_tv": jsd(prob_t, prob_v),
        "D_ta": jsd(prob_t, prob_a),
        "D_va": jsd(prob_v, prob_a),
    }


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, eps)


def _parse_bandwidth(bandwidth: str | float) -> float | str:
    if isinstance(bandwidth, str):
        if bandwidth == "median":
            return bandwidth
        return float(bandwidth)
    return float(bandwidth)


def _median_bandwidth(
    arrays: tuple[np.ndarray, ...],
    *,
    max_samples: int = 2048,
    seed: int = 0,
) -> float:
    features = np.concatenate([_l2_normalize(array) for array in arrays], axis=0)
    if features.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        sample = rng.choice(features.shape[0], size=max_samples, replace=False)
        features = features[sample]
    squared = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    distances = np.sqrt(squared[np.triu_indices(features.shape[0], k=1)])
    distances = distances[distances > 0]
    if distances.size == 0:
        return 1.0
    return float(np.median(distances))


def _resolve_bandwidth(
    arrays: tuple[np.ndarray, ...],
    bandwidth: str | float,
    *,
    seed: int = 0,
) -> float:
    parsed = _parse_bandwidth(bandwidth)
    if parsed == "median":
        return _median_bandwidth(arrays, seed=seed)
    if parsed <= 0:
        raise ValueError("kernel bandwidth must be positive.")
    return float(parsed)


def resolve_kernel_bandwidth(
    h_t: np.ndarray,
    h_v: np.ndarray,
    h_a: np.ndarray,
    bandwidth: str | float = "median",
    *,
    seed: int = 0,
) -> float:
    return _resolve_bandwidth((h_t, h_v, h_a), bandwidth, seed=seed)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    left = _l2_normalize(left)
    right = _l2_normalize(right)
    squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def _rbf_point_disagreement(
    left: np.ndarray,
    right: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    left = _l2_normalize(left)
    right = _l2_normalize(right)
    squared = np.sum((left - right) ** 2, axis=-1)
    return 1.0 - np.exp(-squared / (2.0 * bandwidth * bandwidth))


def _class_conditional_mmd(
    left: np.ndarray,
    right: np.ndarray,
    pred_labels: np.ndarray,
    bandwidth: float,
    *,
    max_class_samples: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    pred_labels = np.asarray(pred_labels).reshape(-1)
    if pred_labels.shape[0] != left.shape[0]:
        raise ValueError("pred_labels must have the same sample count as features.")
    result = np.zeros(left.shape[0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    for label in np.unique(pred_labels):
        indices = np.where(pred_labels == label)[0]
        if indices.size == 0:
            continue
        used = indices
        if used.size > max_class_samples:
            used = rng.choice(used, size=max_class_samples, replace=False)
        k_xx = _rbf_kernel(left[used], left[used], bandwidth).mean()
        k_yy = _rbf_kernel(right[used], right[used], bandwidth).mean()
        k_xy = _rbf_kernel(left[used], right[used], bandwidth).mean()
        result[indices] = max(0.0, float(k_xx + k_yy - 2.0 * k_xy))
    return result


def _kernel_disagreement_pair(
    left: np.ndarray,
    right: np.ndarray,
    pred_labels: np.ndarray,
    bandwidth: float,
    *,
    class_weight: float = 0.5,
    max_class_samples: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    if not 0.0 <= class_weight <= 1.0:
        raise ValueError("class_weight must be in [0, 1].")
    point = _rbf_point_disagreement(left, right, bandwidth)
    if class_weight == 0.0:
        return point
    class_mmd = _class_conditional_mmd(
        left,
        right,
        pred_labels,
        bandwidth,
        max_class_samples=max_class_samples,
        seed=seed,
    )
    return (1.0 - class_weight) * point + class_weight * class_mmd


def kernel_pairwise_disagreement(
    h_t: np.ndarray,
    h_v: np.ndarray,
    h_a: np.ndarray,
    pred_labels: np.ndarray,
    *,
    bandwidth: str | float = "median",
    class_weight: float = 0.5,
    max_class_samples: int = 1024,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    resolved = _resolve_bandwidth((h_t, h_v, h_a), bandwidth, seed=seed)
    return {
        "D_tv": _kernel_disagreement_pair(
            h_t,
            h_v,
            pred_labels,
            resolved,
            class_weight=class_weight,
            max_class_samples=max_class_samples,
            seed=seed + 1,
        ),
        "D_ta": _kernel_disagreement_pair(
            h_t,
            h_a,
            pred_labels,
            resolved,
            class_weight=class_weight,
            max_class_samples=max_class_samples,
            seed=seed + 2,
        ),
        "D_va": _kernel_disagreement_pair(
            h_v,
            h_a,
            pred_labels,
            resolved,
            class_weight=class_weight,
            max_class_samples=max_class_samples,
            seed=seed + 3,
        ),
    }


def sample_disagreement_from_pairwise(
    distances: dict[str, np.ndarray],
    *,
    pair_mode: str = "full_pair",
) -> np.ndarray:
    if pair_mode == "text_anchor":
        return (distances["D_tv"] + distances["D_ta"]) / 2.0
    if pair_mode == "full_pair":
        return (distances["D_tv"] + distances["D_ta"] + distances["D_va"]) / 3.0
    raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")


def agreement_from_distances(
    distances: dict[str, np.ndarray],
    tau_agreement: float,
) -> dict[str, np.ndarray]:
    if tau_agreement <= 0:
        raise ValueError("tau_agreement must be positive.")
    return {
        "A_tv": np.exp(-distances["D_tv"] / tau_agreement),
        "A_ta": np.exp(-distances["D_ta"] / tau_agreement),
        "A_va": np.exp(-distances["D_va"] / tau_agreement),
    }


def pairwise_agreement(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
    tau_agreement: float,
) -> dict[str, np.ndarray]:
    distances = pairwise_disagreement(prob_t, prob_v, prob_a)
    return agreement_from_distances(distances, tau_agreement)


def normalized_reliability(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    prob = np.clip(prob, eps, 1.0)
    entropy = -np.sum(prob * np.log(prob), axis=-1)
    return 1.0 - entropy / np.log(prob.shape[-1])


def label_support(prob: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    prob = np.clip(prob, eps, 1.0)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if prob.shape[0] != labels.shape[0]:
        raise ValueError(
            f"prob has {prob.shape[0]} samples but labels has {labels.shape[0]} samples."
        )
    return prob[np.arange(labels.shape[0]), labels]


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


def label_aware_reliability(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    c_text = normalized_reliability(prob_t)
    c_vision = normalized_reliability(prob_v)
    c_audio = normalized_reliability(prob_a)
    s_text = label_support(prob_t, labels)
    s_vision = label_support(prob_v, labels)
    s_audio = label_support(prob_a, labels)
    r_text = c_text * s_text
    r_vision = c_vision * s_vision
    r_audio = c_audio * s_audio
    return {
        "C_text": c_text,
        "C_vision": c_vision,
        "C_audio": c_audio,
        "S_text": s_text,
        "S_vision": s_vision,
        "S_audio": s_audio,
        "R_label_text": r_text,
        "R_label_vision": r_vision,
        "R_label_audio": r_audio,
        "R_label_sample": (r_text + r_vision + r_audio) / 3.0,
    }


def relation_gates(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
    reliability: dict[str, np.ndarray],
    tau_agreement: float,
    *,
    prefix: str = "",
    distances: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    agreements = (
        pairwise_agreement(prob_t, prob_v, prob_a, tau_agreement)
        if distances is None
        else agreement_from_distances(distances, tau_agreement)
    )
    r_text = reliability[f"{prefix}text"]
    r_vision = reliability[f"{prefix}vision"]
    r_audio = reliability[f"{prefix}audio"]
    products = {
        "tv": r_text * r_vision,
        "ta": r_text * r_audio,
        "va": r_vision * r_audio,
    }
    return {
        **agreements,
        "g_tv_agr": products["tv"] * agreements["A_tv"],
        "g_tv_comp": products["tv"] * (1.0 - agreements["A_tv"]),
        "g_tv_noise": 1.0 - products["tv"],
        "g_ta_agr": products["ta"] * agreements["A_ta"],
        "g_ta_comp": products["ta"] * (1.0 - agreements["A_ta"]),
        "g_ta_noise": 1.0 - products["ta"],
        "g_va_agr": products["va"] * agreements["A_va"],
        "g_va_comp": products["va"] * (1.0 - agreements["A_va"]),
        "g_va_noise": 1.0 - products["va"],
    }


def label_aware_relation_gates(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
    reliability: dict[str, np.ndarray],
    tau_agreement: float,
    *,
    distances: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    agreements = (
        pairwise_agreement(prob_t, prob_v, prob_a, tau_agreement)
        if distances is None
        else agreement_from_distances(distances, tau_agreement)
    )
    c_text = reliability["C_text"]
    c_vision = reliability["C_vision"]
    c_audio = reliability["C_audio"]
    s_text = reliability["S_text"]
    s_vision = reliability["S_vision"]
    s_audio = reliability["S_audio"]
    q_tv = c_text * c_vision
    q_ta = c_text * c_audio
    q_va = c_vision * c_audio
    b_tv = np.maximum(s_text, s_vision)
    b_ta = np.maximum(s_text, s_audio)
    b_va = np.maximum(s_vision, s_audio)
    g_tv_dis = q_tv * b_tv * (1.0 - agreements["A_tv"])
    g_ta_dis = q_ta * b_ta * (1.0 - agreements["A_ta"])
    g_va_dis = q_va * b_va * (1.0 - agreements["A_va"])
    return {
        **agreements,
        "B_tv_label": b_tv,
        "B_ta_label": b_ta,
        "B_va_label": b_va,
        "g_tv_agr": q_tv * s_text * s_vision * agreements["A_tv"],
        "g_tv_dis": g_tv_dis,
        "g_tv_comp": g_tv_dis,
        "g_tv_noise": 1.0 - q_tv,
        "g_ta_agr": q_ta * s_text * s_audio * agreements["A_ta"],
        "g_ta_dis": g_ta_dis,
        "g_ta_comp": g_ta_dis,
        "g_ta_noise": 1.0 - q_ta,
        "g_va_agr": q_va * s_vision * s_audio * agreements["A_va"],
        "g_va_dis": g_va_dis,
        "g_va_comp": g_va_dis,
        "g_va_noise": 1.0 - q_va,
    }


def build_label_aware_relation_frame(
    pred: dict[str, np.ndarray],
    tau_agreement: float,
    *,
    disagreement_metric: str = "prob_jsd",
    kernel_bandwidth: str | float = "median",
    kernel_pair_mode: str = "text_anchor",
    kernel_class_weight: float = 0.5,
    kernel_max_class_samples: int = 1024,
    seed: int = 0,
) -> pd.DataFrame:
    reliability = label_aware_reliability(
        pred["prob_t"],
        pred["prob_v"],
        pred["prob_a"],
        pred["y_true"],
    )
    if disagreement_metric == "prob_jsd":
        distances = pairwise_disagreement(pred["prob_t"], pred["prob_v"], pred["prob_a"])
    elif disagreement_metric == "kernel_mmd":
        distances = kernel_pairwise_disagreement(
            pred["h_t"],
            pred["h_v"],
            pred["h_a"],
            pred["y_pred"],
            bandwidth=kernel_bandwidth,
            class_weight=kernel_class_weight,
            max_class_samples=kernel_max_class_samples,
            seed=seed,
        )
    else:
        raise ValueError("disagreement_metric must be 'prob_jsd' or 'kernel_mmd'.")
    gates = label_aware_relation_gates(
        pred["prob_t"],
        pred["prob_v"],
        pred["prob_a"],
        reliability,
        tau_agreement,
        distances=distances,
    )
    return pd.DataFrame(
        {
            "index": pred["index"],
            "label_reg": pred["y_reg"],
            "label_cls": pred["y_true"],
            "prediction": pred["y_pred"],
            **reliability,
            **distances,
            **gates,
            "disagreement_metric": disagreement_metric,
            "kernel_pair_mode": kernel_pair_mode,
        }
    )


def summarize_relation_frame(frame: pd.DataFrame, split: str) -> dict[str, float | str]:
    columns = [
        "C_text",
        "C_vision",
        "C_audio",
        "S_text",
        "S_vision",
        "S_audio",
        "R_label_text",
        "R_label_vision",
        "R_label_audio",
        "R_label_sample",
        "A_tv",
        "A_ta",
        "A_va",
        "B_tv_label",
        "B_ta_label",
        "B_va_label",
        "g_tv_agr",
        "g_tv_dis",
        "g_tv_comp",
        "g_tv_noise",
        "g_ta_agr",
        "g_ta_dis",
        "g_ta_comp",
        "g_ta_noise",
        "g_va_agr",
        "g_va_dis",
        "g_va_comp",
        "g_va_noise",
    ]
    payload: dict[str, float | str] = {"split": split, "n": float(len(frame))}
    for column in columns:
        payload[f"{column}_mean"] = float(frame[column].mean())
        payload[f"{column}_std"] = float(frame[column].std(ddof=1))
    return payload


def validation_thresholds(valid_disagreement: np.ndarray) -> tuple[float, float]:
    q33, q66 = np.quantile(valid_disagreement, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def reliability_threshold(valid_reliability: np.ndarray) -> float:
    return float(np.quantile(valid_reliability, 0.5))


def assign_groups(disagreement: np.ndarray, q33: float, q66: float) -> np.ndarray:
    groups = np.empty(disagreement.shape[0], dtype=object)
    groups[disagreement <= q33] = "Low-D"
    groups[(disagreement > q33) & (disagreement <= q66)] = "Mid-D"
    groups[disagreement > q66] = "High-D"
    return groups


def assign_high_d_reliability_groups(
    groups: np.ndarray,
    r_sample: np.ndarray,
    r_threshold: float | None = None,
) -> np.ndarray:
    reliability_groups = np.full(groups.shape[0], "", dtype=object)
    high_indices = np.where(groups == "High-D")[0]
    if high_indices.shape[0] == 0:
        return reliability_groups
    if r_threshold is not None:
        reliability_groups[
            high_indices[r_sample[high_indices] < r_threshold]
        ] = "High-D+Low-R"
        reliability_groups[
            high_indices[r_sample[high_indices] >= r_threshold]
        ] = "High-D+High-R"
        return reliability_groups
    sorted_high = high_indices[np.argsort(r_sample[high_indices])]
    split = max(1, sorted_high.shape[0] // 2)
    reliability_groups[sorted_high[:split]] = "High-D+Low-R"
    reliability_groups[sorted_high[split:]] = "High-D+High-R"
    return reliability_groups


def assign_relation_state_groups(
    groups: np.ndarray,
    r_sample: np.ndarray,
    r_threshold: float,
) -> np.ndarray:
    relation_states = np.full(groups.shape[0], "Mid-D", dtype=object)
    low_d = groups == "Low-D"
    high_d = groups == "High-D"
    high_r = r_sample >= r_threshold
    relation_states[low_d & high_r] = "RA"
    relation_states[low_d & ~high_r] = "UA"
    relation_states[high_d & high_r] = "RD"
    relation_states[high_d & ~high_r] = "ND"
    return relation_states


def build_group_frame(
    test_pred: dict[str, np.ndarray],
    disagreement: np.ndarray,
    groups: np.ndarray,
    reliability: dict[str, np.ndarray] | None = None,
    reliability_groups: np.ndarray | None = None,
    relation_state_groups: np.ndarray | None = None,
    relations: dict[str, np.ndarray] | None = None,
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
    if relations is not None:
        payload.update(relations)
    if reliability_groups is not None:
        payload["high_d_reliability_group"] = reliability_groups
    if relation_state_groups is not None:
        payload["relation_state"] = relation_state_groups
        payload["relation_state_desc"] = [
            RELATION_STATE_DESCRIPTIONS.get(str(group), str(group))
            for group in relation_state_groups
        ]
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
