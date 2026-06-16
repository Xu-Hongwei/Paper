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
    """计算 KL 散度 D_KL(p || q) = sum(p * log(p/q))。

    Args:
        p: 第一个概率分布，shape (..., K)。
        q: 第二个概率分布，shape 与 p 相同。
        eps: 数值稳定性常数，将概率裁剪到 [eps, 1-eps] 以避免 log(0)。

    Returns:
        KL 散度值，shape 为 p 去掉最后一维。
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """计算 Jensen-Shannon 散度 JSD(p || q) = 0.5*KL(p||m) + 0.5*KL(q||m)，其中 m = (p+q)/2。

    JSD 是对称且有界的散度度量，值域为 [0, log(2)]，常用于衡量两个概率分布之间的差异。

    Args:
        p: 第一个概率分布，shape (..., K)。
        q: 第二个概率分布，shape 与 p 相同。
        eps: 传递给 KL 散度的数值稳定性常数。

    Returns:
        JSD 值，shape 与 p、q 的广播结果一致。
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m, eps=eps) + 0.5 * _kl_div(q, m, eps=eps)


def pairwise_disagreement(

    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
) -> dict[str, np.ndarray]:
    """计算三模态两两之间的 JSD 分歧。

    对文本、视觉、音频三个模态的预测概率，计算所有模态对之间的 JSD。

    Args:
        prob_t: 文本模态预测概率，shape (N, K)。
        prob_v: 视觉模态预测概率，shape (N, K)。
        prob_a: 音频模态预测概率，shape (N, K)。

    Returns:
        dict，包含 D_tv、D_ta、D_va 三个键，每个值为 shape (N,) 的 JSD 数组。
    """
    return {
        "D_tv": jsd(prob_t, prob_v),
        "D_ta": jsd(prob_t, prob_a),
        "D_va": jsd(prob_v, prob_a),
    }


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """沿最后一维对特征向量做 L2 归一化。

    Args:
        features: 输入特征，shape (..., D)。
        eps: 防止除零的小常数。

    Returns:
        L2 归一化后的特征，shape 与输入相同。
    """
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, eps)


def _parse_bandwidth(bandwidth: str | float) -> float | str:
    """解析带宽参数，若为字符串 "median" 则原样返回，否则转为 float。

    Args:
        bandwidth: 带宽值或字符串 "median"。

    Returns:
        浮点带宽值或字符串 "median"。
    """
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
    """用所有特征拼接后两两 L2 距离的中位数作为 RBF 核带宽。

    先对每组特征做 L2 归一化并拼接，然后计算上三角成对距离，取中位数。

    Args:
        arrays: 待拼接的特征数组元组，每个 shape 为 (N_i, D)。
        max_samples: 计算中位数时的最大采样数，用于控制计算量。
        seed: 随机采样的种子。

    Returns:
        中位数距离作为带宽值。若无有效距离则返回 1.0。
    """
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
    """解析并解析带宽：若为 "median" 则自动计算中位数带宽，否则直接使用数值。

    Args:
        arrays: 特征数组元组，用于中位数带宽计算。
        bandwidth: 带宽值或 "median"。
        seed: 中位数带宽计算时的随机种子。

    Returns:
        解析后的浮点带宽值。

    Raises:
        ValueError: 带宽非正数时抛出。
    """
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
    """解析文本、视觉、音频三模态特征的 RBF 核带宽。

    Args:
        h_t: 文本模态特征，shape (N, D_t)。
        h_v: 视觉模态特征，shape (N, D_v)。
        h_a: 音频模态特征，shape (N, D_a)。
        bandwidth: 带宽值或 "median"（默认自动计算中位数带宽）。
        seed: 随机种子。

    Returns:
        解析后的浮点带宽值。
    """
    return _resolve_bandwidth((h_t, h_v, h_a), bandwidth, seed=seed)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    """计算两组特征之间的 RBF (高斯) 核矩阵。

    先对特征做 L2 归一化，再计算 k(x_i, y_j) = exp(-||x_i - y_j||^2 / (2*σ^2))。

    Args:
        left: 第一组特征，shape (N, D)。
        right: 第二组特征，shape (M, D)。
        bandwidth: RBF 核带宽 σ。

    Returns:
        核矩阵，shape (N, M)。
    """
    left = _l2_normalize(left)
    right = _l2_normalize(right)
    squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def _rbf_point_disagreement(
    left: np.ndarray,
    right: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """计算逐点 RBF 分歧: d(x_i, y_i) = 1 - exp(-||x_i - y_i||^2 / (2*σ^2))。

    值域 [0, 1]，值越大表示两个对应样本的特征差异越大。

    Args:
        left: 第一组特征，shape (N, D)。
        right: 第二组特征，shape (N, D)。
        bandwidth: RBF 核带宽 σ。

    Returns:
        逐点分歧值，shape (N,)。
    """
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
    """计算类别条件的 MMD（最大均值差异）。

    对每个预测类别，分别计算左右两组特征在该类别上的 MMD^2，然后将结果赋回
    属于该类别的所有样本。大类会随机降采样以控制计算量。

    Args:
        left: 第一组特征，shape (N, D)。
        right: 第二组特征，shape (N, D)。
        pred_labels: 预测标签，shape (N,)。
        bandwidth: RBF 核带宽。
        max_class_samples: 每个类别的最大采样数。
        seed: 随机种子。

    Returns:
        每个样本的类别条件 MMD 值，shape (N,)。

    Raises:
        ValueError: pred_labels 与特征样本数不匹配时抛出。
    """
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
    """计算一对模态之间的核分歧：逐点分歧与类别条件 MMD 的加权组合。

    disagreement = (1 - class_weight) * point_disagreement + class_weight * class_mmd。

    Args:
        left: 第一个模态特征，shape (N, D)。
        right: 第二个模态特征，shape (N, D)。
        pred_labels: 预测标签，shape (N,)。
        bandwidth: RBF 核带宽。
        class_weight: 类别条件 MMD 的权重，范围 [0, 1]。
        max_class_samples: 每个类别的最大采样数。
        seed: 随机种子。

    Returns:
        加权分歧值，shape (N,)。

    Raises:
        ValueError: class_weight 不在 [0, 1] 范围内时抛出。
    """
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
    """计算三模态两两之间的核分歧。

    对文本、视觉、音频特征，使用逐点 RBF 分歧与类别条件 MMD 的加权组合，
    计算所有模态对之间的分歧。

    Args:
        h_t: 文本模态特征，shape (N, D_t)。
        h_v: 视觉模态特征，shape (N, D_v)。
        h_a: 音频模态特征，shape (N, D_a)。
        pred_labels: 预测标签，shape (N,)。
        bandwidth: RBF 核带宽，默认 "median" 自动计算。
        class_weight: 类别条件 MMD 权重。
        max_class_samples: 每个类别的最大采样数。
        seed: 随机种子。

    Returns:
        dict，包含 D_tv、D_ta、D_va 三个键，每个值为 shape (N,) 的分歧数组。
    """
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
    """从成对分歧聚合成样本级分歧。

    - text_anchor: 以文本为锚点，平均 D_tv 和 D_ta。
    - full_pair: 平均全部三对 (D_tv, D_ta, D_va)。

    Args:
        distances: 包含 D_tv、D_ta、D_va 的字典，每个值 shape (N,)。
        pair_mode: 聚合模式，"text_anchor" 或 "full_pair"。

    Returns:
        样本级分歧值，shape (N,)。

    Raises:
        ValueError: pair_mode 不合法时抛出。
    """
    if pair_mode == "text_anchor":
        return (distances["D_tv"] + distances["D_ta"]) / 2.0
    if pair_mode == "full_pair":
        return (distances["D_tv"] + distances["D_ta"] + distances["D_va"]) / 3.0
    raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")


def agreement_from_distances(
    distances: dict[str, np.ndarray],
    tau_agreement: float,
) -> dict[str, np.ndarray]:
    """将分歧距离转换为一致性：A = exp(-D / τ)。

    τ 控制转换的温度：τ 越大，对分歧越宽容。

    Args:
        distances: 包含 D_tv、D_ta、D_va 的字典，每个值 shape (N,)。
        tau_agreement: 温度参数，必须为正。

    Returns:
        dict，包含 A_tv、A_ta、A_va，每个值 shape (N,)，值域 (0, 1]。

    Raises:
        ValueError: tau_agreement <= 0 时抛出。
    """
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
    """从三模态预测概率直接计算成对一致性。

    先计算 JSD 分歧，再通过 exp(-D/τ) 转换为一致性。

    Args:
        prob_t: 文本模态预测概率，shape (N, K)。
        prob_v: 视觉模态预测概率，shape (N, K)。
        prob_a: 音频模态预测概率，shape (N, K)。
        tau_agreement: 温度参数，必须为正。

    Returns:
        dict，包含 A_tv、A_ta、A_va。
    """
    distances = pairwise_disagreement(prob_t, prob_v, prob_a)
    return agreement_from_distances(distances, tau_agreement)


def normalized_reliability(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """计算归一化可靠性：R = 1 - H(prob) / log(K)，值域 [0, 1]。

    可靠性越高表示预测越确定（低熵）。当熵为 0 时可靠性为 1，当熵最大
    （均匀分布）时可靠性为 0。

    Args:
        prob: 预测概率分布，shape (..., K)。
        eps: 数值稳定性常数。

    Returns:
        归一化可靠性，shape 为 prob 去掉最后一维。
    """
    prob = np.clip(prob, eps, 1.0)
    entropy = -np.sum(prob * np.log(prob), axis=-1)
    return 1.0 - entropy / np.log(prob.shape[-1])


def sample_reliability(
    prob_t: np.ndarray,
    prob_v: np.ndarray,
    prob_a: np.ndarray,
    *,
    pair_mode: str = "text_anchor",
) -> dict[str, np.ndarray]:
    """计算三模态的单模态可靠性、成对可靠性及样本级可靠性。

    成对可靠性 = 两个模态归一化可靠性的乘积。
    样本级可靠性 = 成对可靠性的平均（按 pair_mode 选择参与平均的对）。

    Args:
        prob_t: 文本模态预测概率，shape (N, K)。
        prob_v: 视觉模态预测概率，shape (N, K)。
        prob_a: 音频模态预测概率，shape (N, K)。
        pair_mode: "text_anchor" 仅平均 R_tv 和 R_ta；"full_pair" 平均全部三对。

    Returns:
        dict，包含 R_text, R_vision, R_audio, R_tv, R_ta, R_va, R_sample。
    """
    r_text = normalized_reliability(prob_t)
    r_vision = normalized_reliability(prob_v)
    r_audio = normalized_reliability(prob_a)
    r_tv = r_text * r_vision
    r_ta = r_text * r_audio
    r_va = r_vision * r_audio
    if pair_mode == "text_anchor":
        r_sample = (r_ta + r_tv) / 2.0
    elif pair_mode == "full_pair":
        r_sample = (r_ta + r_tv + r_va) / 3.0
    else:
        raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")
    return {
        "R_text": r_text,
        "R_vision": r_vision,
        "R_audio": r_audio,
        "R_tv": r_tv,
        "R_ta": r_ta,
        "R_va": r_va,
        "R_sample": r_sample,
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
    """计算每个模态对的关系门控值（一致 / 互补 / 噪声）。

    对每对模态 (tv, ta, va)：
    - g_*_agr = R_prod * A（两者都可靠且一致）
    - g_*_comp = R_prod * (1 - A)（两者都可靠但不一致，即互补）
    - g_*_noise = 1 - R_prod（至少一个不可靠，即噪声）

    Args:
        prob_t: 文本模态预测概率，shape (N, K)。
        prob_v: 视觉模态预测概率，shape (N, K)。
        prob_a: 音频模态预测概率，shape (N, K)。
        reliability: 包含各模态可靠性值的字典（键如 "text", "vision", "audio"）。
        tau_agreement: 一致性温度参数。
        prefix: reliability 字典键的前缀。
        distances: 可选的预计算分岐字典，避免重复计算。

    Returns:
        dict，包含 A_tv/A_ta/A_va 及所有 g_*_{agr,comp,noise} 门控值。
    """
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


def validation_thresholds(valid_disagreement: np.ndarray) -> tuple[float, float]:
    """在验证集上计算分歧的三分位数阈值（1/3 和 2/3 分位点）。

    用于将样本划分为 Low-D / Mid-D / High-D 三组。

    Args:
        valid_disagreement: 验证集样本分歧值，shape (N,)。

    Returns:
        (q33, q66) 两个分位阈值。
    """
    q33, q66 = np.quantile(valid_disagreement, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def reliability_threshold(valid_reliability: np.ndarray) -> float:
    """在验证集上计算可靠性中位数阈值。

    Args:
        valid_reliability: 验证集可靠性值，shape (N,)。

    Returns:
        中位数可靠性阈值。
    """
    return float(np.quantile(valid_reliability, 0.5))


def within_group_reliability_thresholds(
    groups: np.ndarray,
    r_sample: np.ndarray,
) -> dict[str, float]:
    """分别计算 Low-D 和 High-D 组内的可靠性中位数阈值。

    Args:
        groups: 分歧分组标签，shape (N,)。
        r_sample: 样本可靠性值，shape (N,)。

    Returns:
        dict，包含 "Low-D"、"High-D" 的组内阈值及 "global" 全局阈值。
    """
    thresholds: dict[str, float] = {}
    for group in ("Low-D", "High-D"):
        values = r_sample[groups == group]
        thresholds[group] = (
            float(np.quantile(values, 0.5))
            if values.shape[0] > 0
            else reliability_threshold(r_sample)
        )
    thresholds["global"] = reliability_threshold(r_sample)
    return thresholds


def assign_groups(disagreement: np.ndarray, q33: float, q66: float) -> np.ndarray:
    """按分歧值将样本划分为 Low-D / Mid-D / High-D 三组。

    Args:
        disagreement: 样本分歧值，shape (N,)。
        q33: 下三分位阈值。
        q66: 上三分位阈值。

    Returns:
        分组标签数组，shape (N,)，取值为 "Low-D" / "Mid-D" / "High-D"。
    """
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
    """在 High-D 组内按可靠性划分为 High-D+Low-R 和 High-D+High-R。

    若提供 r_threshold 则按阈值划分；否则按可靠性排序后中位切分。

    Args:
        groups: 分歧分组标签，shape (N,)。
        r_sample: 样本可靠性值，shape (N,)。
        r_threshold: 可靠性阈值，若为 None 则中位切分。

    Returns:
        可靠性子组标签，shape (N,)，非 High-D 样本为空字符串。
    """
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
    """按分歧和可靠性划分关系状态：RA / UA / Mid-D / RD / ND。

    划分规则：
    - Low-D + High-R → RA (Reliable Agreement)
    - Low-D + Low-R  → UA (Unreliable Agreement)
    - Mid-D           → Mid-D
    - High-D + High-R → RD (Reliable Disagreement)
    - High-D + Low-R  → ND (Noisy Disagreement)

    Args:
        groups: 分歧分组标签，shape (N,)。
        r_sample: 样本可靠性值，shape (N,)。
        r_threshold: 全局可靠性阈值。

    Returns:
        关系状态标签数组，shape (N,)。
    """
    relation_states = np.full(groups.shape[0], "Mid-D", dtype=object)
    low_d = groups == "Low-D"
    high_d = groups == "High-D"
    high_r = r_sample >= r_threshold
    relation_states[low_d & high_r] = "RA"
    relation_states[low_d & ~high_r] = "UA"
    relation_states[high_d & high_r] = "RD"
    relation_states[high_d & ~high_r] = "ND"
    return relation_states


def assign_relation_state_groups_balanced(
    groups: np.ndarray,
    r_sample: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """按分歧和分组内可靠性阈值划分关系状态（组内均衡版）。

    与 assign_relation_state_groups 不同的是，Low-D 和 High-D 分别使用
    各自组内的可靠性中位数作为阈值，避免组间样本量不均导致的偏差。

    Args:
        groups: 分歧分组标签，shape (N,)。
        r_sample: 样本可靠性值，shape (N,)。
        thresholds: 包含 "Low-D"、"High-D"、"global" 阈值的字典。

    Returns:
        关系状态标签数组，shape (N,)。
    """
    relation_states = np.full(groups.shape[0], "Mid-D", dtype=object)
    low_d = groups == "Low-D"
    high_d = groups == "High-D"
    low_threshold = thresholds.get("Low-D", thresholds.get("global", 0.0))
    high_threshold = thresholds.get("High-D", thresholds.get("global", 0.0))
    relation_states[low_d & (r_sample >= low_threshold)] = "RA"
    relation_states[low_d & (r_sample < low_threshold)] = "UA"
    relation_states[high_d & (r_sample >= high_threshold)] = "RD"
    relation_states[high_d & (r_sample < high_threshold)] = "ND"
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
    """构建包含预测、分歧、分组等信息的汇总 DataFrame。

    Args:
        test_pred: 测试集预测结果，需含 "index", "y_reg", "y_true", "y_pred"。
        disagreement: 样本级分歧值，shape (N,)。
        groups: 分歧分组标签，shape (N,)。
        reliability: 可选的可靠性字典。
        reliability_groups: 可选的 High-D 可靠性子组标签。
        relation_state_groups: 可选的关系状态标签。
        relations: 可选的关系门控值字典。

    Returns:
        包含所有字段的 DataFrame，每行一个样本。
    """
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
    """按分组计算分类指标（acc, macro_f1）。

    Args:
        y_true: 真实标签，shape (N,)。
        y_pred: 预测标签，shape (N,)。
        groups: 分组标签，shape (N,)。
        group_order: 分组的遍历顺序。
        include_overall: 是否包含 Overall 汇总行。

    Returns:
        嵌套字典，外层键为组名，内层为 {"acc", "macro_f1", "n"}。
        空组返回 NaN。
    """
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
    """将 grouped_metrics 的输出展平为表格行列表，便于构建结果汇总表。

    Args:
        method: 方法名称。
        metrics: grouped_metrics 的输出字典。
        lambda_align: 可选的对齐超参数 λ，用于记录实验配置。
        group_order: 分组的遍历顺序。
        include_overall: 是否包含 Overall 行。

    Returns:
        行字典列表，每行含 method, group, n, acc, macro_f1（及可选的 lambda_align）。
    """
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
