from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .disagreement import RELATION_STATE_GROUP_ORDER
from .metrics import classification_metrics


MODALITIES = ("t", "v", "a")
RESIDUAL_PAIR_SCOPES = ("text_anchor", "full_pair")
RESIDUAL_MODES = ("abs", "signed", "prod", "all")


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """沿 axis=1 对特征做 L2 归一化。

    Args:
        features: 输入特征，shape (N, D)。
        eps: 防止除零的小常数。

    Returns:
        L2 归一化后的特征。
    """
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, eps)


def _normalized_hidden(pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """从预测字典中提取三模态隐层特征并做 L2 归一化。

    Args:
        pred: 预测结果字典，需含 "h_t", "h_v", "h_a"。

    Returns:
        dict，键为 "t"/"v"/"a"，值为归一化后的隐层特征。
    """
    return {modality: _l2_normalize(pred[f"h_{modality}"]) for modality in MODALITIES}


def common_parts(pred: dict[str, np.ndarray]) -> np.ndarray:
    """拼接三模态 L2 归一化隐层特征作为 "共性部分"。

    Args:
        pred: 预测结果字典，需含 "h_t", "h_v", "h_a"。

    Returns:
        三模态拼接特征，shape (N, D_t+D_v+D_a)。
    """
    hidden = _normalized_hidden(pred)
    return np.concatenate([hidden[modality] for modality in MODALITIES], axis=1)


def residual_parts(
    pred: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算共性特征、残差特征及两者拼接。

    残差 = |h_t - h_a|, |h_t - h_v|, |h_a - h_v| 的拼接（绝对值差）。

    Args:
        pred: 预测结果字典。

    Returns:
        (common_features, residual_features, combined_features) 三元组。
    """
    hidden = _normalized_hidden(pred)
    common_features = np.concatenate([hidden[modality] for modality in MODALITIES], axis=1)
    residual_features = np.concatenate(
        [
            np.abs(hidden["t"] - hidden["a"]),
            np.abs(hidden["t"] - hidden["v"]),
            np.abs(hidden["a"] - hidden["v"]),
        ],
        axis=1,
    )
    combined_features = np.concatenate([common_features, residual_features], axis=1)
    return common_features, residual_features, combined_features


def text_anchor_residual_parts(
    pred: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """计算以文本为锚点的残差特征及拼接特征。

    仅使用 |h_t - h_a| 和 |h_t - h_v| 作为残差。

    Args:
        pred: 预测结果字典。

    Returns:
        (residual_features, combined_features) 二元组。
    """
    hidden = _normalized_hidden(pred)
    common_features = np.concatenate([hidden[modality] for modality in MODALITIES], axis=1)
    residual_features = np.concatenate(
        [
            np.abs(hidden["t"] - hidden["a"]),
            np.abs(hidden["t"] - hidden["v"]),
        ],
        axis=1,
    )
    combined_features = np.concatenate([common_features, residual_features], axis=1)
    return residual_features, combined_features


def _pair_residual(left: np.ndarray, right: np.ndarray, mode: str) -> np.ndarray:
    """计算一对模态之间的残差特征。

    Args:
        left: 第一个模态特征，shape (N, D)。
        right: 第二个模态特征，shape (N, D)。
        mode: 残差模式——
            "abs": |left - right|
            "signed": left - right
            "prod": left * right（逐元素乘积）
            "all": 以上三种拼接

    Returns:
        残差特征矩阵。

    Raises:
        ValueError: mode 不合法时抛出。
    """
    signed = left - right
    if mode == "abs":
        return np.abs(signed)
    if mode == "signed":
        return signed
    if mode == "prod":
        return left * right
    if mode == "all":
        return np.concatenate([np.abs(signed), signed, left * right], axis=1)
    raise ValueError("residual mode must be one of: abs, signed, prod, all.")


def residual_features_by_mode(
    pred: dict[str, np.ndarray],
    *,
    residual_mode: str,
    residual_scope: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按指定残差模式和范围计算共性/残差/组合特征。

    Args:
        pred: 预测结果字典。
        residual_mode: 残差计算模式（"abs"/"signed"/"prod"/"all"）。
        residual_scope: 残差模态对范围——
            "text_anchor" 仅 (t,a) 和 (t,v)；
            "full_pair" 额外包含 (a,v)。

    Returns:
        (common_features, residual_features, combined_features) 三元组。

    Raises:
        ValueError: residual_scope 不合法时抛出。
    """
    hidden = _normalized_hidden(pred)
    common_features = np.concatenate([hidden[modality] for modality in MODALITIES], axis=1)
    pairs = [
        ("t", "a"),
        ("t", "v"),
    ]
    if residual_scope == "full_pair":
        pairs.append(("a", "v"))
    elif residual_scope != "text_anchor":
        raise ValueError("residual_scope must be 'text_anchor' or 'full_pair'.")
    residual_features = np.concatenate(
        [
            _pair_residual(hidden[left], hidden[right], residual_mode)
            for left, right in pairs
        ],
        axis=1,
    )
    combined_features = np.concatenate([common_features, residual_features], axis=1)
    return common_features, residual_features, combined_features


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """训练线性探针（StandardScaler + 平衡 LogisticRegression）并评估。

    若训练/测试样本不足或类别数不足，返回 NaN。

    Args:
        x_train: 训练特征，shape (N_train, D)。
        y_train: 训练标签，shape (N_train,)。
        x_test: 测试特征，shape (N_test, D)。
        y_test: 测试标签，shape (N_test,)。
        seed: 随机种子。

    Returns:
        dict，包含 "acc" 和 "macro_f1"。
    """
    if x_train.shape[0] < 4 or x_test.shape[0] == 0:
        return {"acc": float("nan"), "macro_f1": float("nan")}
    if np.unique(y_train).shape[0] < 2 or np.unique(y_test).shape[0] < 2:
        return {"acc": float("nan"), "macro_f1": float("nan")}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return classification_metrics(y_test, y_pred)


def residual_probe_frame(
    train_pred: dict[str, np.ndarray],
    test_pred: dict[str, np.ndarray],
    train_relation_states: np.ndarray,
    test_relation_states: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """按关系状态分组，用线性探针评估共性/残差特征的分类能力。

    对每个关系状态组，分别训练：
    - 共性探针（三模态拼接）
    - 残差探针（模态间绝对差）
    - 组合探针（共性 + 残差）
    并包含标签打乱和特征打乱两种消融对照。

    Args:
        train_pred: 训练集预测结果字典。
        test_pred: 测试集预测结果字典。
        train_relation_states: 训练集关系状态标签。
        test_relation_states: 测试集关系状态标签。
        seed: 随机种子。

    Returns:
        DataFrame，每行一个关系状态组，含各探针的 acc 和 macro_f1。
    """
    train_common, train_residual, train_combined = residual_parts(train_pred)
    test_common, test_residual, test_combined = residual_parts(test_pred)
    train_text_anchor_residual, train_text_anchor_combined = text_anchor_residual_parts(
        train_pred,
    )
    test_text_anchor_residual, test_text_anchor_combined = text_anchor_residual_parts(
        test_pred,
    )
    y_train = train_pred["y_true"].astype(np.int64)
    y_test = test_pred["y_true"].astype(np.int64)
    rng = np.random.default_rng(seed)
    rows = []
    for group in RELATION_STATE_GROUP_ORDER:
        train_mask = train_relation_states == group
        test_mask = test_relation_states == group
        common_metrics = _fit_probe(
            train_common[train_mask],
            y_train[train_mask],
            test_common[test_mask],
            y_test[test_mask],
            seed,
        )
        residual_metrics = _fit_probe(
            train_residual[train_mask],
            y_train[train_mask],
            test_residual[test_mask],
            y_test[test_mask],
            seed,
        )
        combined_metrics = _fit_probe(
            train_combined[train_mask],
            y_train[train_mask],
            test_combined[test_mask],
            y_test[test_mask],
            seed,
        )
        text_anchor_residual_metrics = _fit_probe(
            train_text_anchor_residual[train_mask],
            y_train[train_mask],
            test_text_anchor_residual[test_mask],
            y_test[test_mask],
            seed,
        )
        text_anchor_combined_metrics = _fit_probe(
            train_text_anchor_combined[train_mask],
            y_train[train_mask],
            test_text_anchor_combined[test_mask],
            y_test[test_mask],
            seed,
        )
        shuffled_labels = y_train[train_mask].copy()
        rng.shuffle(shuffled_labels)
        shuffled_metrics = _fit_probe(
            train_residual[train_mask],
            shuffled_labels,
            test_residual[test_mask],
            y_test[test_mask],
            seed,
        )
        text_anchor_shuffled_metrics = _fit_probe(
            train_text_anchor_residual[train_mask],
            shuffled_labels,
            test_text_anchor_residual[test_mask],
            y_test[test_mask],
            seed,
        )
        train_residual_shuffle = train_residual[train_mask].copy()
        test_residual_shuffle = test_residual[test_mask].copy()
        rng.shuffle(train_residual_shuffle, axis=0)
        rng.shuffle(test_residual_shuffle, axis=0)
        shuffled_combined_metrics = _fit_probe(
            np.concatenate([train_common[train_mask], train_residual_shuffle], axis=1),
            y_train[train_mask],
            np.concatenate([test_common[test_mask], test_residual_shuffle], axis=1),
            y_test[test_mask],
            seed,
        )
        train_text_anchor_residual_shuffle = train_text_anchor_residual[train_mask].copy()
        test_text_anchor_residual_shuffle = test_text_anchor_residual[test_mask].copy()
        rng.shuffle(train_text_anchor_residual_shuffle, axis=0)
        rng.shuffle(test_text_anchor_residual_shuffle, axis=0)
        text_anchor_shuffled_combined_metrics = _fit_probe(
            np.concatenate(
                [train_common[train_mask], train_text_anchor_residual_shuffle],
                axis=1,
            ),
            y_train[train_mask],
            np.concatenate(
                [test_common[test_mask], test_text_anchor_residual_shuffle],
                axis=1,
            ),
            y_test[test_mask],
            seed,
        )
        rows.append(
            {
                "group": group,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "probe_feature_source": "label_free_l2_hidden",
                "common_only_acc": common_metrics["acc"],
                "common_only_macro_f1": common_metrics["macro_f1"],
                "residual_only_acc": residual_metrics["acc"],
                "residual_only_macro_f1": residual_metrics["macro_f1"],
                "common_residual_acc": combined_metrics["acc"],
                "common_residual_macro_f1": combined_metrics["macro_f1"],
                "residual_gain_macro_f1": combined_metrics["macro_f1"]
                - common_metrics["macro_f1"],
                "shuffled_residual_only_macro_f1": shuffled_metrics["macro_f1"],
                "text_anchor_residual_only_macro_f1": text_anchor_residual_metrics[
                    "macro_f1"
                ],
                "text_anchor_common_residual_macro_f1": text_anchor_combined_metrics[
                    "macro_f1"
                ],
                "text_anchor_residual_gain_macro_f1": text_anchor_combined_metrics[
                    "macro_f1"
                ]
                - common_metrics["macro_f1"],
                "text_anchor_shuffled_residual_macro_f1": text_anchor_shuffled_metrics[
                    "macro_f1"
                ],
                "common_shuffled_residual_macro_f1": shuffled_combined_metrics[
                    "macro_f1"
                ],
                "residual_gain_vs_feature_shuffle_macro_f1": combined_metrics[
                    "macro_f1"
                ]
                - shuffled_combined_metrics["macro_f1"],
                "text_anchor_common_shuffled_residual_macro_f1": (
                    text_anchor_shuffled_combined_metrics["macro_f1"]
                ),
                "text_anchor_residual_gain_vs_feature_shuffle_macro_f1": (
                    text_anchor_combined_metrics["macro_f1"]
                    - text_anchor_shuffled_combined_metrics["macro_f1"]
                ),
            }
        )
    return pd.DataFrame(rows)


def residual_probe_by_mode_frame(
    train_pred: dict[str, np.ndarray],
    test_pred: dict[str, np.ndarray],
    train_relation_states: np.ndarray,
    test_relation_states: np.ndarray,
    seed: int,
    *,
    residual_modes: list[str] | tuple[str, ...] = RESIDUAL_MODES,
    residual_scopes: list[str] | tuple[str, ...] = RESIDUAL_PAIR_SCOPES,
) -> pd.DataFrame:
    """按残差模式和范围组合，系统地评估线性探针性能。

    在 residual_scopes × residual_modes × relation_states 的笛卡尔积上
    训练探针，并包含标签打乱和样本打乱两种消融。

    Args:
        train_pred: 训练集预测结果字典。
        test_pred: 测试集预测结果字典。
        train_relation_states: 训练集关系状态标签。
        test_relation_states: 测试集关系状态标签。
        seed: 随机种子。
        residual_modes: 残差模式列表（默认 ["abs", "signed", "prod", "all"]）。
        residual_scopes: 残差范围列表（默认 ["text_anchor", "full_pair"]）。

    Returns:
        DataFrame，每行一个 (scope, mode, group) 组合的评估结果。
    """
    y_train = train_pred["y_true"].astype(np.int64)
    y_test = test_pred["y_true"].astype(np.int64)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for residual_scope in residual_scopes:
        for residual_mode in residual_modes:
            train_common, train_residual, train_combined = residual_features_by_mode(
                train_pred,
                residual_mode=residual_mode,
                residual_scope=residual_scope,
            )
            test_common, test_residual, test_combined = residual_features_by_mode(
                test_pred,
                residual_mode=residual_mode,
                residual_scope=residual_scope,
            )
            for group in RELATION_STATE_GROUP_ORDER:
                train_mask = train_relation_states == group
                test_mask = test_relation_states == group
                common_metrics = _fit_probe(
                    train_common[train_mask],
                    y_train[train_mask],
                    test_common[test_mask],
                    y_test[test_mask],
                    seed,
                )
                residual_metrics = _fit_probe(
                    train_residual[train_mask],
                    y_train[train_mask],
                    test_residual[test_mask],
                    y_test[test_mask],
                    seed,
                )
                combined_metrics = _fit_probe(
                    train_combined[train_mask],
                    y_train[train_mask],
                    test_combined[test_mask],
                    y_test[test_mask],
                    seed,
                )
                shuffled_labels = y_train[train_mask].copy()
                rng.shuffle(shuffled_labels)
                label_shuffled_metrics = _fit_probe(
                    train_residual[train_mask],
                    shuffled_labels,
                    test_residual[test_mask],
                    y_test[test_mask],
                    seed,
                )
                train_residual_shuffle = train_residual[train_mask].copy()
                test_residual_shuffle = test_residual[test_mask].copy()
                rng.shuffle(train_residual_shuffle, axis=0)
                rng.shuffle(test_residual_shuffle, axis=0)
                sample_shuffled_metrics = _fit_probe(
                    np.concatenate([train_common[train_mask], train_residual_shuffle], axis=1),
                    y_train[train_mask],
                    np.concatenate([test_common[test_mask], test_residual_shuffle], axis=1),
                    y_test[test_mask],
                    seed,
                )
                rows.append(
                    {
                        "group": group,
                        "residual_scope": residual_scope,
                        "residual_mode": residual_mode,
                        "train_n": int(train_mask.sum()),
                        "test_n": int(test_mask.sum()),
                        "probe_feature_source": "label_free_l2_hidden",
                        "common_only_acc": common_metrics["acc"],
                        "common_only_macro_f1": common_metrics["macro_f1"],
                        "matched_residual_only_acc": residual_metrics["acc"],
                        "matched_residual_only_macro_f1": residual_metrics["macro_f1"],
                        "matched_common_residual_acc": combined_metrics["acc"],
                        "matched_common_residual_macro_f1": combined_metrics["macro_f1"],
                        "matched_residual_gain_macro_f1": (
                            combined_metrics["macro_f1"] - common_metrics["macro_f1"]
                        ),
                        "label_shuffled_residual_only_macro_f1": label_shuffled_metrics[
                            "macro_f1"
                        ],
                        "sample_shuffled_common_residual_macro_f1": sample_shuffled_metrics[
                            "macro_f1"
                        ],
                        "gain_vs_sample_shuffle_macro_f1": (
                            combined_metrics["macro_f1"]
                            - sample_shuffled_metrics["macro_f1"]
                        ),
                    }
                )
    return pd.DataFrame(rows)
