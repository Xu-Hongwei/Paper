from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .disagreement import RELATION_STATE_GROUP_ORDER
from .metrics import classification_metrics


MODALITIES = ("t", "v", "a")


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=eps)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = _l2_normalize(left)
    right_norm = _l2_normalize(right)
    return 1.0 - np.sum(left_norm * right_norm, axis=1)


def feature_disagreement(pred: dict[str, np.ndarray]) -> np.ndarray:
    d_tv = _cosine_distance(pred["h_t"], pred["h_v"])
    d_ta = _cosine_distance(pred["h_t"], pred["h_a"])
    d_va = _cosine_distance(pred["h_v"], pred["h_a"])
    return (d_tv + d_ta + d_va) / 3.0


def feature_consistency_frame(
    d_pred: np.ndarray,
    d_feat: np.ndarray,
    relation_states: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for group in RELATION_STATE_GROUP_ORDER:
        mask = relation_states == group
        if mask.sum() < 2:
            corr = float("nan")
        else:
            corr = float(pd.Series(d_pred[mask]).corr(pd.Series(d_feat[mask]), method="spearman"))
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "avg_Dpred": _safe_mean(d_pred[mask]),
                "avg_Dfeat": _safe_mean(d_feat[mask]),
                "spearman_Dpred_Dfeat": corr,
            }
        )
    return pd.DataFrame(rows)


def class_means(pred: dict[str, np.ndarray], num_classes: int) -> dict[str, np.ndarray]:
    labels = pred["y_true"].astype(np.int64)
    means: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        features = pred[f"h_{modality}"]
        global_mean = features.mean(axis=0)
        modality_means = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
        for cls in range(num_classes):
            mask = labels == cls
            modality_means[cls] = features[mask].mean(axis=0) if mask.any() else global_mean
        means[modality] = modality_means
    return means


def residual_parts(
    pred: dict[str, np.ndarray],
    means: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = pred["y_true"].astype(np.int64)
    common_parts = []
    residual_parts_ = []
    for modality in MODALITIES:
        common = means[modality][labels]
        residual = pred[f"h_{modality}"] - common
        common_parts.append(common)
        residual_parts_.append(residual)
    common_features = np.concatenate(common_parts, axis=1)
    residual_features = np.concatenate(residual_parts_, axis=1)
    combined_features = np.concatenate([common_features, residual_features], axis=1)
    return common_features, residual_features, combined_features


def text_anchor_residual_parts(
    pred: dict[str, np.ndarray],
    means: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    common_features, _, _ = residual_parts(pred, means)
    residual_features = np.concatenate(
        [
            np.abs(pred["h_t"] - pred["h_a"]),
            np.abs(pred["h_t"] - pred["h_v"]),
        ],
        axis=1,
    )
    combined_features = np.concatenate([common_features, residual_features], axis=1)
    return residual_features, combined_features


def residual_diagnostic_frame(
    pred: dict[str, np.ndarray],
    d_pred: np.ndarray,
    d_feat: np.ndarray,
    relation_states: np.ndarray,
    means: dict[str, np.ndarray],
) -> pd.DataFrame:
    labels = pred["y_true"].astype(np.int64)
    num_classes = int(max(labels.max(initial=0), *(mean.shape[0] - 1 for mean in means.values())) + 1)
    rows = []
    residuals = {
        modality: pred[f"h_{modality}"] - means[modality][labels]
        for modality in MODALITIES
    }
    for group in RELATION_STATE_GROUP_ORDER:
        mask = relation_states == group
        residual_distances = []
        sep_values = []
        for cls in range(num_classes):
            cls_mask = mask & (labels == cls)
            if cls_mask.sum() < 2:
                continue
            stats = {}
            for modality in MODALITIES:
                values = residuals[modality][cls_mask]
                stats[modality] = {
                    "mean": values.mean(axis=0),
                    "std": values.std(axis=0),
                }
            for left, right in (("t", "v"), ("t", "a"), ("v", "a")):
                mean_dist = np.sum((stats[left]["mean"] - stats[right]["mean"]) ** 2)
                std_dist = np.sum((stats[left]["std"] - stats[right]["std"]) ** 2)
                residual_distances.append(float(mean_dist + std_dist))
        for modality in MODALITIES:
            class_centers = []
            within_values = []
            for cls in range(num_classes):
                cls_values = residuals[modality][mask & (labels == cls)]
                if cls_values.shape[0] == 0:
                    continue
                center = cls_values.mean(axis=0)
                class_centers.append(center)
                within_values.extend(np.sum((cls_values - center) ** 2, axis=1).tolist())
            if len(class_centers) < 2 or not within_values:
                continue
            between = []
            for i in range(len(class_centers)):
                for j in range(i + 1, len(class_centers)):
                    between.append(float(np.sum((class_centers[i] - class_centers[j]) ** 2)))
            sep_values.append(float(np.mean(between) / (np.mean(within_values) + 1e-8)))
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "avg_Dpred": _safe_mean(d_pred[mask]),
                "avg_Dfeat": _safe_mean(d_feat[mask]),
                "residual_dist": _safe_mean(np.asarray(residual_distances)),
                "residual_sep": _safe_mean(np.asarray(sep_values)),
            }
        )
    return pd.DataFrame(rows)


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, float]:
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
    means: dict[str, np.ndarray],
    seed: int,
) -> pd.DataFrame:
    train_common, train_residual, train_combined = residual_parts(train_pred, means)
    test_common, test_residual, test_combined = residual_parts(test_pred, means)
    train_text_anchor_residual, train_text_anchor_combined = text_anchor_residual_parts(
        train_pred,
        means,
    )
    test_text_anchor_residual, test_text_anchor_combined = text_anchor_residual_parts(
        test_pred,
        means,
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
        rows.append(
            {
                "group": group,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
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
            }
        )
    return pd.DataFrame(rows)


def selective_agreement_prototype_frame(
    train_pred: dict[str, np.ndarray],
    test_pred: dict[str, np.ndarray],
    train_groups: np.ndarray,
    train_relation_states: np.ndarray,
    test_groups: np.ndarray,
) -> pd.DataFrame:
    train_features = np.concatenate(
        [train_pred["h_t"], train_pred["h_v"], train_pred["h_a"]],
        axis=1,
    )
    test_features = np.concatenate(
        [test_pred["h_t"], test_pred["h_v"], test_pred["h_a"]],
        axis=1,
    )
    train_labels = train_pred["y_true"].astype(np.int64)
    test_labels = test_pred["y_true"].astype(np.int64)
    num_classes = train_pred["prob_f"].shape[1]
    strategies = {
        "All Low-D": train_groups == "Low-D",
        "Selective RA": train_relation_states == "RA",
    }
    rows = []
    eval_mask = test_groups == "Low-D"
    for strategy, mask in strategies.items():
        prototypes = np.zeros((num_classes, train_features.shape[1]), dtype=np.float32)
        global_mean = train_features.mean(axis=0)
        for cls in range(num_classes):
            cls_mask = mask & (train_labels == cls)
            prototypes[cls] = train_features[cls_mask].mean(axis=0) if cls_mask.any() else global_mean
        prototype_norm = _l2_normalize(prototypes)
        eval_features = test_features[eval_mask]
        if eval_features.shape[0] == 0:
            metrics = {"acc": float("nan"), "macro_f1": float("nan")}
        else:
            pred_labels = (_l2_normalize(eval_features) @ prototype_norm.T).argmax(axis=1)
            metrics = classification_metrics(test_labels[eval_mask], pred_labels)
        compactness_values = []
        for cls in range(num_classes):
            cls_features = train_features[mask & (train_labels == cls)]
            if cls_features.shape[0] == 0:
                continue
            compactness_values.extend(
                (_l2_normalize(cls_features) @ prototype_norm[cls]).tolist()
            )
        label_support_values = train_pred["prob_f"][
            np.arange(train_labels.shape[0]),
            train_labels,
        ][mask]
        rows.append(
            {
                "prototype": strategy,
                "train_n": int(mask.sum()),
                "eval_group": "Low-D",
                "prototype_purity": _safe_mean(label_support_values),
                "intra_class_compactness": _safe_mean(np.asarray(compactness_values)),
                "nearest_proto_acc": metrics["acc"],
                "nearest_proto_macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)
