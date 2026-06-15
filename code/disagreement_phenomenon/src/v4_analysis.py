from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .disagreement import RELATION_STATE_GROUP_ORDER
from .metrics import classification_metrics


MODALITIES = ("t", "v", "a")


def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, eps)


def _normalized_hidden(pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {modality: _l2_normalize(pred[f"h_{modality}"]) for modality in MODALITIES}


def common_parts(pred: dict[str, np.ndarray]) -> np.ndarray:
    hidden = _normalized_hidden(pred)
    return np.concatenate([hidden[modality] for modality in MODALITIES], axis=1)


def residual_parts(
    pred: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    seed: int,
) -> pd.DataFrame:
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
