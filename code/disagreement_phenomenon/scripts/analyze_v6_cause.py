from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultimodalSplitDataset, infer_input_dims, load_npz_splits  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.model import MultimodalClassifier  # noqa: E402
from src.train import predict  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


DISAGREEMENT_GROUP_ORDER = ("Overall", "Low-D", "Mid-D", "High-D")
RELATION_STATE_ORDER = ("RA", "UA", "Mid-D", "RD", "ND")
POLARITY_BIN_ORDER = ("Low-P", "Mid-P", "High-P")
CAUSE_GROUPS = (
    ("overall", "Overall", "Overall"),
    ("disagreement", "Low-D", "Low-D"),
    ("disagreement", "Mid-D", "Mid-D"),
    ("disagreement", "High-D", "High-D"),
    ("relation_state", "RA", "RA"),
    ("relation_state", "UA", "UA"),
    ("relation_state", "Mid-D", "Mid-D"),
    ("relation_state", "RD", "RD"),
    ("relation_state", "ND", "ND"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline v6 cause analysis over completed multi-seed disagreement runs. "
            "This script does not retrain models."
        )
    )
    parser.add_argument("--dataset", default="mosei")
    parser.add_argument(
        "--multi_seed_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Completed multi_seed_* directories to merge.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to outputs/<dataset>/v6_cause_analysis_<min>_<max>.",
    )
    parser.add_argument(
        "--expected_seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional exact seed set. The script fails if any are missing or duplicated.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for offline prediction. Default: cpu.",
    )
    parser.add_argument("--error_min_seeds", type=int, default=5)
    parser.add_argument("--error_sign_rate", type=float, default=0.8)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def discover_run_dirs(multi_seed_dirs: list[Path], dataset: str) -> list[tuple[int, Path]]:
    runs: list[tuple[int, Path]] = []
    for multi_seed_dir in multi_seed_dirs:
        run_root = multi_seed_dir / "runs" / dataset
        if not run_root.exists():
            raise FileNotFoundError(f"Missing run directory: {run_root}")
        for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue
            config = load_json(config_path)
            runs.append((int(config["seed"]), run_dir))
    return sorted(runs, key=lambda item: item[0])


def validate_seed_coverage(
    run_dirs: list[tuple[int, Path]],
    expected_seeds: list[int] | None,
) -> dict[str, object]:
    seeds = [seed for seed, _ in run_dirs]
    unique_seeds = sorted(set(seeds))
    duplicate_seeds = sorted(seed for seed in unique_seeds if seeds.count(seed) > 1)
    if expected_seeds is None or len(expected_seeds) == 0:
        missing_seeds: list[int] = []
        unexpected_seeds: list[int] = []
    else:
        expected = sorted(set(expected_seeds))
        missing_seeds = sorted(set(expected) - set(unique_seeds))
        unexpected_seeds = sorted(set(unique_seeds) - set(expected))
    ok = not duplicate_seeds and not missing_seeds and not unexpected_seeds
    return {
        "ok": ok,
        "seeds": unique_seeds,
        "n_runs": len(run_dirs),
        "n_unique_seeds": len(unique_seeds),
        "duplicate_seeds": duplicate_seeds,
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
    }


def t_critical_95(count: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    if count <= 30:
        return table.get(count, math.nan)
    return 1.96


def aggregate_numeric(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str] | None = None,
    *,
    sign_cols: list[str] | None = None,
    min_count: int = 5,
    sign_rate_threshold: float = 0.8,
) -> pd.DataFrame:
    sign_cols = sign_cols or []
    if value_cols is None:
        value_cols = [
            column
            for column in frame.columns
            if column not in group_cols
            and pd.api.types.is_numeric_dtype(frame[column])
            and column != "seed"
        ]
    rows: list[dict[str, object]] = []
    grouped = (
        [((), frame)]
        if not group_cols
        else frame.groupby(group_cols, dropna=False, sort=False)
    )
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_cols, keys))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for value in value_cols:
            series = pd.to_numeric(group[value], errors="coerce").dropna()
            count = int(series.shape[0])
            mean = float(series.mean()) if count else math.nan
            std = float(series.std(ddof=1)) if count > 1 else math.nan
            sem = std / math.sqrt(count) if count > 1 else math.nan
            critical = t_critical_95(count)
            margin = critical * sem if count > 1 and not math.isnan(critical) else math.nan
            row[f"{value}_mean"] = mean
            row[f"{value}_std"] = std
            row[f"{value}_count"] = count
            row[f"{value}_ci95_low"] = mean - margin if not math.isnan(margin) else math.nan
            row[f"{value}_ci95_high"] = mean + margin if not math.isnan(margin) else math.nan
            if value in sign_cols:
                positive_rate = float((series > 0).mean()) if count else math.nan
                negative_rate = float((series < 0).mean()) if count else math.nan
                sign_consistency = (
                    max(positive_rate, negative_rate)
                    if not math.isnan(positive_rate) and not math.isnan(negative_rate)
                    else math.nan
                )
                ci_low = row[f"{value}_ci95_low"]
                ci_high = row[f"{value}_ci95_high"]
                ci_excludes_zero = (
                    isinstance(ci_low, float)
                    and isinstance(ci_high, float)
                    and not math.isnan(ci_low)
                    and not math.isnan(ci_high)
                    and (ci_low > 0 or ci_high < 0)
                )
                row[f"{value}_positive_rate"] = positive_rate
                row[f"{value}_negative_rate"] = negative_rate
                row[f"{value}_sign_consistency"] = sign_consistency
                row[f"{value}_ci95_excludes_zero"] = ci_excludes_zero
                row[f"{value}_passes_error_control"] = bool(
                    count >= min_count
                    and sign_consistency >= sign_rate_threshold
                    and ci_excludes_zero
                )
        rows.append(row)
    return pd.DataFrame(rows)


def label_entropy(labels: np.ndarray, num_classes: int) -> float:
    if labels.size == 0:
        return math.nan
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def group_mask(groups: pd.DataFrame, group_type: str, group: str) -> np.ndarray:
    if group_type == "overall":
        return np.ones(len(groups), dtype=bool)
    if group_type == "disagreement":
        return (groups["group"].to_numpy() == group)
    if group_type == "relation_state":
        return (groups["relation_state"].to_numpy() == group)
    raise ValueError(f"Unknown group_type: {group_type}")


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"acc": math.nan, "macro_f1": math.nan}
    return classification_metrics(y_true, y_pred)


def safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return math.nan
    return float((y_true == y_pred).mean())


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if frame.shape[0] < 3:
        return math.nan
    if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan
    return float(frame["x"].corr(frame["y"], method=method))


def top2_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    if sorted_probs.shape[1] < 2:
        return np.zeros(sorted_probs.shape[0], dtype=np.float64)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def prediction_polarity_confidence(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] >= 3:
        return np.maximum(probs[:, 0], probs[:, 2])
    return probs.max(axis=1)


def polarity_bins(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], "Mid-P", dtype=object)
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return result
    q33, q66 = np.quantile(values[valid], [1.0 / 3.0, 2.0 / 3.0])
    if math.isclose(float(q33), float(q66)):
        return result
    result[valid & (values <= q33)] = "Low-P"
    result[valid & (values > q33) & (values <= q66)] = "Mid-P"
    result[valid & (values > q66)] = "High-P"
    return result


def load_model_predictions(
    run_dir: Path,
    config: dict,
    *,
    model_filename: str,
    batch_size: int | None,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    data_path = Path(config["data_path"])
    splits = load_npz_splits(data_path, label_mode=config.get("label_mode", "three_class"))
    input_dims = config.get("input_dims") or infer_input_dims(splits["train"])
    model = MultimodalClassifier(
        text_dim=int(input_dims["text"]),
        vision_dim=int(input_dims["vision"]),
        audio_dim=int(input_dims["audio"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        num_classes=int(config.get("num_classes", 3)),
        direct_add_alpha=0.0,
        direct_add_pair_mode=config.get("direct_add_pair_mode", "text_anchor"),
        use_nce_projection=bool(config.get("use_nce_projection", True)),
        nce_proj_dim=int(config.get("nce_proj_dim", 128)),
        use_dynamic_fusion=False,
    )
    state_path = run_dir / model_filename
    try:
        state = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    loader = DataLoader(
        MultimodalSplitDataset(splits["test"]),
        batch_size=int(batch_size or config.get("batch_size", 1024)),
        shuffle=False,
        num_workers=num_workers,
    )
    pred = predict(model, loader, device)
    order = np.argsort(pred["index"])
    return {
        key: value[order]
        for key, value in pred.items()
        if isinstance(value, np.ndarray) and value.shape[0] == order.shape[0]
    }


def per_run_profiles(
    seed: int,
    run_dir: Path,
    *,
    batch_size: int | None,
    num_workers: int,
    device: torch.device,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    config = load_json(run_dir / "config.json")
    groups = pd.read_csv(run_dir / "test_groups.csv", encoding="utf-8-sig")
    groups = groups.sort_values("index").reset_index(drop=True)
    diagnostic_pred = load_model_predictions(
        run_dir,
        config,
        model_filename="diagnostic_model.pt",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    concat_pred = load_model_predictions(
        run_dir,
        config,
        model_filename="concat_model.pt",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    y_true = diagnostic_pred["y_true"].astype(np.int64)
    if not np.array_equal(y_true, groups["label_cls"].to_numpy(dtype=np.int64)):
        raise ValueError(f"Prediction labels and test_groups labels differ in {run_dir}")
    if not np.array_equal(y_true, concat_pred["y_true"].astype(np.int64)):
        raise ValueError(f"Diagnostic and Concat labels differ in {run_dir}")

    num_classes = int(config.get("num_classes", int(y_true.max()) + 1))
    modality_preds = {
        "text": diagnostic_pred["prob_t"].argmax(axis=1).astype(np.int64),
        "audio": diagnostic_pred["prob_a"].argmax(axis=1).astype(np.int64),
        "vision": diagnostic_pred["prob_v"].argmax(axis=1).astype(np.int64),
        "fusion": diagnostic_pred["prob_f"].argmax(axis=1).astype(np.int64),
    }
    concat_fusion_pred = concat_pred["prob_f"].argmax(axis=1).astype(np.int64)
    label_abs_polarity = groups["label_reg"].abs().to_numpy(dtype=np.float64)
    pred_polarity_conf = prediction_polarity_confidence(diagnostic_pred["prob_f"])
    pred_confidence = diagnostic_pred["prob_f"].max(axis=1)
    pred_margin = top2_margin(diagnostic_pred["prob_f"])
    polarity_bin = polarity_bins(label_abs_polarity)
    candidate_stack = np.stack(list(modality_preds.values()), axis=1)
    any_correct = (candidate_stack == y_true[:, None]).any(axis=1)
    oracle_pred = np.where(any_correct, y_true, modality_preds["fusion"])

    cause_rows: list[dict[str, object]] = []
    modality_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    class_prior_rows: list[dict[str, object]] = []
    class_wise_rows: list[dict[str, object]] = []
    polarity_corr_rows: list[dict[str, object]] = []
    polarity_bin_rows: list[dict[str, object]] = []
    decoupled_payload: dict[str, object] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "groups": groups[["index", "group", "D_sample"]].copy(),
        "y_true": y_true.copy(),
        "concat_pred": concat_fusion_pred.copy(),
    }

    correlation_signals = {
        "label_abs_polarity": label_abs_polarity,
        "pred_polarity_conf": pred_polarity_conf,
        "pred_confidence": pred_confidence,
        "pred_margin": pred_margin,
        "R_sample": groups["R_sample"].to_numpy(dtype=np.float64),
    }
    d_values = groups["D_sample"].to_numpy(dtype=np.float64)
    for signal_name, signal_values in correlation_signals.items():
        for method in ("pearson", "spearman"):
            polarity_corr_rows.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "d_metric": "D_pred",
                    "polarity_signal": signal_name,
                    "correlation_type": method,
                    "correlation": safe_corr(d_values, signal_values, method),
                    "n": int(np.isfinite(d_values).sum()),
                }
            )

    for bin_name in POLARITY_BIN_ORDER:
        bin_mask = polarity_bin == bin_name
        for d_group in ("Low-D", "Mid-D", "High-D"):
            d_mask = groups["group"].to_numpy() == d_group
            mask = bin_mask & d_mask
            n = int(mask.sum())
            group_y = y_true[mask]
            concat_metrics = safe_metrics(group_y, concat_fusion_pred[mask])
            if n:
                counts = np.bincount(group_y.astype(np.int64), minlength=num_classes)
                majority_acc = float(counts.max() / counts.sum())
            else:
                majority_acc = math.nan
            polarity_bin_rows.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "polarity_bin": bin_name,
                    "group": d_group,
                    "n": n,
                    "avg_abs_label_reg": float(label_abs_polarity[mask].mean())
                    if n
                    else math.nan,
                    "avg_D_sample": float(d_values[mask].mean()) if n else math.nan,
                    "class_prior_majority_acc": majority_acc,
                    "concat_acc": concat_metrics["acc"],
                    "concat_macro_f1": concat_metrics["macro_f1"],
                }
            )

    for group_type, group, display_group in CAUSE_GROUPS:
        mask = group_mask(groups, group_type, group)
        n = int(mask.sum())
        group_y = y_true[mask]
        row: dict[str, object] = {
            "seed": seed,
            "run_dir": str(run_dir),
            "group_type": group_type,
            "group": display_group,
            "n": n,
            "avg_D_sample": float(groups.loc[mask, "D_sample"].mean()) if n else math.nan,
            "avg_R": float(groups.loc[mask, "R_sample"].mean()) if n else math.nan,
            "avg_label_reg": float(groups.loc[mask, "label_reg"].mean()) if n else math.nan,
            "avg_abs_label_reg": float(groups.loc[mask, "label_reg"].abs().mean())
            if n
            else math.nan,
            "label_entropy": label_entropy(group_y, num_classes),
        }
        for label in range(num_classes):
            row[f"class_{label}_ratio"] = float((group_y == label).mean()) if n else math.nan
        cause_rows.append(row)

        if n:
            counts = np.bincount(group_y.astype(np.int64), minlength=num_classes)
            majority_class = int(counts.argmax())
            majority_acc = float(counts.max() / counts.sum())
        else:
            majority_class = math.nan
            majority_acc = math.nan
        class_prior_row: dict[str, object] = {
            "seed": seed,
            "run_dir": str(run_dir),
            "group_type": group_type,
            "group": display_group,
            "n": n,
            "class_prior_majority_class": majority_class,
            "class_prior_majority_acc": majority_acc,
        }
        for label in range(num_classes):
            class_prior_row[f"class_{label}_ratio"] = (
                float((group_y == label).mean()) if n else math.nan
            )
        class_prior_rows.append(class_prior_row)

        metrics_row: dict[str, object] = {
            "seed": seed,
            "run_dir": str(run_dir),
            "group_type": group_type,
            "group": display_group,
            "n": n,
        }
        for name, values in modality_preds.items():
            metrics = safe_metrics(group_y, values[mask])
            metrics_row[f"{name}_acc"] = metrics["acc"]
            metrics_row[f"{name}_macro_f1"] = metrics["macro_f1"]
        concat_metrics = safe_metrics(group_y, concat_fusion_pred[mask])
        metrics_row["concat_fusion_acc"] = concat_metrics["acc"]
        metrics_row["concat_fusion_macro_f1"] = concat_metrics["macro_f1"]
        metrics_row["fusion_minus_text_macro_f1"] = (
            metrics_row["fusion_macro_f1"] - metrics_row["text_macro_f1"]
        )
        best_unimodal = max(
            metrics_row["text_macro_f1"],
            metrics_row["audio_macro_f1"],
            metrics_row["vision_macro_f1"],
        )
        metrics_row["best_unimodal_macro_f1"] = best_unimodal
        metrics_row["fusion_minus_best_unimodal_macro_f1"] = (
            metrics_row["fusion_macro_f1"] - best_unimodal
        )
        modality_rows.append(metrics_row)

        for label in range(num_classes):
            class_mask = mask & (y_true == label)
            class_n = int(class_mask.sum())
            class_row: dict[str, object] = {
                "seed": seed,
                "run_dir": str(run_dir),
                "group_type": group_type,
                "group": display_group,
                "class_id": label,
                "n": class_n,
                "class_ratio": float(class_n / n) if n else math.nan,
                "concat_acc": safe_accuracy(y_true[class_mask], concat_fusion_pred[class_mask]),
            }
            for name, values in modality_preds.items():
                class_row[f"{name}_acc"] = safe_accuracy(y_true[class_mask], values[class_mask])
            class_wise_rows.append(class_row)

        if group_type == "relation_state" and display_group in {"RD", "ND"}:
            oracle_metrics = safe_metrics(group_y, oracle_pred[mask])
            oracle_row = {
                "seed": seed,
                "run_dir": str(run_dir),
                "group": display_group,
                "n": n,
                "avg_D_sample": row["avg_D_sample"],
                "avg_R": row["avg_R"],
                "text_macro_f1": metrics_row["text_macro_f1"],
                "audio_macro_f1": metrics_row["audio_macro_f1"],
                "vision_macro_f1": metrics_row["vision_macro_f1"],
                "fusion_macro_f1": metrics_row["fusion_macro_f1"],
                "concat_fusion_macro_f1": metrics_row["concat_fusion_macro_f1"],
                "oracle_macro_f1": oracle_metrics["macro_f1"],
                "oracle_acc": oracle_metrics["acc"],
                "oracle_minus_fusion_macro_f1": (
                    oracle_metrics["macro_f1"] - metrics_row["fusion_macro_f1"]
                ),
            }
            oracle_rows.append(oracle_row)
    return (
        cause_rows,
        modality_rows,
        oracle_rows,
        class_prior_rows,
        class_wise_rows,
        polarity_corr_rows,
        polarity_bin_rows,
        decoupled_payload,
    )


def decoupled_d_group_rows(
    payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for source in payloads:
        source_seed = int(source["seed"])
        source_groups = source["groups"]
        source_y = source["y_true"]
        if not isinstance(source_groups, pd.DataFrame):
            raise TypeError("decoupled payload groups must be a DataFrame")
        for eval_payload in payloads:
            eval_seed = int(eval_payload["seed"])
            if eval_seed == source_seed:
                continue
            eval_y = eval_payload["y_true"]
            eval_pred = eval_payload["concat_pred"]
            if not np.array_equal(source_y, eval_y):
                raise ValueError(
                    f"Cannot decouple source seed {source_seed} and eval seed {eval_seed}: "
                    "test labels differ."
                )
            pair_metrics: dict[str, dict[str, float]] = {}
            for d_group in ("Low-D", "Mid-D", "High-D"):
                mask = source_groups["group"].to_numpy() == d_group
                group_y = source_y[mask]
                metrics = safe_metrics(group_y, eval_pred[mask])
                pair_metrics[d_group] = metrics
                rows.append(
                    {
                        "seed": source_seed,
                        "source_seed": source_seed,
                        "eval_seed": eval_seed,
                        "source_run_dir": source["run_dir"],
                        "eval_run_dir": eval_payload["run_dir"],
                        "group": d_group,
                        "n": int(mask.sum()),
                        "avg_D_sample": float(source_groups.loc[mask, "D_sample"].mean())
                        if int(mask.sum())
                        else math.nan,
                        "concat_acc": metrics["acc"],
                        "concat_macro_f1": metrics["macro_f1"],
                    }
                )
            contrast_rows.append(
                {
                    "seed": source_seed,
                    "source_seed": source_seed,
                    "eval_seed": eval_seed,
                    "high_minus_low_macro_f1": (
                        pair_metrics["High-D"]["macro_f1"]
                        - pair_metrics["Low-D"]["macro_f1"]
                    ),
                    "high_minus_mid_macro_f1": (
                        pair_metrics["High-D"]["macro_f1"]
                        - pair_metrics["Mid-D"]["macro_f1"]
                    ),
                    "mid_minus_low_macro_f1": (
                        pair_metrics["Mid-D"]["macro_f1"]
                        - pair_metrics["Low-D"]["macro_f1"]
                    ),
                    "high_minus_low_acc": (
                        pair_metrics["High-D"]["acc"] - pair_metrics["Low-D"]["acc"]
                    ),
                    "high_minus_mid_acc": (
                        pair_metrics["High-D"]["acc"] - pair_metrics["Mid-D"]["acc"]
                    ),
                    "mid_minus_low_acc": (
                        pair_metrics["Mid-D"]["acc"] - pair_metrics["Low-D"]["acc"]
                    ),
                }
            )
    return rows, contrast_rows


def read_summary_csv(multi_seed_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for multi_seed_dir in multi_seed_dirs:
        path = multi_seed_dir / "summary" / filename
        if path.exists():
            frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def method_delta_frame(
    multi_seed_dirs: list[Path],
    *,
    min_count: int,
    sign_rate_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("UncondAlign", "d_group", "multi_seed_delta_all.csv"),
        ("UncondAlign", "relation_state", "relation_state_delta_all.csv"),
        ("UncondInfoNCE", "d_group", "infonce_delta_all.csv"),
        ("UncondInfoNCE", "relation_state", "infonce_relation_state_delta_all.csv"),
        ("DynamicFusion", "d_group", "dynamic_fusion_delta_all.csv"),
        ("DynamicFusion", "relation_state", "dynamic_fusion_relation_state_delta_all.csv"),
        ("BalancedDirectAdd", "d_group", "balanced_direct_add_delta_all.csv"),
        (
            "BalancedDirectAdd",
            "relation_state",
            "balanced_direct_add_relation_state_delta_all.csv",
        ),
    ]
    rows = []
    for method, source_scope, filename in specs:
        frame = read_summary_csv(multi_seed_dirs, filename)
        if frame.empty:
            continue
        for _, record in frame.iterrows():
            group = str(record["group"])
            if source_scope == "d_group" and group not in {"Overall", "High-D"}:
                continue
            if source_scope == "relation_state" and group != "RD":
                continue
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "method": method,
                    "target": group,
                    "source_scope": source_scope,
                    "delta_acc": record.get("delta_acc", math.nan),
                    "delta_macro_f1": record.get("delta_macro_f1", math.nan),
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw
    summary = aggregate_numeric(
        raw,
        ["method", "target", "source_scope"],
        ["delta_acc", "delta_macro_f1"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=min_count,
        sign_rate_threshold=sign_rate_threshold,
    )
    return raw, summary


def add_sort_keys(frame: pd.DataFrame) -> pd.DataFrame:
    group_type_order = {"overall": 0, "disagreement": 1, "relation_state": 2}
    group_order = {
        ("overall", "Overall"): 0,
        ("disagreement", "Low-D"): 0,
        ("disagreement", "Mid-D"): 1,
        ("disagreement", "High-D"): 2,
        ("relation_state", "RA"): 0,
        ("relation_state", "UA"): 1,
        ("relation_state", "Mid-D"): 2,
        ("relation_state", "RD"): 3,
        ("relation_state", "ND"): 4,
    }
    result = frame.copy()
    if "group_type" not in result.columns and "group" in result.columns:
        simple_order = {"Overall": 0, "Low-D": 1, "Mid-D": 2, "High-D": 3}
        result["_group_order"] = result["group"].map(simple_order).fillna(99)
        return result.sort_values(["_group_order"]).drop(columns=["_group_order"])
    result["_group_type_order"] = result["group_type"].map(group_type_order).fillna(99)
    result["_group_order"] = [
        group_order.get((group_type, group), 99)
        for group_type, group in zip(result["group_type"], result["group"])
    ]
    return result.sort_values(["_group_type_order", "_group_order"]).drop(
        columns=["_group_type_order", "_group_order"]
    )


def save_polarity_correlation_plot(summary: pd.DataFrame, path: Path) -> None:
    plot_df = summary[
        (summary["d_metric"] == "D_pred")
        & (summary["correlation_type"] == "spearman")
    ].copy()
    if plot_df.empty:
        return
    order = [
        "label_abs_polarity",
        "pred_polarity_conf",
        "pred_confidence",
        "pred_margin",
        "R_sample",
    ]
    plot_df["polarity_signal"] = pd.Categorical(
        plot_df["polarity_signal"],
        categories=[value for value in order if value in set(plot_df["polarity_signal"])],
        ordered=True,
    )
    plot_df = plot_df.sort_values("polarity_signal")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        plot_df["polarity_signal"].astype(str),
        plot_df["correlation_mean"],
        color="#4C78A8",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Spearman corr. with D_pred")
    ax.set_xlabel("Polarity / confidence signal")
    ax.set_title("D_pred vs polarity/confidence correlation")
    ax.tick_params(axis="x", rotation=25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4 if height >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_polarity_bin_control_plot(summary: pd.DataFrame, path: Path) -> None:
    plot_df = summary[summary["group"].isin(("Low-D", "Mid-D", "High-D"))].copy()
    if plot_df.empty:
        return
    plot_df["polarity_bin"] = pd.Categorical(
        plot_df["polarity_bin"],
        categories=list(POLARITY_BIN_ORDER),
        ordered=True,
    )
    plot_df["group"] = pd.Categorical(
        plot_df["group"],
        categories=["Low-D", "Mid-D", "High-D"],
        ordered=True,
    )
    plot_df = plot_df.sort_values(["polarity_bin", "group"])
    bins = [value for value in POLARITY_BIN_ORDER if value in set(plot_df["polarity_bin"])]
    groups = ["Low-D", "Mid-D", "High-D"]
    x = np.arange(len(bins), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"Low-D": "#4C78A8", "Mid-D": "#F58518", "High-D": "#54A24B"}
    for offset_index, group in enumerate(groups):
        values = []
        for bin_name in bins:
            row = plot_df[
                (plot_df["polarity_bin"] == bin_name) & (plot_df["group"] == group)
            ]
            values.append(float(row["concat_acc_mean"].iloc[0]) if not row.empty else math.nan)
        ax.bar(
            x + (offset_index - 1) * width,
            values,
            width=width,
            label=group,
            color=colors[group],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Concat Acc")
    ax.set_xlabel("|label| polarity bin")
    ax.set_title("D group performance controlled by polarity bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dirs = discover_run_dirs(args.multi_seed_dirs, args.dataset)
    coverage = validate_seed_coverage(run_dirs, args.expected_seeds)
    if not coverage["ok"]:
        print(f"Seed coverage check failed: {coverage}", file=sys.stderr)
        return 2
    if not run_dirs:
        print("No completed seed runs found.", file=sys.stderr)
        return 2

    seeds = coverage["seeds"]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            ROOT
            / "outputs"
            / args.dataset
            / f"v6_cause_analysis_{min(seeds)}_{max(seeds)}"
        )
    output_dir = ensure_dir(output_dir)
    device = torch.device(args.device)

    cause_rows: list[dict[str, object]] = []
    modality_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    class_prior_rows: list[dict[str, object]] = []
    class_wise_rows: list[dict[str, object]] = []
    polarity_corr_rows: list[dict[str, object]] = []
    polarity_bin_rows: list[dict[str, object]] = []
    decoupled_payloads: list[dict[str, object]] = []
    for seed, run_dir in run_dirs:
        (
            per_cause,
            per_modality,
            per_oracle,
            per_class_prior,
            per_class_wise,
            per_polarity_corr,
            per_polarity_bin,
            per_decoupled_payload,
        ) = per_run_profiles(
            seed,
            run_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        cause_rows.extend(per_cause)
        modality_rows.extend(per_modality)
        oracle_rows.extend(per_oracle)
        class_prior_rows.extend(per_class_prior)
        class_wise_rows.extend(per_class_wise)
        polarity_corr_rows.extend(per_polarity_corr)
        polarity_bin_rows.extend(per_polarity_bin)
        decoupled_payloads.append(per_decoupled_payload)

    decoupled_rows, decoupled_contrast_rows = decoupled_d_group_rows(decoupled_payloads)

    cause_all = add_sort_keys(pd.DataFrame(cause_rows))
    modality_all = add_sort_keys(pd.DataFrame(modality_rows))
    oracle_all = pd.DataFrame(oracle_rows)
    class_prior_all = add_sort_keys(pd.DataFrame(class_prior_rows))
    class_wise_all = add_sort_keys(pd.DataFrame(class_wise_rows))
    polarity_corr_all = pd.DataFrame(polarity_corr_rows)
    polarity_bin_all = pd.DataFrame(polarity_bin_rows)
    decoupled_all = add_sort_keys(pd.DataFrame(decoupled_rows))
    decoupled_contrast_all = pd.DataFrame(decoupled_contrast_rows)

    cause_summary = add_sort_keys(
        aggregate_numeric(cause_all, ["group_type", "group"])
    )
    modality_summary = add_sort_keys(
        aggregate_numeric(modality_all, ["group_type", "group"])
    )
    oracle_summary = aggregate_numeric(oracle_all, ["group"])
    class_ratio_cols = [
        column
        for column in class_prior_all.columns
        if column.startswith("class_") and column.endswith("_ratio")
    ]
    class_prior_summary = add_sort_keys(
        aggregate_numeric(
            class_prior_all,
            ["group_type", "group"],
            value_cols=[
                "n",
                "class_prior_majority_acc",
                *class_ratio_cols,
            ],
        )
    )
    class_wise_summary = add_sort_keys(
        aggregate_numeric(
            class_wise_all,
            ["group_type", "group", "class_id"],
            value_cols=[
                "n",
                "class_ratio",
                "concat_acc",
                "text_acc",
                "audio_acc",
                "vision_acc",
                "fusion_acc",
            ],
        )
    )
    polarity_corr_summary = aggregate_numeric(
        polarity_corr_all,
        ["d_metric", "polarity_signal", "correlation_type"],
        value_cols=["correlation", "n"],
    )
    polarity_bin_summary = aggregate_numeric(
        polarity_bin_all,
        ["polarity_bin", "group"],
        value_cols=[
            "n",
            "avg_abs_label_reg",
            "avg_D_sample",
            "class_prior_majority_acc",
            "concat_acc",
            "concat_macro_f1",
        ],
    )
    decoupled_summary = add_sort_keys(
        aggregate_numeric(
            decoupled_all,
            ["group"],
            value_cols=["n", "avg_D_sample", "concat_acc", "concat_macro_f1"],
        )
    )
    decoupled_contrast_summary = aggregate_numeric(
        decoupled_contrast_all,
        ["source_seed"],
        value_cols=[
            "high_minus_low_macro_f1",
            "high_minus_mid_macro_f1",
            "mid_minus_low_macro_f1",
            "high_minus_low_acc",
            "high_minus_mid_acc",
            "mid_minus_low_acc",
        ],
        sign_cols=[
            "high_minus_low_macro_f1",
            "high_minus_mid_macro_f1",
            "mid_minus_low_macro_f1",
        ],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    decoupled_contrast_overall = aggregate_numeric(
        decoupled_contrast_all,
        [],
        value_cols=[
            "high_minus_low_macro_f1",
            "high_minus_mid_macro_f1",
            "mid_minus_low_macro_f1",
            "high_minus_low_acc",
            "high_minus_mid_acc",
            "mid_minus_low_acc",
        ],
        sign_cols=[
            "high_minus_low_macro_f1",
            "high_minus_mid_macro_f1",
            "mid_minus_low_macro_f1",
        ],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )

    method_raw, method_summary = method_delta_frame(
        args.multi_seed_dirs,
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )

    cause_all.to_csv(output_dir / "group_cause_profile_all.csv", index=False, encoding="utf-8-sig")
    cause_summary.to_csv(output_dir / "group_cause_profile.csv", index=False, encoding="utf-8-sig")
    modality_all.to_csv(
        output_dir / "group_unimodal_fusion_profile_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    modality_summary.to_csv(
        output_dir / "group_unimodal_fusion_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oracle_all.to_csv(output_dir / "rd_nd_oracle_profile_all.csv", index=False, encoding="utf-8-sig")
    oracle_summary.to_csv(
        output_dir / "rd_nd_oracle_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_prior_all.to_csv(
        output_dir / "group_class_prior_control_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_prior_summary.to_csv(
        output_dir / "group_class_prior_control.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_wise_all.to_csv(
        output_dir / "class_wise_accuracy_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_wise_summary.to_csv(
        output_dir / "class_wise_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    polarity_corr_all.to_csv(
        output_dir / "d_polarity_correlation_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    polarity_corr_summary.to_csv(
        output_dir / "d_polarity_correlation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    polarity_bin_all.to_csv(
        output_dir / "polarity_bin_d_control_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    polarity_bin_summary.to_csv(
        output_dir / "polarity_bin_d_control.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decoupled_all.to_csv(
        output_dir / "decoupled_d_group_eval_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decoupled_summary.to_csv(
        output_dir / "decoupled_d_group_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decoupled_contrast_all.to_csv(
        output_dir / "decoupled_high_d_contrast_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decoupled_contrast_summary.to_csv(
        output_dir / "decoupled_high_d_contrast_by_source_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decoupled_contrast_overall.to_csv(
        output_dir / "decoupled_high_d_contrast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_polarity_correlation_plot(
        polarity_corr_summary,
        output_dir / "d_polarity_correlation.png",
    )
    save_polarity_bin_control_plot(
        polarity_bin_summary,
        output_dir / "polarity_bin_d_control.png",
    )
    method_raw.to_csv(
        output_dir / "method_insufficiency_1_15_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    method_summary.to_csv(
        output_dir / "method_insufficiency_1_15.csv",
        index=False,
        encoding="utf-8-sig",
    )

    high_order_consistency = {}
    d_profile = modality_all[modality_all["group_type"] == "disagreement"]
    for seed, group in d_profile.groupby("seed"):
        values = {
            row["group"]: row["concat_fusion_macro_f1"]
            for _, row in group.iterrows()
            if row["group"] in {"Low-D", "Mid-D", "High-D"}
        }
        high_order_consistency[int(seed)] = bool(
            values.get("High-D", -math.inf)
            > values.get("Mid-D", math.inf)
            > values.get("Low-D", math.inf)
        )
    summary_payload = {
        "dataset": args.dataset,
        "multi_seed_dirs": [str(path) for path in args.multi_seed_dirs],
        "output_dir": str(output_dir),
        "seed_coverage": coverage,
        "concat_high_d_order_consistency": {
            "passed": int(sum(high_order_consistency.values())),
            "total": len(high_order_consistency),
            "by_seed": high_order_consistency,
        },
        "oracle_definition": (
            "Oracle uses the true label only for offline upper-bound analysis: "
            "if any diagnostic text/audio/vision/fusion prediction is correct, "
            "oracle prediction is y_true; otherwise it falls back to diagnostic fusion."
        ),
        "class_prior_majority_definition": (
            "Class-prior majority accuracy is an offline label-composition diagnostic: "
            "within each already-defined test group, it reports the largest true-label "
            "class ratio. It is not a deployable no-label baseline."
        ),
        "class_wise_accuracy_definition": (
            "Class-wise accuracy fixes the true class and reports recall-like accuracy "
            "within each disagreement or relation-state group; no single-class Macro-F1 "
            "is reported."
        ),
        "polarity_correlation_definition": (
            "D-polarity correlation reports Pearson/Spearman correlations between D_pred "
            "and offline polarity or prediction-confidence signals. label_abs_polarity "
            "uses |label_reg| and is label-aware diagnostic evidence."
        ),
        "polarity_bin_control_definition": (
            "Polarity-bin control splits each seed's test set into Low/Mid/High |label_reg| "
            "tertiles, then compares Low-D/Mid-D/High-D performance inside each bin."
        ),
        "decoupled_d_group_definition": (
            "Cross-seed decoupled diagnostic uses source_seed diagnostic D groups and "
            "evaluates them with eval_seed Concat predictions, excluding source_seed == eval_seed. "
            "This reduces the circularity of one model both defining and validating D groups."
        ),
    }
    save_json(output_dir / "v6_cause_analysis_summary.json", summary_payload)

    print(f"Saved v6 cause analysis to: {output_dir}")
    print("Core outputs:")
    for name in (
        "group_cause_profile.csv",
        "group_unimodal_fusion_profile.csv",
        "rd_nd_oracle_profile.csv",
        "group_class_prior_control.csv",
        "class_wise_accuracy.csv",
        "d_polarity_correlation.csv",
        "polarity_bin_d_control.csv",
        "decoupled_d_group_eval.csv",
        "decoupled_high_d_contrast.csv",
        "method_insufficiency_1_15.csv",
        "v6_cause_analysis_summary.json",
    ):
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
