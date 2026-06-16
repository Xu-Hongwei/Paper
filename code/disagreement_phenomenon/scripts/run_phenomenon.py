from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, ExperimentConfig  # noqa: E402
from src.data import MultimodalSplitDataset, infer_input_dims, load_npz_splits  # noqa: E402
from src.disagreement import (  # noqa: E402
    HIGH_D_RELIABILITY_GROUP_ORDER,
    RELATION_STATE_GROUP_ORDER,
    assign_high_d_reliability_groups,
    assign_relation_state_groups,
    assign_relation_state_groups_balanced,
    build_group_frame,
    grouped_metrics,
    kernel_pairwise_disagreement,
    pairwise_disagreement,
    reliability_threshold,
    relation_gates,
    resolve_kernel_bandwidth,
    rows_for_method,
    sample_disagreement_from_pairwise,
    sample_reliability,
    validation_thresholds,
    within_group_reliability_thresholds,
    assign_groups,
)
from src.model import MultimodalClassifier  # noqa: E402
from src.plotting import save_delta_plot, save_lambda_curve_plot  # noqa: E402
from src.train import predict, train_model  # noqa: E402
from src.utils import choose_device, ensure_dir, save_json, set_seed  # noqa: E402
from src.v4_analysis import (  # noqa: E402
    residual_probe_by_mode_frame,
    residual_probe_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CMU-MOSI/MOSEI disagreement phenomenon experiment."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\Xu\data\MultiBench"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT / "outputs",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eta_unimodal", type=float, default=0.1)
    parser.add_argument(
        "--label_mode",
        choices=("three_class", "binary"),
        default="three_class",
        help="Target label conversion. binary uses sentiment score > 0.",
    )
    parser.add_argument(
        "--lambda_align_values",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.05, 0.1],
    )
    parser.add_argument(
        "--direct_add_alpha_values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 1.0],
        help="Alpha sweep for pair-mode-aware DirectAdd.",
    )
    parser.add_argument(
        "--pair_mode",
        choices=("text_anchor", "full_pair"),
        default=None,
        help=(
            "Unified pair graph for this run. text_anchor uses T-A/T-V; "
            "full_pair also uses A-V."
        ),
    )
    parser.add_argument(
        "--align_pair_mode",
        choices=("text_anchor", "full_pair"),
        default=None,
        help="Deprecated override for UncondAlign; must match --pair_mode when set.",
    )
    parser.add_argument(
        "--direct_add_pair_mode",
        choices=("text_anchor", "full_pair"),
        default=None,
        help=(
            "DirectAdd appendix mode. text_anchor is reported as TextInject; "
            "BalancedDirectAdd is always run as a separate appendix baseline."
        ),
    )
    parser.add_argument(
        "--run_infonce",
        action="store_true",
        help="Train unconditional same-sample InfoNCE baselines.",
    )
    parser.add_argument(
        "--lambda_nce_values",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.5],
    )
    parser.add_argument("--nce_temperature", type=float, default=0.1)
    parser.add_argument(
        "--use_nce_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use projection heads z_m=P_m(h_m) for InfoNCE while classification uses h_m.",
    )
    parser.add_argument("--nce_proj_dim", type=int, default=128)
    parser.add_argument(
        "--nce_pair_mode",
        choices=("text_anchor", "full_pair"),
        default=None,
        help="Deprecated override for InfoNCE; must match --pair_mode when set.",
    )
    parser.add_argument(
        "--disagreement_metric",
        choices=("prob_jsd", "kernel_mmd"),
        default="prob_jsd",
        help="Metric used for Low/Mid/High-D and relation-state diagnostics.",
    )
    parser.add_argument(
        "--disagreement_pair_mode",
        choices=("text_anchor", "full_pair"),
        default=None,
        help="Deprecated override for prob_jsd D_sample; must match --pair_mode when set.",
    )
    parser.add_argument(
        "--kernel_bandwidth",
        default="median",
        help="RBF bandwidth for kernel_mmd diagnostics; use 'median' or a positive number.",
    )
    parser.add_argument(
        "--kernel_pair_mode",
        choices=("text_anchor", "full_pair"),
        default=None,
        help="Deprecated override for kernel D_sample; must match --pair_mode when set.",
    )
    parser.add_argument(
        "--kernel_class_weight",
        type=float,
        default=0.5,
        help="Blend weight for predicted-class MMD versus paired RBF distance.",
    )
    parser.add_argument(
        "--kernel_max_class_samples",
        type=int,
        default=1024,
        help="Maximum predicted-class samples used for each class-conditional MMD.",
    )
    parser.add_argument(
        "--relation_split",
        choices=("balanced_within_d", "global_r"),
        default="balanced_within_d",
        help=(
            "Relation-state reliability split. balanced_within_d uses validation "
            "Low-D/High-D medians; global_r uses one validation R median."
        ),
    )
    parser.add_argument(
        "--residual_modes",
        choices=("abs", "signed", "prod", "all"),
        nargs="+",
        default=["abs", "signed", "prod", "all"],
        help="Residual feature modes for supplementary probe diagnostics.",
    )
    parser.add_argument(
        "--tau_agreement",
        type=float,
        default=0.1,
        help="Temperature for diagnostic A_ij = exp(-D_ij/tau_agreement).",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable stricter deterministic seeding for lower run-to-run variance.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-epoch tqdm progress bars.",
    )
    args = parser.parse_args()
    resolve_pair_modes(args, parser)
    return args


def resolve_pair_modes(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    specific_names = (
        "align_pair_mode",
        "nce_pair_mode",
        "disagreement_pair_mode",
        "kernel_pair_mode",
    )
    explicit_modes = {
        getattr(args, name) for name in specific_names if getattr(args, name) is not None
    }
    if args.pair_mode is None:
        if len(explicit_modes) > 1:
            parser.error(
                "Pair-mode arguments conflict. Use a single --pair_mode, or make all "
                "specific *_pair_mode arguments identical."
            )
        args.pair_mode = next(iter(explicit_modes), "text_anchor")
    for name in specific_names:
        value = getattr(args, name)
        if value is None:
            setattr(args, name, args.pair_mode)
        elif value != args.pair_mode:
            parser.error(
                f"--{name}={value} conflicts with --pair_mode={args.pair_mode}. "
                "Use one consistent pair mode for the whole run."
            )
    if args.direct_add_pair_mode is None:
        args.direct_add_pair_mode = args.pair_mode


def make_loaders(
    splits: dict[str, object],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    datasets = {name: MultimodalSplitDataset(split) for name, split in splits.items()}
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=generator,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        ),
    }


def reset_training_rng(loaders: dict[str, DataLoader], cfg: ExperimentConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    train_generator = getattr(loaders["train"], "generator", None)
    if train_generator is not None:
        train_generator.manual_seed(cfg.seed)


def disagreement_distances(
    pred: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    *,
    kernel_bandwidth: str | float | None = None,
    seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    if cfg.disagreement_metric == "prob_jsd":
        return pairwise_disagreement(pred["prob_t"], pred["prob_v"], pred["prob_a"])
    if cfg.disagreement_metric == "kernel_mmd":
        return kernel_pairwise_disagreement(
            pred["h_t"],
            pred["h_v"],
            pred["h_a"],
            pred["y_pred"],
            bandwidth=cfg.kernel_bandwidth if kernel_bandwidth is None else kernel_bandwidth,
            class_weight=cfg.kernel_class_weight,
            max_class_samples=cfg.kernel_max_class_samples,
            seed=cfg.seed + seed_offset,
        )
    raise ValueError("disagreement_metric must be 'prob_jsd' or 'kernel_mmd'.")


def sample_disagreement_for_run(
    pred: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    *,
    kernel_bandwidth: str | float | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    distances = disagreement_distances(
        pred,
        cfg,
        kernel_bandwidth=kernel_bandwidth,
        seed_offset=seed_offset,
    )
    if cfg.disagreement_metric == "kernel_mmd":
        sample = sample_disagreement_from_pairwise(
            distances,
            pair_mode=cfg.kernel_pair_mode,
        )
    else:
        sample = sample_disagreement_from_pairwise(
            distances,
            pair_mode=cfg.disagreement_pair_mode,
        )
    return sample, distances


def new_model(
    input_dims: dict[str, int],
    cfg: ExperimentConfig,
    *,
    direct_add_alpha: float = 0.0,
    direct_add_pair_mode: str | None = None,
) -> MultimodalClassifier:
    return MultimodalClassifier(
        text_dim=input_dims["text"],
        vision_dim=input_dims["vision"],
        audio_dim=input_dims["audio"],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        direct_add_alpha=direct_add_alpha,
        direct_add_pair_mode=direct_add_pair_mode or cfg.direct_add_pair_mode,
        num_classes=cfg.num_classes,
        use_nce_projection=cfg.use_nce_projection,
        nce_proj_dim=cfg.nce_proj_dim,
    )


def train_concat(
    input_dims: dict[str, int],
    loaders: dict[str, DataLoader],
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[MultimodalClassifier, dict[str, float]]:
    model = new_model(input_dims, cfg)
    return train_model(
        model,
        loaders["train"],
        loaders["valid"],
        device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eta_unimodal=0.0,
        lambda_align=0.0,
        patience=cfg.patience,
        desc="Concat",
        show_progress=not cfg.quiet,
    )


def train_diagnostic(
    input_dims: dict[str, int],
    loaders: dict[str, DataLoader],
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[MultimodalClassifier, dict[str, float]]:
    model = new_model(input_dims, cfg)
    return train_model(
        model,
        loaders["train"],
        loaders["valid"],
        device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eta_unimodal=cfg.eta_unimodal,
        lambda_align=0.0,
        patience=cfg.patience,
        desc="Diagnostic",
        show_progress=not cfg.quiet,
    )


def train_best_alignment(
    input_dims: dict[str, int],
    loaders: dict[str, DataLoader],
    cfg: ExperimentConfig,
    device: torch.device,
    concat_grouped: dict[str, dict[str, float]],
    concat_reliability_grouped: dict[str, dict[str, float]],
    test_groups,
    test_reliability_groups,
) -> tuple[
    MultimodalClassifier,
    dict[str, float],
    float,
    list[dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
]:
    best_model: MultimodalClassifier | None = None
    best_metrics: dict[str, float] = {}
    best_grouped: dict[str, dict[str, float]] = {}
    best_reliability_grouped: dict[str, dict[str, float]] = {}
    best_lambda = cfg.lambda_align_values[0]
    best_score = -1.0
    sweep_rows: list[dict[str, float]] = []
    lambda_delta_rows: list[dict[str, float]] = []
    lambda_reliability_delta_rows: list[dict[str, float]] = []

    for lambda_align in cfg.lambda_align_values:
        reset_training_rng(loaders, cfg)
        model = new_model(input_dims, cfg)
        trained, metrics = train_model(
            model,
            loaders["train"],
            loaders["valid"],
            device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            eta_unimodal=0.0,
            lambda_align=lambda_align,
            align_pair_mode=cfg.align_pair_mode,
            patience=cfg.patience,
            desc=f"Align λ={lambda_align}",
            show_progress=not cfg.quiet,
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append(
            {
                "lambda_align": lambda_align,
                "align_pair_mode": cfg.align_pair_mode,
                **metrics,
            }
        )

        align_pred = predict(trained, loaders["test"], device)
        align_grouped = grouped_metrics(
            align_pred["y_true"],
            align_pred["y_pred"],
            test_groups,
        )
        align_reliability_grouped = grouped_metrics(
            align_pred["y_true"],
            align_pred["y_pred"],
            test_reliability_groups,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
        for group in ("Low-D", "Mid-D", "High-D", "Overall"):
            lambda_delta_rows.append(
                {
                    "lambda_align": lambda_align,
                    "group": group,
                    "n": align_grouped[group]["n"],
                    "concat_acc": concat_grouped[group]["acc"],
                    "align_acc": align_grouped[group]["acc"],
                    "delta_acc": align_grouped[group]["acc"] - concat_grouped[group]["acc"],
                    "concat_macro_f1": concat_grouped[group]["macro_f1"],
                    "align_macro_f1": align_grouped[group]["macro_f1"],
                    "delta_macro_f1": align_grouped[group]["macro_f1"]
                    - concat_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                    "align_pair_mode": cfg.align_pair_mode,
                }
            )
        for group in HIGH_D_RELIABILITY_GROUP_ORDER:
            lambda_reliability_delta_rows.append(
                {
                    "lambda_align": lambda_align,
                    "group": group,
                    "n": align_reliability_grouped[group]["n"],
                    "concat_acc": concat_reliability_grouped[group]["acc"],
                    "align_acc": align_reliability_grouped[group]["acc"],
                    "delta_acc": align_reliability_grouped[group]["acc"]
                    - concat_reliability_grouped[group]["acc"],
                    "concat_macro_f1": concat_reliability_grouped[group]["macro_f1"],
                    "align_macro_f1": align_reliability_grouped[group]["macro_f1"],
                    "delta_macro_f1": align_reliability_grouped[group]["macro_f1"]
                    - concat_reliability_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                    "align_pair_mode": cfg.align_pair_mode,
                }
            )
        if score > best_score:
            best_score = score
            best_model = trained
            best_metrics = metrics
            best_grouped = align_grouped
            best_reliability_grouped = align_reliability_grouped
            best_lambda = lambda_align

    if best_model is None:
        raise RuntimeError("Alignment sweep did not train any model.")
    return (
        best_model,
        best_metrics,
        best_lambda,
        sweep_rows,
        best_grouped,
        best_reliability_grouped,
        lambda_delta_rows,
        lambda_reliability_delta_rows,
    )


def train_best_direct_add(
    input_dims: dict[str, int],
    loaders: dict[str, DataLoader],
    cfg: ExperimentConfig,
    device: torch.device,
    concat_grouped: dict[str, dict[str, float]],
    test_groups,
    *,
    direct_add_pair_mode: str | None = None,
) -> tuple[
    MultimodalClassifier,
    dict[str, float],
    float,
    list[dict[str, float]],
    dict[str, dict[str, float]],
    list[dict[str, float]],
]:
    best_model: MultimodalClassifier | None = None
    best_metrics: dict[str, float] = {}
    best_grouped: dict[str, dict[str, float]] = {}
    best_alpha = cfg.direct_add_alpha_values[0]
    best_score = -1.0
    sweep_rows: list[dict[str, float]] = []
    alpha_delta_rows: list[dict[str, float]] = []
    direct_add_pair_mode = direct_add_pair_mode or cfg.direct_add_pair_mode

    for alpha in cfg.direct_add_alpha_values:
        reset_training_rng(loaders, cfg)
        model = new_model(
            input_dims,
            cfg,
            direct_add_alpha=alpha,
            direct_add_pair_mode=direct_add_pair_mode,
        )
        trained, metrics = train_model(
            model,
            loaders["train"],
            loaders["valid"],
            device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            eta_unimodal=0.0,
            lambda_align=0.0,
            patience=cfg.patience,
            desc=f"{direct_add_method_name(direct_add_pair_mode)} alpha={alpha}",
            show_progress=not cfg.quiet,
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append(
            {
                "direct_add_alpha": alpha,
                "direct_add_pair_mode": direct_add_pair_mode,
                **metrics,
            }
        )

        direct_pred = predict(trained, loaders["test"], device)
        direct_grouped = grouped_metrics(
            direct_pred["y_true"],
            direct_pred["y_pred"],
            test_groups,
        )
        for group in ("Low-D", "Mid-D", "High-D", "Overall"):
            alpha_delta_rows.append(
                {
                    "direct_add_alpha": alpha,
                    "group": group,
                    "n": direct_grouped[group]["n"],
                    "concat_acc": concat_grouped[group]["acc"],
                    "direct_add_acc": direct_grouped[group]["acc"],
                    "delta_acc": direct_grouped[group]["acc"] - concat_grouped[group]["acc"],
                    "concat_macro_f1": concat_grouped[group]["macro_f1"],
                    "direct_add_macro_f1": direct_grouped[group]["macro_f1"],
                    "delta_macro_f1": direct_grouped[group]["macro_f1"]
                    - concat_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                    "direct_add_pair_mode": direct_add_pair_mode,
                }
            )
        if score > best_score:
            best_score = score
            best_model = trained
            best_metrics = metrics
            best_grouped = direct_grouped
            best_alpha = alpha

    if best_model is None:
        raise RuntimeError("DirectAdd sweep did not train any model.")
    return (
        best_model,
        best_metrics,
        best_alpha,
        sweep_rows,
        best_grouped,
        alpha_delta_rows,
    )


def train_best_infonce(
    input_dims: dict[str, int],
    loaders: dict[str, DataLoader],
    cfg: ExperimentConfig,
    device: torch.device,
    concat_grouped: dict[str, dict[str, float]],
    concat_reliability_grouped: dict[str, dict[str, float]],
    test_groups,
    test_reliability_groups,
) -> tuple[
    MultimodalClassifier,
    dict[str, float],
    float,
    list[dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
]:
    best_model: MultimodalClassifier | None = None
    best_metrics: dict[str, float] = {}
    best_grouped: dict[str, dict[str, float]] = {}
    best_reliability_grouped: dict[str, dict[str, float]] = {}
    best_lambda = cfg.lambda_nce_values[0]
    best_score = -1.0
    sweep_rows: list[dict[str, float]] = []
    lambda_delta_rows: list[dict[str, float]] = []
    lambda_reliability_delta_rows: list[dict[str, float]] = []

    for lambda_nce in cfg.lambda_nce_values:
        reset_training_rng(loaders, cfg)
        model = new_model(input_dims, cfg)
        trained, metrics = train_model(
            model,
            loaders["train"],
            loaders["valid"],
            device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            eta_unimodal=0.0,
            lambda_nce=lambda_nce,
            nce_temperature=cfg.nce_temperature,
            nce_pair_mode=cfg.nce_pair_mode,
            patience=cfg.patience,
            desc=f"InfoNCE λ={lambda_nce}",
            show_progress=not cfg.quiet,
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append(
            {
                "lambda_nce": lambda_nce,
                "nce_temperature": cfg.nce_temperature,
                "nce_pair_mode": cfg.nce_pair_mode,
                "use_nce_projection": cfg.use_nce_projection,
                "nce_proj_dim": cfg.nce_proj_dim,
                **metrics,
            }
        )

        infonce_pred = predict(trained, loaders["test"], device)
        infonce_grouped = grouped_metrics(
            infonce_pred["y_true"],
            infonce_pred["y_pred"],
            test_groups,
        )
        infonce_reliability_grouped = grouped_metrics(
            infonce_pred["y_true"],
            infonce_pred["y_pred"],
            test_reliability_groups,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
        for group in ("Low-D", "Mid-D", "High-D", "Overall"):
            lambda_delta_rows.append(
                {
                    "lambda_nce": lambda_nce,
                    "nce_temperature": cfg.nce_temperature,
                    "nce_pair_mode": cfg.nce_pair_mode,
                    "use_nce_projection": cfg.use_nce_projection,
                    "nce_proj_dim": cfg.nce_proj_dim,
                    "group": group,
                    "n": infonce_grouped[group]["n"],
                    "concat_acc": concat_grouped[group]["acc"],
                    "infonce_acc": infonce_grouped[group]["acc"],
                    "delta_acc": infonce_grouped[group]["acc"] - concat_grouped[group]["acc"],
                    "concat_macro_f1": concat_grouped[group]["macro_f1"],
                    "infonce_macro_f1": infonce_grouped[group]["macro_f1"],
                    "delta_macro_f1": infonce_grouped[group]["macro_f1"]
                    - concat_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                }
            )
        for group in HIGH_D_RELIABILITY_GROUP_ORDER:
            lambda_reliability_delta_rows.append(
                {
                    "lambda_nce": lambda_nce,
                    "nce_temperature": cfg.nce_temperature,
                    "nce_pair_mode": cfg.nce_pair_mode,
                    "use_nce_projection": cfg.use_nce_projection,
                    "nce_proj_dim": cfg.nce_proj_dim,
                    "group": group,
                    "n": infonce_reliability_grouped[group]["n"],
                    "concat_acc": concat_reliability_grouped[group]["acc"],
                    "infonce_acc": infonce_reliability_grouped[group]["acc"],
                    "delta_acc": infonce_reliability_grouped[group]["acc"]
                    - concat_reliability_grouped[group]["acc"],
                    "concat_macro_f1": concat_reliability_grouped[group]["macro_f1"],
                    "infonce_macro_f1": infonce_reliability_grouped[group]["macro_f1"],
                    "delta_macro_f1": infonce_reliability_grouped[group]["macro_f1"]
                    - concat_reliability_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                }
            )
        if score > best_score:
            best_score = score
            best_model = trained
            best_metrics = metrics
            best_grouped = infonce_grouped
            best_reliability_grouped = infonce_reliability_grouped
            best_lambda = lambda_nce

    if best_model is None:
        raise RuntimeError("InfoNCE sweep did not train any model.")
    return (
        best_model,
        best_metrics,
        best_lambda,
        sweep_rows,
        best_grouped,
        best_reliability_grouped,
        lambda_delta_rows,
        lambda_reliability_delta_rows,
    )


def relation_state_split(
    train_groups: np.ndarray,
    valid_groups: np.ndarray,
    test_groups: np.ndarray,
    train_r: np.ndarray,
    valid_r: np.ndarray,
    test_r: np.ndarray,
    *,
    relation_split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    q_r = reliability_threshold(valid_r)
    if relation_split == "global_r":
        thresholds = {"global": q_r, "Low-D": q_r, "High-D": q_r}
        train_states = assign_relation_state_groups(train_groups, train_r, q_r)
        test_states = assign_relation_state_groups(test_groups, test_r, q_r)
        return train_states, test_states, thresholds
    if relation_split == "balanced_within_d":
        thresholds = within_group_reliability_thresholds(valid_groups, valid_r)
        train_states = assign_relation_state_groups_balanced(
            train_groups,
            train_r,
            thresholds,
        )
        test_states = assign_relation_state_groups_balanced(
            test_groups,
            test_r,
            thresholds,
        )
        return train_states, test_states, thresholds
    raise ValueError("relation_split must be 'balanced_within_d' or 'global_r'.")


def relation_state_calibration_frame(
    pred: dict[str, np.ndarray],
    relation_states: np.ndarray,
    reliability: dict[str, np.ndarray],
    *,
    label_mode: str,
    relation_split: str,
) -> pd.DataFrame:
    y_true = pred["y_true"]
    modality_preds = {
        "text_acc": pred["prob_t"].argmax(axis=1),
        "audio_acc": pred["prob_a"].argmax(axis=1),
        "vision_acc": pred["prob_v"].argmax(axis=1),
        "fusion_acc": pred["prob_f"].argmax(axis=1),
    }
    rows: list[dict[str, object]] = []
    classes = sorted(int(label) for label in np.unique(y_true))
    for group in RELATION_STATE_GROUP_ORDER:
        mask = relation_states == group
        n = int(mask.sum())
        row: dict[str, object] = {
            "group": group,
            "relation_state_desc": {
                "RA": "Low-D+High-R",
                "UA": "Low-D+Low-R",
                "Mid-D": "Mid-D",
                "RD": "High-D+High-R",
                "ND": "High-D+Low-R",
            }.get(group, group),
            "label_mode": label_mode,
            "relation_split": relation_split,
            "n": n,
            "avg_R": float(np.mean(reliability["R_sample"][mask])) if n else float("nan"),
            "avg_R_text": float(np.mean(reliability["R_text"][mask])) if n else float("nan"),
            "avg_R_audio": float(np.mean(reliability["R_audio"][mask])) if n else float("nan"),
            "avg_R_vision": float(np.mean(reliability["R_vision"][mask])) if n else float("nan"),
        }
        for label in classes:
            row[f"class_{label}_ratio"] = (
                float((y_true[mask] == label).mean()) if n else float("nan")
            )
        for name, y_pred in modality_preds.items():
            row[name] = float((y_pred[mask] == y_true[mask]).mean()) if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def direct_add_method_name(direct_add_pair_mode: str) -> str:
    if direct_add_pair_mode == "text_anchor":
        return "TextInject"
    return "DirectAdd"


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        eta_unimodal=args.eta_unimodal,
        label_mode=args.label_mode,
        lambda_align_values=args.lambda_align_values,
        direct_add_alpha_values=args.direct_add_alpha_values,
        pair_mode=args.pair_mode,
        align_pair_mode=args.align_pair_mode,
        direct_add_pair_mode=args.direct_add_pair_mode,
        run_infonce=args.run_infonce,
        lambda_nce_values=args.lambda_nce_values,
        nce_temperature=args.nce_temperature,
        nce_pair_mode=args.nce_pair_mode,
        use_nce_projection=args.use_nce_projection,
        nce_proj_dim=args.nce_proj_dim,
        disagreement_metric=args.disagreement_metric,
        disagreement_pair_mode=args.disagreement_pair_mode,
        kernel_bandwidth=args.kernel_bandwidth,
        kernel_pair_mode=args.kernel_pair_mode,
        kernel_class_weight=args.kernel_class_weight,
        kernel_max_class_samples=args.kernel_max_class_samples,
        relation_split=args.relation_split,
        residual_modes=args.residual_modes,
        tau_agreement=args.tau_agreement,
        patience=args.patience,
        deterministic=args.deterministic,
        quiet=args.quiet,
    )

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = choose_device()

    print(f"Dataset: {cfg.dataset}")
    print(f"Data file: {cfg.data_path}")
    print(f"Label mode: {cfg.label_mode}")
    print(f"Relation split: {cfg.relation_split}")
    print(f"Device: {device}")
    print(f"Deterministic: {cfg.deterministic}")

    try:
        splits = load_npz_splits(cfg.data_path, label_mode=cfg.label_mode)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_dir = ensure_dir(
        cfg.output_root
        / cfg.dataset
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print(f"Output: {run_dir}")

    input_dims = infer_input_dims(splits["train"])
    loaders = make_loaders(splits, cfg.batch_size, cfg.num_workers, cfg.seed)

    save_json(
        run_dir / "config.json",
        {
            "dataset": cfg.dataset,
            "data_path": str(cfg.data_path),
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "eta_unimodal": cfg.eta_unimodal,
            "label_mode": cfg.label_mode,
            "num_classes": cfg.num_classes,
            "lambda_align_values": cfg.lambda_align_values,
            "direct_add_alpha_values": cfg.direct_add_alpha_values,
            "pair_mode": cfg.pair_mode,
            "align_pair_mode": cfg.align_pair_mode,
            "direct_add_pair_mode": cfg.direct_add_pair_mode,
            "run_infonce": cfg.run_infonce,
            "lambda_nce_values": cfg.lambda_nce_values,
            "nce_temperature": cfg.nce_temperature,
            "nce_pair_mode": cfg.nce_pair_mode,
            "use_nce_projection": cfg.use_nce_projection,
            "nce_proj_dim": cfg.nce_proj_dim,
            "disagreement_metric": cfg.disagreement_metric,
            "disagreement_pair_mode": cfg.disagreement_pair_mode,
            "kernel_bandwidth": cfg.kernel_bandwidth,
            "kernel_pair_mode": cfg.kernel_pair_mode,
            "kernel_class_weight": cfg.kernel_class_weight,
            "kernel_max_class_samples": cfg.kernel_max_class_samples,
            "relation_split": cfg.relation_split,
            "residual_modes": cfg.residual_modes,
            "tau_agreement": cfg.tau_agreement,
            "deterministic": cfg.deterministic,
            "input_dims": input_dims,
            "quiet": cfg.quiet,
        },
    )

    diagnostic_model, diagnostic_valid = train_diagnostic(input_dims, loaders, cfg, device)
    train_diag = predict(diagnostic_model, loaders["train"], device)
    valid_diag = predict(diagnostic_model, loaders["valid"], device)
    test_diag = predict(diagnostic_model, loaders["test"], device)
    resolved_kernel_bandwidth: str | float = cfg.kernel_bandwidth
    if cfg.disagreement_metric == "kernel_mmd":
        resolved_kernel_bandwidth = resolve_kernel_bandwidth(
            valid_diag["h_t"],
            valid_diag["h_v"],
            valid_diag["h_a"],
            cfg.kernel_bandwidth,
            seed=cfg.seed,
        )
    train_d, train_distances = sample_disagreement_for_run(
        train_diag,
        cfg,
        kernel_bandwidth=resolved_kernel_bandwidth,
        seed_offset=30,
    )
    valid_d, valid_distances = sample_disagreement_for_run(
        valid_diag,
        cfg,
        kernel_bandwidth=resolved_kernel_bandwidth,
        seed_offset=40,
    )
    test_d, test_distances = sample_disagreement_for_run(
        test_diag,
        cfg,
        kernel_bandwidth=resolved_kernel_bandwidth,
        seed_offset=50,
    )
    q33, q66 = validation_thresholds(valid_d)
    train_groups = assign_groups(train_d, q33, q66)
    valid_groups = assign_groups(valid_d, q33, q66)
    test_groups = assign_groups(test_d, q33, q66)
    train_reliability = sample_reliability(
        train_diag["prob_t"],
        train_diag["prob_v"],
        train_diag["prob_a"],
        pair_mode=cfg.pair_mode,
    )
    valid_reliability = sample_reliability(
        valid_diag["prob_t"],
        valid_diag["prob_v"],
        valid_diag["prob_a"],
        pair_mode=cfg.pair_mode,
    )
    test_reliability = sample_reliability(
        test_diag["prob_t"],
        test_diag["prob_v"],
        test_diag["prob_a"],
        pair_mode=cfg.pair_mode,
    )
    train_relation_states, test_relation_states, relation_thresholds = relation_state_split(
        train_groups,
        valid_groups,
        test_groups,
        train_reliability["R_sample"],
        valid_reliability["R_sample"],
        test_reliability["R_sample"],
        relation_split=cfg.relation_split,
    )
    q_r = relation_thresholds["global"]
    test_relations = {
        **test_distances,
        **relation_gates(
            test_diag["prob_t"],
            test_diag["prob_v"],
            test_diag["prob_a"],
            test_reliability,
            cfg.tau_agreement,
            prefix="R_",
            distances=test_distances,
        ),
    }
    test_reliability_groups = assign_high_d_reliability_groups(
        test_groups,
        test_reliability["R_sample"],
        relation_thresholds["High-D"]
        if cfg.relation_split == "balanced_within_d"
        else q_r,
    )
    group_df = build_group_frame(
        test_diag,
        test_d,
        test_groups,
        reliability=test_reliability,
        reliability_groups=test_reliability_groups,
        relation_state_groups=test_relation_states,
        relations=test_relations,
    )
    group_df["pair_mode"] = cfg.pair_mode
    group_df["disagreement_metric"] = cfg.disagreement_metric
    group_df["disagreement_pair_mode"] = cfg.disagreement_pair_mode
    group_df["kernel_pair_mode"] = cfg.kernel_pair_mode
    group_df["label_mode"] = cfg.label_mode
    group_df["relation_split"] = cfg.relation_split
    group_df["resolved_kernel_bandwidth"] = (
        resolved_kernel_bandwidth if cfg.disagreement_metric == "kernel_mmd" else ""
    )
    group_df.to_csv(run_dir / "test_groups.csv", index=False, encoding="utf-8-sig")

    calibration_df = relation_state_calibration_frame(
        test_diag,
        test_relation_states,
        test_reliability,
        label_mode=cfg.label_mode,
        relation_split=cfg.relation_split,
    )
    calibration_df.to_csv(
        run_dir / "relation_state_distribution_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    residual_probe_df = residual_probe_frame(
        train_diag,
        test_diag,
        train_relation_states,
        test_relation_states,
        cfg.seed,
    )
    residual_probe_df.to_csv(
        run_dir / "residual_discriminative_probe.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_probe_by_mode_df = residual_probe_by_mode_frame(
        train_diag,
        test_diag,
        train_relation_states,
        test_relation_states,
        cfg.seed,
        residual_modes=cfg.residual_modes,
    )
    residual_probe_by_mode_df.to_csv(
        run_dir / "residual_probe_by_mode.csv",
        index=False,
        encoding="utf-8-sig",
    )

    concat_model, concat_valid = train_concat(input_dims, loaders, cfg, device)
    concat_pred = predict(concat_model, loaders["test"], device)
    concat_grouped = grouped_metrics(concat_pred["y_true"], concat_pred["y_pred"], test_groups)
    concat_reliability_grouped = grouped_metrics(
        concat_pred["y_true"],
        concat_pred["y_pred"],
        test_reliability_groups,
        group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
        include_overall=False,
    )

    (
        align_model,
        align_valid,
        best_lambda,
        sweep_rows,
        align_grouped,
        align_reliability_grouped,
        lambda_delta_rows,
        lambda_reliability_delta_rows,
    ) = train_best_alignment(
        input_dims,
        loaders,
        cfg,
        device,
        concat_grouped,
        concat_reliability_grouped,
        test_groups,
        test_reliability_groups,
    )
    align_pred = predict(align_model, loaders["test"], device)
    concat_relation_grouped = grouped_metrics(
        concat_pred["y_true"],
        concat_pred["y_pred"],
        test_relation_states,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )
    align_relation_grouped = grouped_metrics(
        align_pred["y_true"],
        align_pred["y_pred"],
        test_relation_states,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )

    (
        direct_add_model,
        direct_add_valid,
        best_direct_add_alpha,
        direct_add_sweep_rows,
        direct_add_grouped,
        direct_add_alpha_delta_rows,
    ) = train_best_direct_add(
        input_dims,
        loaders,
        cfg,
        device,
        concat_grouped,
        test_groups,
    )
    direct_add_pred = predict(direct_add_model, loaders["test"], device)
    direct_add_relation_grouped = grouped_metrics(
        direct_add_pred["y_true"],
        direct_add_pred["y_pred"],
        test_relation_states,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )
    pd.DataFrame(direct_add_sweep_rows).to_csv(
        run_dir / "direct_add_alpha_sweep_valid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(direct_add_alpha_delta_rows).to_csv(
        run_dir / "direct_add_alpha_test_delta_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_delta_rows = []
    for group in ("Low-D", "Mid-D", "High-D", "Overall"):
        direct_add_delta_rows.append(
            {
                "group": group,
                "delta_acc": direct_add_grouped[group]["acc"] - concat_grouped[group]["acc"],
                "delta_macro_f1": direct_add_grouped[group]["macro_f1"]
                - concat_grouped[group]["macro_f1"],
                "direct_add_alpha": best_direct_add_alpha,
                "direct_add_pair_mode": cfg.direct_add_pair_mode,
            }
        )
    direct_add_delta_df = pd.DataFrame(direct_add_delta_rows)
    direct_add_delta_df.to_csv(
        run_dir / "direct_add_delta_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        balanced_direct_add_model,
        balanced_direct_add_valid,
        best_balanced_direct_add_alpha,
        balanced_direct_add_sweep_rows,
        balanced_direct_add_grouped,
        balanced_direct_add_alpha_delta_rows,
    ) = train_best_direct_add(
        input_dims,
        loaders,
        cfg,
        device,
        concat_grouped,
        test_groups,
        direct_add_pair_mode="balanced",
    )
    balanced_direct_add_pred = predict(balanced_direct_add_model, loaders["test"], device)
    balanced_direct_add_relation_grouped = grouped_metrics(
        balanced_direct_add_pred["y_true"],
        balanced_direct_add_pred["y_pred"],
        test_relation_states,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )
    pd.DataFrame(balanced_direct_add_sweep_rows).to_csv(
        run_dir / "balanced_direct_add_alpha_sweep_valid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(balanced_direct_add_alpha_delta_rows).to_csv(
        run_dir / "balanced_direct_add_alpha_test_delta_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    balanced_direct_add_delta_rows = []
    for group in ("Low-D", "Mid-D", "High-D", "Overall"):
        balanced_direct_add_delta_rows.append(
            {
                "group": group,
                "delta_acc": balanced_direct_add_grouped[group]["acc"]
                - concat_grouped[group]["acc"],
                "delta_macro_f1": balanced_direct_add_grouped[group]["macro_f1"]
                - concat_grouped[group]["macro_f1"],
                "direct_add_alpha": best_balanced_direct_add_alpha,
                "direct_add_pair_mode": "balanced",
            }
        )
    balanced_direct_add_delta_df = pd.DataFrame(balanced_direct_add_delta_rows)
    balanced_direct_add_delta_df.to_csv(
        run_dir / "balanced_direct_add_delta_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    infonce_model = None
    infonce_valid = None
    best_lambda_nce = None
    infonce_grouped = None
    infonce_reliability_grouped = None
    infonce_relation_grouped = None
    infonce_delta_df = pd.DataFrame()
    infonce_reliability_delta_df = pd.DataFrame()
    if cfg.run_infonce:
        (
            infonce_model,
            infonce_valid,
            best_lambda_nce,
            infonce_sweep_rows,
            infonce_grouped,
            infonce_reliability_grouped,
            infonce_lambda_delta_rows,
            infonce_lambda_reliability_delta_rows,
        ) = train_best_infonce(
            input_dims,
            loaders,
            cfg,
            device,
            concat_grouped,
            concat_reliability_grouped,
            test_groups,
            test_reliability_groups,
        )
        infonce_pred = predict(infonce_model, loaders["test"], device)
        infonce_relation_grouped = grouped_metrics(
            infonce_pred["y_true"],
            infonce_pred["y_pred"],
            test_relation_states,
            group_order=RELATION_STATE_GROUP_ORDER,
            include_overall=False,
        )
        pd.DataFrame(infonce_sweep_rows).to_csv(
            run_dir / "infonce_lambda_sweep_valid.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(infonce_lambda_delta_rows).to_csv(
            run_dir / "infonce_lambda_test_delta_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(infonce_lambda_reliability_delta_rows).to_csv(
            run_dir / "infonce_lambda_high_d_reliability_delta.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_delta_rows = []
        for group in ("Low-D", "Mid-D", "High-D", "Overall"):
            infonce_delta_rows.append(
                {
                    "group": group,
                    "delta_acc": infonce_grouped[group]["acc"] - concat_grouped[group]["acc"],
                    "delta_macro_f1": infonce_grouped[group]["macro_f1"]
                    - concat_grouped[group]["macro_f1"],
                    "lambda_nce": best_lambda_nce,
                    "nce_temperature": cfg.nce_temperature,
                    "nce_pair_mode": cfg.nce_pair_mode,
                    "use_nce_projection": cfg.use_nce_projection,
                    "nce_proj_dim": cfg.nce_proj_dim,
                }
            )
        infonce_delta_df = pd.DataFrame(infonce_delta_rows)
        infonce_delta_df.to_csv(
            run_dir / "infonce_delta_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_reliability_delta_rows = []
        for group in HIGH_D_RELIABILITY_GROUP_ORDER:
            infonce_reliability_delta_rows.append(
                {
                    "group": group,
                    "delta_acc": infonce_reliability_grouped[group]["acc"]
                    - concat_reliability_grouped[group]["acc"],
                    "delta_macro_f1": infonce_reliability_grouped[group]["macro_f1"]
                    - concat_reliability_grouped[group]["macro_f1"],
                    "lambda_nce": best_lambda_nce,
                    "nce_temperature": cfg.nce_temperature,
                    "nce_pair_mode": cfg.nce_pair_mode,
                    "use_nce_projection": cfg.use_nce_projection,
                    "nce_proj_dim": cfg.nce_proj_dim,
                }
            )
        infonce_reliability_delta_df = pd.DataFrame(infonce_reliability_delta_rows)
        infonce_reliability_delta_df.to_csv(
            run_dir / "infonce_high_d_reliability_delta.csv",
            index=False,
            encoding="utf-8-sig",
        )

    relation_state_rows = []
    relation_state_rows.extend(
        rows_for_method(
            "Concat",
            concat_relation_grouped,
            group_order=RELATION_STATE_GROUP_ORDER,
            include_overall=False,
        )
    )
    relation_state_rows.extend(
        rows_for_method(
            "UncondAlign",
            align_relation_grouped,
            lambda_align=best_lambda,
            group_order=RELATION_STATE_GROUP_ORDER,
            include_overall=False,
        )
    )
    direct_add_relation_rows = rows_for_method(
        direct_add_method_name(cfg.direct_add_pair_mode),
        direct_add_relation_grouped,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )
    for row in direct_add_relation_rows:
        row["direct_add_alpha"] = best_direct_add_alpha
        row["direct_add_pair_mode"] = cfg.direct_add_pair_mode
    relation_state_rows.extend(direct_add_relation_rows)
    balanced_direct_add_relation_rows = rows_for_method(
        "BalancedDirectAdd",
        balanced_direct_add_relation_grouped,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )
    for row in balanced_direct_add_relation_rows:
        row["direct_add_alpha"] = best_balanced_direct_add_alpha
        row["direct_add_pair_mode"] = "balanced"
    relation_state_rows.extend(balanced_direct_add_relation_rows)
    if cfg.run_infonce and infonce_relation_grouped is not None:
        infonce_relation_rows = rows_for_method(
            "UncondInfoNCE",
            infonce_relation_grouped,
            group_order=RELATION_STATE_GROUP_ORDER,
            include_overall=False,
        )
        for row in infonce_relation_rows:
            row["lambda_nce"] = best_lambda_nce
            row["nce_temperature"] = cfg.nce_temperature
            row["nce_pair_mode"] = cfg.nce_pair_mode
            row["use_nce_projection"] = cfg.use_nce_projection
            row["nce_proj_dim"] = cfg.nce_proj_dim
        relation_state_rows.extend(infonce_relation_rows)
    for row in relation_state_rows:
        if row.get("method") == "UncondAlign":
            row["align_pair_mode"] = cfg.align_pair_mode
    relation_state_metrics_df = pd.DataFrame(relation_state_rows)
    relation_state_metrics_df.to_csv(
        run_dir / "relation_state_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    relation_state_delta_rows = []
    for group in RELATION_STATE_GROUP_ORDER:
        relation_state_delta_rows.append(
            {
                "group": group,
                "delta_acc": align_relation_grouped[group]["acc"]
                - concat_relation_grouped[group]["acc"],
                "delta_macro_f1": align_relation_grouped[group]["macro_f1"]
                - concat_relation_grouped[group]["macro_f1"],
                "lambda_align": best_lambda,
                "align_pair_mode": cfg.align_pair_mode,
            }
        )
    relation_state_delta_df = pd.DataFrame(relation_state_delta_rows)
    relation_state_delta_df.to_csv(
        run_dir / "relation_state_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    relation_state_delta_df.to_csv(
        run_dir / "uncond_align_relation_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_relation_state_delta_rows = []
    for group in RELATION_STATE_GROUP_ORDER:
        direct_add_relation_state_delta_rows.append(
            {
                "group": group,
                "delta_acc": direct_add_relation_grouped[group]["acc"]
                - concat_relation_grouped[group]["acc"],
                "delta_macro_f1": direct_add_relation_grouped[group]["macro_f1"]
                - concat_relation_grouped[group]["macro_f1"],
                "direct_add_alpha": best_direct_add_alpha,
                "direct_add_pair_mode": cfg.direct_add_pair_mode,
            }
        )
    direct_add_relation_state_delta_df = pd.DataFrame(direct_add_relation_state_delta_rows)
    direct_add_relation_state_delta_df.to_csv(
        run_dir / "direct_add_relation_state_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    balanced_direct_add_relation_state_delta_rows = []
    for group in RELATION_STATE_GROUP_ORDER:
        balanced_direct_add_relation_state_delta_rows.append(
            {
                "group": group,
                "delta_acc": balanced_direct_add_relation_grouped[group]["acc"]
                - concat_relation_grouped[group]["acc"],
                "delta_macro_f1": balanced_direct_add_relation_grouped[group]["macro_f1"]
                - concat_relation_grouped[group]["macro_f1"],
                "direct_add_alpha": best_balanced_direct_add_alpha,
                "direct_add_pair_mode": "balanced",
            }
        )
    balanced_direct_add_relation_state_delta_df = pd.DataFrame(
        balanced_direct_add_relation_state_delta_rows
    )
    balanced_direct_add_relation_state_delta_df.to_csv(
        run_dir / "balanced_direct_add_relation_state_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if cfg.run_infonce and infonce_relation_grouped is not None:
        infonce_relation_state_delta_rows = []
        for group in RELATION_STATE_GROUP_ORDER:
            infonce_relation_state_delta_rows.append(
                {
                    "group": group,
                    "delta_acc": infonce_relation_grouped[group]["acc"]
                    - concat_relation_grouped[group]["acc"],
                    "delta_macro_f1": infonce_relation_grouped[group]["macro_f1"]
                    - concat_relation_grouped[group]["macro_f1"],
                    "lambda_nce": best_lambda_nce,
                    "nce_temperature": cfg.nce_temperature,
                    "nce_pair_mode": cfg.nce_pair_mode,
                    "use_nce_projection": cfg.use_nce_projection,
                    "nce_proj_dim": cfg.nce_proj_dim,
                }
            )
        infonce_relation_delta_df = pd.DataFrame(infonce_relation_state_delta_rows)
        infonce_relation_delta_df.to_csv(
            run_dir / "infonce_relation_state_delta.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_relation_delta_df.to_csv(
            run_dir / "infonce_relation_delta.csv",
            index=False,
            encoding="utf-8-sig",
        )
    probe_by_group = {
        str(row["group"]): row for _, row in residual_probe_df.iterrows()
    }
    concat_aware_rows = []
    for group in RELATION_STATE_GROUP_ORDER:
        probe_row = probe_by_group.get(group, {})
        concat_aware_rows.append(
            {
                "group": group,
                "concat_macro_f1": concat_relation_grouped[group]["macro_f1"],
                "uncond_align_macro_f1": align_relation_grouped[group]["macro_f1"],
                "direct_add_macro_f1": direct_add_relation_grouped[group]["macro_f1"],
                "balanced_direct_add_macro_f1": balanced_direct_add_relation_grouped[
                    group
                ]["macro_f1"],
                "infonce_macro_f1": (
                    infonce_relation_grouped[group]["macro_f1"]
                    if infonce_relation_grouped is not None
                    else float("nan")
                ),
                "soft_split_probe_macro_f1": probe_row.get(
                    "common_residual_macro_f1",
                    float("nan"),
                ),
                "text_anchor_probe_macro_f1": probe_row.get(
                    "text_anchor_common_residual_macro_f1",
                    float("nan"),
                ),
                "residual_gain_macro_f1": probe_row.get(
                    "residual_gain_macro_f1",
                    float("nan"),
                ),
                "text_anchor_residual_gain_macro_f1": probe_row.get(
                    "text_anchor_residual_gain_macro_f1",
                    float("nan"),
                ),
                "shuffled_residual_only_macro_f1": probe_row.get(
                    "shuffled_residual_only_macro_f1",
                    float("nan"),
                ),
                "text_anchor_shuffled_residual_macro_f1": probe_row.get(
                    "text_anchor_shuffled_residual_macro_f1",
                    float("nan"),
                ),
                "common_shuffled_residual_macro_f1": probe_row.get(
                    "common_shuffled_residual_macro_f1",
                    float("nan"),
                ),
                "residual_gain_vs_feature_shuffle_macro_f1": probe_row.get(
                    "residual_gain_vs_feature_shuffle_macro_f1",
                    float("nan"),
                ),
                "text_anchor_common_shuffled_residual_macro_f1": probe_row.get(
                    "text_anchor_common_shuffled_residual_macro_f1",
                    float("nan"),
                ),
                "text_anchor_residual_gain_vs_feature_shuffle_macro_f1": probe_row.get(
                    "text_anchor_residual_gain_vs_feature_shuffle_macro_f1",
                    float("nan"),
                ),
                "lambda_align": best_lambda,
                "align_pair_mode": cfg.align_pair_mode,
                "direct_add_alpha": best_direct_add_alpha,
                "direct_add_pair_mode": cfg.direct_add_pair_mode,
                "balanced_direct_add_alpha": best_balanced_direct_add_alpha,
                "lambda_nce": best_lambda_nce,
                "nce_pair_mode": cfg.nce_pair_mode if cfg.run_infonce else "",
                "use_nce_projection": cfg.use_nce_projection if cfg.run_infonce else "",
                "nce_proj_dim": cfg.nce_proj_dim if cfg.run_infonce else "",
            }
        )
    concat_aware_df = pd.DataFrame(concat_aware_rows)
    concat_aware_df.to_csv(
        run_dir / "concat_aware_motivation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    rows.extend(rows_for_method("Concat", concat_grouped))
    rows.extend(rows_for_method("UncondAlign", align_grouped, lambda_align=best_lambda))
    direct_add_rows = rows_for_method(
        direct_add_method_name(cfg.direct_add_pair_mode),
        direct_add_grouped,
    )
    for row in direct_add_rows:
        row["direct_add_alpha"] = best_direct_add_alpha
        row["direct_add_pair_mode"] = cfg.direct_add_pair_mode
    rows.extend(direct_add_rows)
    balanced_direct_add_rows = rows_for_method(
        "BalancedDirectAdd",
        balanced_direct_add_grouped,
    )
    for row in balanced_direct_add_rows:
        row["direct_add_alpha"] = best_balanced_direct_add_alpha
        row["direct_add_pair_mode"] = "balanced"
    rows.extend(balanced_direct_add_rows)
    if cfg.run_infonce and infonce_grouped is not None:
        infonce_rows = rows_for_method("UncondInfoNCE", infonce_grouped)
        for row in infonce_rows:
            row["lambda_nce"] = best_lambda_nce
            row["nce_temperature"] = cfg.nce_temperature
            row["nce_pair_mode"] = cfg.nce_pair_mode
            row["use_nce_projection"] = cfg.use_nce_projection
            row["nce_proj_dim"] = cfg.nce_proj_dim
        rows.extend(infonce_rows)
    for row in rows:
        if row.get("method") == "UncondAlign":
            row["align_pair_mode"] = cfg.align_pair_mode
    results_df = pd.DataFrame(rows)
    results_df.to_csv(run_dir / "group_metrics.csv", index=False, encoding="utf-8-sig")

    delta_rows = []
    for group in ("Low-D", "Mid-D", "High-D", "Overall"):
        delta_rows.append(
            {
                "group": group,
                "delta_acc": align_grouped[group]["acc"] - concat_grouped[group]["acc"],
                "delta_macro_f1": align_grouped[group]["macro_f1"]
                - concat_grouped[group]["macro_f1"],
                "lambda_align": best_lambda,
                "align_pair_mode": cfg.align_pair_mode,
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(run_dir / "delta_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sweep_rows).to_csv(
        run_dir / "lambda_sweep_valid.csv", index=False, encoding="utf-8-sig"
    )
    lambda_delta_df = pd.DataFrame(lambda_delta_rows)
    lambda_delta_df.to_csv(
        run_dir / "lambda_test_delta_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_reliability_delta_df = pd.DataFrame(lambda_reliability_delta_rows)
    lambda_reliability_delta_df.to_csv(
        run_dir / "lambda_high_d_reliability_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_delta_plot(delta_df, run_dir / "delta_macro_f1.png")
    save_lambda_curve_plot(
        lambda_delta_df,
        run_dir / "lambda_delta_macro_f1_curve.png",
    )

    reliability_rows = []
    reliability_rows.extend(
        rows_for_method(
            "Concat",
            concat_reliability_grouped,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
    )
    reliability_rows.extend(
        rows_for_method(
            "UncondAlign",
            align_reliability_grouped,
            lambda_align=best_lambda,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
    )
    if cfg.run_infonce and infonce_reliability_grouped is not None:
        infonce_reliability_rows = rows_for_method(
            "UncondInfoNCE",
            infonce_reliability_grouped,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
        for row in infonce_reliability_rows:
            row["lambda_nce"] = best_lambda_nce
            row["nce_temperature"] = cfg.nce_temperature
            row["nce_pair_mode"] = cfg.nce_pair_mode
            row["use_nce_projection"] = cfg.use_nce_projection
            row["nce_proj_dim"] = cfg.nce_proj_dim
        reliability_rows.extend(infonce_reliability_rows)
    for row in reliability_rows:
        if row.get("method") == "UncondAlign":
            row["align_pair_mode"] = cfg.align_pair_mode
    reliability_results_df = pd.DataFrame(reliability_rows)
    reliability_results_df.to_csv(
        run_dir / "high_d_reliability_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability_delta_rows = []
    for group in HIGH_D_RELIABILITY_GROUP_ORDER:
        reliability_delta_rows.append(
            {
                "group": group,
                "delta_acc": align_reliability_grouped[group]["acc"]
                - concat_reliability_grouped[group]["acc"],
                "delta_macro_f1": align_reliability_grouped[group]["macro_f1"]
                - concat_reliability_grouped[group]["macro_f1"],
                "lambda_align": best_lambda,
                "align_pair_mode": cfg.align_pair_mode,
            }
        )
    reliability_delta_df = pd.DataFrame(reliability_delta_rows)
    reliability_delta_df.to_csv(
        run_dir / "high_d_reliability_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(diagnostic_model.state_dict(), run_dir / "diagnostic_model.pt")
    torch.save(concat_model.state_dict(), run_dir / "concat_model.pt")
    torch.save(align_model.state_dict(), run_dir / "uncond_align_model.pt")
    torch.save(direct_add_model.state_dict(), run_dir / "direct_add_model.pt")
    torch.save(
        balanced_direct_add_model.state_dict(),
        run_dir / "balanced_direct_add_model.pt",
    )
    if cfg.run_infonce and infonce_model is not None:
        torch.save(infonce_model.state_dict(), run_dir / "infonce_model.pt")
    save_json(
        run_dir / "summary.json",
        {
            "diagnostic_valid": diagnostic_valid,
            "concat_valid": concat_valid,
            "uncond_align_valid": align_valid,
            "best_lambda_align": best_lambda,
            "label_mode": cfg.label_mode,
            "num_classes": cfg.num_classes,
            "pair_mode": cfg.pair_mode,
            "align_pair_mode": cfg.align_pair_mode,
            "direct_add_valid": direct_add_valid,
            "best_direct_add_alpha": best_direct_add_alpha,
            "direct_add_pair_mode": cfg.direct_add_pair_mode,
            "balanced_direct_add_valid": balanced_direct_add_valid,
            "best_balanced_direct_add_alpha": best_balanced_direct_add_alpha,
            "infonce_valid": infonce_valid,
            "best_lambda_nce": best_lambda_nce,
            "nce_temperature": cfg.nce_temperature,
            "nce_pair_mode": cfg.nce_pair_mode,
            "use_nce_projection": cfg.use_nce_projection,
            "nce_proj_dim": cfg.nce_proj_dim,
            "disagreement_metric": cfg.disagreement_metric,
            "disagreement_pair_mode": cfg.disagreement_pair_mode,
            "kernel_bandwidth": cfg.kernel_bandwidth,
            "kernel_pair_mode": cfg.kernel_pair_mode,
            "kernel_class_weight": cfg.kernel_class_weight,
            "kernel_max_class_samples": cfg.kernel_max_class_samples,
            "relation_split": cfg.relation_split,
            "residual_modes": cfg.residual_modes,
            "resolved_kernel_bandwidth": resolved_kernel_bandwidth
            if cfg.disagreement_metric == "kernel_mmd"
            else None,
            "thresholds": {
                "q33": q33,
                "q66": q66,
                "q_r": q_r,
                **{f"r_{key}": value for key, value in relation_thresholds.items()},
            },
            "high_d_reliability_counts": {
                group: int((test_reliability_groups == group).sum())
                for group in HIGH_D_RELIABILITY_GROUP_ORDER
            },
            "relation_state_counts": {
                group: int((test_relation_states == group).sum())
                for group in RELATION_STATE_GROUP_ORDER
            },
        },
    )

    print("\nGroup metrics:")
    print(results_df.to_string(index=False))
    print("\nDelta metrics:")
    print(delta_df.to_string(index=False))
    print("\nDirectAdd delta metrics:")
    print(direct_add_delta_df.to_string(index=False))
    print("\nBalancedDirectAdd delta metrics:")
    print(balanced_direct_add_delta_df.to_string(index=False))
    if cfg.run_infonce:
        print("\nInfoNCE delta metrics:")
        print(infonce_delta_df.to_string(index=False))
        print("\nInfoNCE High-D reliability delta metrics:")
        print(infonce_reliability_delta_df.to_string(index=False))
    print("\nHigh-D reliability delta metrics:")
    print(reliability_delta_df.to_string(index=False))
    print("\nRelation-state delta metrics:")
    print(relation_state_delta_df.to_string(index=False))
    print("\nRelation-state distribution/calibration:")
    print(calibration_df.to_string(index=False))
    print("\nDirectAdd relation-state delta metrics:")
    print(direct_add_relation_state_delta_df.to_string(index=False))
    print("\nBalancedDirectAdd relation-state delta metrics:")
    print(balanced_direct_add_relation_state_delta_df.to_string(index=False))
    print("\nConcat-aware motivation table:")
    print(concat_aware_df.to_string(index=False))
    print(f"\nSaved outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
