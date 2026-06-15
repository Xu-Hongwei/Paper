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
    build_group_frame,
    build_label_aware_relation_frame,
    grouped_metrics,
    pairwise_disagreement,
    reliability_threshold,
    relation_gates,
    rows_for_method,
    sample_disagreement,
    sample_reliability,
    summarize_relation_frame,
    validation_thresholds,
    assign_groups,
)
from src.model import MultimodalClassifier  # noqa: E402
from src.plotting import save_delta_plot, save_lambda_curve_plot  # noqa: E402
from src.train import predict, train_model  # noqa: E402
from src.utils import choose_device, ensure_dir, save_json, set_seed  # noqa: E402
from src.v4_analysis import (  # noqa: E402
    class_means,
    feature_consistency_frame,
    feature_disagreement,
    residual_diagnostic_frame,
    residual_probe_frame,
    selective_agreement_prototype_frame,
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
    parser.add_argument("--eta_unimodal", type=float, default=0.3)
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
        help="Alpha sweep for DirectAdd: h_m + alpha * mean(h_t,h_v,h_a).",
    )
    parser.add_argument(
        "--run_copa",
        action="store_true",
        help="Train the label-aware CoPA prototype model in addition to baselines.",
    )
    parser.add_argument(
        "--lambda_copa_values",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
    )
    parser.add_argument(
        "--tau_agreement",
        type=float,
        default=0.1,
        help="Temperature for A_ij = exp(-JSD/tau_agreement).",
    )
    parser.add_argument("--copa_proto_weight", type=float, default=1.0)
    parser.add_argument("--copa_agr_weight", type=float, default=1.0)
    parser.add_argument("--copa_comp_weight", type=float, default=0.5)
    parser.add_argument("--copa_comp_margin", type=float, default=0.2)
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
    return parser.parse_args()


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


def new_model(
    input_dims: dict[str, int],
    cfg: ExperimentConfig,
    *,
    direct_add_alpha: float = 0.0,
) -> MultimodalClassifier:
    return MultimodalClassifier(
        text_dim=input_dims["text"],
        vision_dim=input_dims["vision"],
        audio_dim=input_dims["audio"],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        direct_add_alpha=direct_add_alpha,
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
            patience=cfg.patience,
            desc=f"Align λ={lambda_align}",
            show_progress=not cfg.quiet,
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append({"lambda_align": lambda_align, **metrics})

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

    for alpha in cfg.direct_add_alpha_values:
        model = new_model(input_dims, cfg, direct_add_alpha=alpha)
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
            desc=f"DirectAdd α={alpha}",
            show_progress=not cfg.quiet,
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append({"direct_add_alpha": alpha, **metrics})

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


def train_best_copa(
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
    best_lambda = cfg.lambda_copa_values[0]
    best_score = -1.0
    sweep_rows: list[dict[str, float]] = []
    lambda_delta_rows: list[dict[str, float]] = []
    lambda_reliability_delta_rows: list[dict[str, float]] = []

    for lambda_copa in cfg.lambda_copa_values:
        model = new_model(input_dims, cfg)
        trained, metrics = train_model(
            model,
            loaders["train"],
            loaders["valid"],
            device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            eta_unimodal=cfg.eta_unimodal,
            lambda_align=0.0,
            lambda_copa=lambda_copa,
            tau_agreement=cfg.tau_agreement,
            copa_proto_weight=cfg.copa_proto_weight,
            copa_agr_weight=cfg.copa_agr_weight,
            copa_comp_weight=cfg.copa_comp_weight,
            copa_comp_margin=cfg.copa_comp_margin,
            patience=cfg.patience,
            desc=f"CoPA λ={lambda_copa}",
            show_progress=not cfg.quiet,
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append({"lambda_copa": lambda_copa, **metrics})

        copa_pred = predict(trained, loaders["test"], device)
        copa_grouped = grouped_metrics(
            copa_pred["y_true"],
            copa_pred["y_pred"],
            test_groups,
        )
        copa_reliability_grouped = grouped_metrics(
            copa_pred["y_true"],
            copa_pred["y_pred"],
            test_reliability_groups,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
        for group in ("Low-D", "Mid-D", "High-D", "Overall"):
            lambda_delta_rows.append(
                {
                    "lambda_copa": lambda_copa,
                    "group": group,
                    "n": copa_grouped[group]["n"],
                    "concat_acc": concat_grouped[group]["acc"],
                    "copa_acc": copa_grouped[group]["acc"],
                    "delta_acc": copa_grouped[group]["acc"] - concat_grouped[group]["acc"],
                    "concat_macro_f1": concat_grouped[group]["macro_f1"],
                    "copa_macro_f1": copa_grouped[group]["macro_f1"],
                    "delta_macro_f1": copa_grouped[group]["macro_f1"]
                    - concat_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                }
            )
        for group in HIGH_D_RELIABILITY_GROUP_ORDER:
            lambda_reliability_delta_rows.append(
                {
                    "lambda_copa": lambda_copa,
                    "group": group,
                    "n": copa_reliability_grouped[group]["n"],
                    "concat_acc": concat_reliability_grouped[group]["acc"],
                    "copa_acc": copa_reliability_grouped[group]["acc"],
                    "delta_acc": copa_reliability_grouped[group]["acc"]
                    - concat_reliability_grouped[group]["acc"],
                    "concat_macro_f1": concat_reliability_grouped[group]["macro_f1"],
                    "copa_macro_f1": copa_reliability_grouped[group]["macro_f1"],
                    "delta_macro_f1": copa_reliability_grouped[group]["macro_f1"]
                    - concat_reliability_grouped[group]["macro_f1"],
                    "valid_macro_f1": metrics.get("macro_f1"),
                    "valid_acc": metrics.get("acc"),
                }
            )
        if score > best_score:
            best_score = score
            best_model = trained
            best_metrics = metrics
            best_grouped = copa_grouped
            best_reliability_grouped = copa_reliability_grouped
            best_lambda = lambda_copa

    if best_model is None:
        raise RuntimeError("CoPA sweep did not train any model.")
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
        lambda_align_values=args.lambda_align_values,
        direct_add_alpha_values=args.direct_add_alpha_values,
        run_copa=args.run_copa,
        lambda_copa_values=args.lambda_copa_values,
        tau_agreement=args.tau_agreement,
        copa_proto_weight=args.copa_proto_weight,
        copa_agr_weight=args.copa_agr_weight,
        copa_comp_weight=args.copa_comp_weight,
        copa_comp_margin=args.copa_comp_margin,
        patience=args.patience,
        deterministic=args.deterministic,
        quiet=args.quiet,
    )

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = choose_device()

    print(f"Dataset: {cfg.dataset}")
    print(f"Data file: {cfg.data_path}")
    print(f"Device: {device}")
    print(f"Deterministic: {cfg.deterministic}")

    try:
        splits = load_npz_splits(cfg.data_path)
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
            "lambda_align_values": cfg.lambda_align_values,
            "direct_add_alpha_values": cfg.direct_add_alpha_values,
            "run_copa": cfg.run_copa,
            "lambda_copa_values": cfg.lambda_copa_values,
            "tau_agreement": cfg.tau_agreement,
            "copa_proto_weight": cfg.copa_proto_weight,
            "copa_agr_weight": cfg.copa_agr_weight,
            "copa_comp_weight": cfg.copa_comp_weight,
            "copa_comp_margin": cfg.copa_comp_margin,
            "deterministic": cfg.deterministic,
            "input_dims": input_dims,
            "quiet": cfg.quiet,
        },
    )

    diagnostic_model, diagnostic_valid = train_diagnostic(input_dims, loaders, cfg, device)
    train_diag = predict(diagnostic_model, loaders["train"], device)
    valid_diag = predict(diagnostic_model, loaders["valid"], device)
    test_diag = predict(diagnostic_model, loaders["test"], device)
    train_label_aware = build_label_aware_relation_frame(
        train_diag,
        tau_agreement=cfg.tau_agreement,
    )
    valid_label_aware = build_label_aware_relation_frame(
        valid_diag,
        tau_agreement=cfg.tau_agreement,
    )
    train_label_aware.to_csv(
        run_dir / "train_label_aware_relations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_label_aware.to_csv(
        run_dir / "valid_label_aware_relations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            summarize_relation_frame(train_label_aware, "train"),
            summarize_relation_frame(valid_label_aware, "valid"),
        ]
    ).to_csv(
        run_dir / "label_aware_relation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    train_d = sample_disagreement(
        train_diag["prob_t"], train_diag["prob_v"], train_diag["prob_a"]
    )
    valid_d = sample_disagreement(
        valid_diag["prob_t"], valid_diag["prob_v"], valid_diag["prob_a"]
    )
    test_d = sample_disagreement(
        test_diag["prob_t"], test_diag["prob_v"], test_diag["prob_a"]
    )
    q33, q66 = validation_thresholds(valid_d)
    train_groups = assign_groups(train_d, q33, q66)
    test_groups = assign_groups(test_d, q33, q66)
    train_reliability = sample_reliability(
        train_diag["prob_t"], train_diag["prob_v"], train_diag["prob_a"]
    )
    valid_reliability = sample_reliability(
        valid_diag["prob_t"], valid_diag["prob_v"], valid_diag["prob_a"]
    )
    test_reliability = sample_reliability(
        test_diag["prob_t"], test_diag["prob_v"], test_diag["prob_a"]
    )
    q_r = reliability_threshold(valid_reliability["R_sample"])
    train_relation_states = assign_relation_state_groups(
        train_groups,
        train_reliability["R_sample"],
        q_r,
    )
    test_relation_states = assign_relation_state_groups(
        test_groups,
        test_reliability["R_sample"],
        q_r,
    )
    test_relations = {
        **pairwise_disagreement(test_diag["prob_t"], test_diag["prob_v"], test_diag["prob_a"]),
        **relation_gates(
            test_diag["prob_t"],
            test_diag["prob_v"],
            test_diag["prob_a"],
            test_reliability,
            cfg.tau_agreement,
            prefix="R_",
        ),
    }
    test_reliability_groups = assign_high_d_reliability_groups(
        test_groups,
        test_reliability["R_sample"],
        q_r,
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
    group_df.to_csv(run_dir / "test_groups.csv", index=False, encoding="utf-8-sig")

    test_d_feat = feature_disagreement(test_diag)
    means = class_means(train_diag, num_classes=train_diag["prob_f"].shape[1])
    feature_consistency_frame(test_d, test_d_feat, test_relation_states).to_csv(
        run_dir / "feature_consistency_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_diagnostic_frame(
        test_diag,
        test_d,
        test_d_feat,
        test_relation_states,
        means,
    ).to_csv(
        run_dir / "residual_distribution_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_probe_df = residual_probe_frame(
        train_diag,
        test_diag,
        train_relation_states,
        test_relation_states,
        means,
        cfg.seed,
    )
    residual_probe_df.to_csv(
        run_dir / "residual_discriminative_probe.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selective_agreement_prototype_frame(
        train_diag,
        test_diag,
        train_groups,
        train_relation_states,
        test_groups,
    ).to_csv(
        run_dir / "selective_agreement_prototype_check.csv",
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
            }
        )
    direct_add_delta_df = pd.DataFrame(direct_add_delta_rows)
    direct_add_delta_df.to_csv(
        run_dir / "direct_add_delta_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    copa_model = None
    copa_valid = None
    best_lambda_copa = None
    copa_grouped = None
    copa_reliability_grouped = None
    copa_relation_grouped = None
    copa_delta_df = pd.DataFrame()
    copa_reliability_delta_df = pd.DataFrame()
    if cfg.run_copa:
        (
            copa_model,
            copa_valid,
            best_lambda_copa,
            copa_sweep_rows,
            copa_grouped,
            copa_reliability_grouped,
            copa_lambda_delta_rows,
            copa_lambda_reliability_delta_rows,
        ) = train_best_copa(
            input_dims,
            loaders,
            cfg,
            device,
            concat_grouped,
            concat_reliability_grouped,
            test_groups,
            test_reliability_groups,
        )
        copa_pred = predict(copa_model, loaders["test"], device)
        copa_relation_grouped = grouped_metrics(
            copa_pred["y_true"],
            copa_pred["y_pred"],
            test_relation_states,
            group_order=RELATION_STATE_GROUP_ORDER,
            include_overall=False,
        )
        pd.DataFrame(copa_sweep_rows).to_csv(
            run_dir / "copa_lambda_sweep_valid.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(copa_lambda_delta_rows).to_csv(
            run_dir / "copa_lambda_test_delta_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(copa_lambda_reliability_delta_rows).to_csv(
            run_dir / "copa_lambda_high_d_reliability_delta.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_delta_rows = []
        for group in ("Low-D", "Mid-D", "High-D", "Overall"):
            copa_delta_rows.append(
                {
                    "group": group,
                    "delta_acc": copa_grouped[group]["acc"] - concat_grouped[group]["acc"],
                    "delta_macro_f1": copa_grouped[group]["macro_f1"]
                    - concat_grouped[group]["macro_f1"],
                    "lambda_copa": best_lambda_copa,
                }
            )
        copa_delta_df = pd.DataFrame(copa_delta_rows)
        copa_delta_df.to_csv(
            run_dir / "copa_delta_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_reliability_delta_rows = []
        for group in HIGH_D_RELIABILITY_GROUP_ORDER:
            copa_reliability_delta_rows.append(
                {
                    "group": group,
                    "delta_acc": copa_reliability_grouped[group]["acc"]
                    - concat_reliability_grouped[group]["acc"],
                    "delta_macro_f1": copa_reliability_grouped[group]["macro_f1"]
                    - concat_reliability_grouped[group]["macro_f1"],
                    "lambda_copa": best_lambda_copa,
                }
            )
        copa_reliability_delta_df = pd.DataFrame(copa_reliability_delta_rows)
        copa_reliability_delta_df.to_csv(
            run_dir / "copa_high_d_reliability_delta.csv",
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
        "DirectAdd",
        direct_add_relation_grouped,
        group_order=RELATION_STATE_GROUP_ORDER,
        include_overall=False,
    )
    for row in direct_add_relation_rows:
        row["direct_add_alpha"] = best_direct_add_alpha
    relation_state_rows.extend(direct_add_relation_rows)
    if cfg.run_copa and copa_relation_grouped is not None:
        copa_relation_rows = rows_for_method(
            "CoPA",
            copa_relation_grouped,
            group_order=RELATION_STATE_GROUP_ORDER,
            include_overall=False,
        )
        for row in copa_relation_rows:
            row["lambda_copa"] = best_lambda_copa
        relation_state_rows.extend(copa_relation_rows)
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
            }
        )
    relation_state_delta_df = pd.DataFrame(relation_state_delta_rows)
    relation_state_delta_df.to_csv(
        run_dir / "relation_state_delta.csv",
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
            }
        )
    direct_add_relation_state_delta_df = pd.DataFrame(direct_add_relation_state_delta_rows)
    direct_add_relation_state_delta_df.to_csv(
        run_dir / "direct_add_relation_state_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if cfg.run_copa and copa_relation_grouped is not None:
        copa_relation_state_delta_rows = []
        for group in RELATION_STATE_GROUP_ORDER:
            copa_relation_state_delta_rows.append(
                {
                    "group": group,
                    "delta_acc": copa_relation_grouped[group]["acc"]
                    - concat_relation_grouped[group]["acc"],
                    "delta_macro_f1": copa_relation_grouped[group]["macro_f1"]
                    - concat_relation_grouped[group]["macro_f1"],
                    "lambda_copa": best_lambda_copa,
                }
            )
        pd.DataFrame(copa_relation_state_delta_rows).to_csv(
            run_dir / "copa_relation_state_delta.csv",
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
                "soft_split_probe_macro_f1": probe_row.get(
                    "common_residual_macro_f1",
                    float("nan"),
                ),
                "residual_gain_macro_f1": probe_row.get(
                    "residual_gain_macro_f1",
                    float("nan"),
                ),
                "shuffled_residual_only_macro_f1": probe_row.get(
                    "shuffled_residual_only_macro_f1",
                    float("nan"),
                ),
                "lambda_align": best_lambda,
                "direct_add_alpha": best_direct_add_alpha,
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
    direct_add_rows = rows_for_method("DirectAdd", direct_add_grouped)
    for row in direct_add_rows:
        row["direct_add_alpha"] = best_direct_add_alpha
    rows.extend(direct_add_rows)
    if cfg.run_copa and copa_grouped is not None:
        copa_rows = rows_for_method("CoPA", copa_grouped)
        for row in copa_rows:
            row["lambda_copa"] = best_lambda_copa
        rows.extend(copa_rows)
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
    if cfg.run_copa and copa_reliability_grouped is not None:
        copa_reliability_rows = rows_for_method(
            "CoPA",
            copa_reliability_grouped,
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            include_overall=False,
        )
        for row in copa_reliability_rows:
            row["lambda_copa"] = best_lambda_copa
        reliability_rows.extend(copa_reliability_rows)
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
    if cfg.run_copa and copa_model is not None:
        torch.save(copa_model.state_dict(), run_dir / "copa_model.pt")
    save_json(
        run_dir / "summary.json",
        {
            "diagnostic_valid": diagnostic_valid,
            "concat_valid": concat_valid,
            "uncond_align_valid": align_valid,
            "best_lambda_align": best_lambda,
            "direct_add_valid": direct_add_valid,
            "best_direct_add_alpha": best_direct_add_alpha,
            "copa_valid": copa_valid,
            "best_lambda_copa": best_lambda_copa,
            "thresholds": {"q33": q33, "q66": q66, "q_r": q_r},
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
    print("\nHigh-D reliability delta metrics:")
    print(reliability_delta_df.to_string(index=False))
    print("\nRelation-state delta metrics:")
    print(relation_state_delta_df.to_string(index=False))
    print("\nDirectAdd relation-state delta metrics:")
    print(direct_add_relation_state_delta_df.to_string(index=False))
    print("\nConcat-aware motivation table:")
    print(concat_aware_df.to_string(index=False))
    if cfg.run_copa:
        print("\nCoPA delta metrics:")
        print(copa_delta_df.to_string(index=False))
        print("\nCoPA High-D reliability delta metrics:")
        print(copa_reliability_delta_df.to_string(index=False))
    print(f"\nSaved outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
