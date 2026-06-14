from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, ExperimentConfig  # noqa: E402
from src.data import MultimodalSplitDataset, infer_input_dims, load_npz_splits  # noqa: E402
from src.disagreement import (  # noqa: E402
    build_group_frame,
    grouped_metrics,
    rows_for_method,
    sample_disagreement,
    validation_thresholds,
    assign_groups,
)
from src.model import MultimodalClassifier  # noqa: E402
from src.plotting import save_delta_plot  # noqa: E402
from src.train import predict, train_model  # noqa: E402
from src.utils import choose_device, ensure_dir, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CMU-MOSI/MOSEI disagreement phenomenon experiment."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\Xu\data"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT / "outputs",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
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
        default=[0.01, 0.05, 0.1, 0.5],
    )
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def make_loaders(
    splits: dict[str, object],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    datasets = {name: MultimodalSplitDataset(split) for name, split in splits.items()}
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def new_model(input_dims: dict[str, int], cfg: ExperimentConfig) -> MultimodalClassifier:
    return MultimodalClassifier(
        text_dim=input_dims["text"],
        vision_dim=input_dims["vision"],
        audio_dim=input_dims["audio"],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
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
    )


def train_best_alignment(
    input_dims: dict[str, int],
    loaders: dict[str, DataLoader],
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[MultimodalClassifier, dict[str, float], float, list[dict[str, float]]]:
    best_model: MultimodalClassifier | None = None
    best_metrics: dict[str, float] = {}
    best_lambda = cfg.lambda_align_values[0]
    best_score = -1.0
    sweep_rows: list[dict[str, float]] = []

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
        )
        score = metrics.get("macro_f1", -1.0)
        sweep_rows.append({"lambda_align": lambda_align, **metrics})
        if score > best_score:
            best_score = score
            best_model = trained
            best_metrics = metrics
            best_lambda = lambda_align

    if best_model is None:
        raise RuntimeError("Alignment sweep did not train any model.")
    return best_model, best_metrics, best_lambda, sweep_rows


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        eta_unimodal=args.eta_unimodal,
        lambda_align_values=args.lambda_align_values,
        patience=args.patience,
    )

    set_seed(cfg.seed)
    device = choose_device()

    print(f"Dataset: {cfg.dataset}")
    print(f"Data file: {cfg.data_path}")
    print(f"Device: {device}")

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
    loaders = make_loaders(splits, cfg.batch_size, cfg.num_workers)

    save_json(
        run_dir / "config.json",
        {
            "dataset": cfg.dataset,
            "data_path": str(cfg.data_path),
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "eta_unimodal": cfg.eta_unimodal,
            "lambda_align_values": cfg.lambda_align_values,
            "input_dims": input_dims,
        },
    )

    diagnostic_model, diagnostic_valid = train_diagnostic(input_dims, loaders, cfg, device)
    valid_diag = predict(diagnostic_model, loaders["valid"], device)
    test_diag = predict(diagnostic_model, loaders["test"], device)
    valid_d = sample_disagreement(
        valid_diag["prob_t"], valid_diag["prob_v"], valid_diag["prob_a"]
    )
    test_d = sample_disagreement(
        test_diag["prob_t"], test_diag["prob_v"], test_diag["prob_a"]
    )
    q33, q66 = validation_thresholds(valid_d)
    test_groups = assign_groups(test_d, q33, q66)
    group_df = build_group_frame(test_diag, test_d, test_groups)
    group_df.to_csv(run_dir / "test_groups.csv", index=False, encoding="utf-8-sig")

    concat_model, concat_valid = train_concat(input_dims, loaders, cfg, device)
    align_model, align_valid, best_lambda, sweep_rows = train_best_alignment(
        input_dims, loaders, cfg, device
    )

    concat_pred = predict(concat_model, loaders["test"], device)
    align_pred = predict(align_model, loaders["test"], device)

    concat_grouped = grouped_metrics(concat_pred["y_true"], concat_pred["y_pred"], test_groups)
    align_grouped = grouped_metrics(align_pred["y_true"], align_pred["y_pred"], test_groups)

    rows = []
    rows.extend(rows_for_method("Concat", concat_grouped))
    rows.extend(rows_for_method("UncondAlign", align_grouped, lambda_align=best_lambda))
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
    save_delta_plot(delta_df, run_dir / "delta_macro_f1.png")

    torch.save(diagnostic_model.state_dict(), run_dir / "diagnostic_model.pt")
    torch.save(concat_model.state_dict(), run_dir / "concat_model.pt")
    torch.save(align_model.state_dict(), run_dir / "uncond_align_model.pt")
    save_json(
        run_dir / "summary.json",
        {
            "diagnostic_valid": diagnostic_valid,
            "concat_valid": concat_valid,
            "uncond_align_valid": align_valid,
            "best_lambda_align": best_lambda,
            "thresholds": {"q33": q33, "q66": q66},
        },
    )

    print("\nGroup metrics:")
    print(results_df.to_string(index=False))
    print("\nDelta metrics:")
    print(delta_df.to_string(index=False))
    print(f"\nSaved outputs to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
