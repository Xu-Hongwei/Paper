from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS  # noqa: E402
from src.plotting import (  # noqa: E402
    save_lambda_curve_plot,
    save_multi_seed_delta_plot,
    save_reliability_delta_plot,
)
from src.utils import ensure_dir, save_json  # noqa: E402

RUNNER = ROOT / "scripts" / "run_phenomenon.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run disagreement phenomenon experiment across multiple seeds."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data_root", type=Path, default=Path(r"E:\Xu\data\MultiBench"))
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
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
        default=[0.01, 0.05, 0.1, 0.5],
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to each single-seed run.",
    )
    return parser.parse_args()


def newest_run_dir(run_root: Path, dataset: str, seen: set[Path]) -> Path:
    candidates = [
        path
        for path in (run_root / dataset).glob("*")
        if path.is_dir() and path not in seen
    ]
    if not candidates:
        raise RuntimeError(f"No new run directory found under {run_root / dataset}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def flatten_summary(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    summary = frame.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{value}_{stat}" for value, stat in summary.columns]
    return summary.reset_index()


def read_seed_csv(run_dirs: list[tuple[int, Path]], filename: str) -> pd.DataFrame:
    frames = []
    for seed, run_dir in run_dirs:
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing expected output: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "seed", seed)
        frame.insert(1, "run_dir", str(run_dir))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_conclusion(delta_summary: pd.DataFrame, path: Path) -> None:
    by_group = {row["group"]: row for _, row in delta_summary.iterrows()}
    high = by_group.get("High-D")
    low = by_group.get("Low-D")
    mid = by_group.get("Mid-D")
    payload = {
        "high_d_delta_macro_f1_mean": None if high is None else high["delta_macro_f1_mean"],
        "low_d_delta_macro_f1_mean": None if low is None else low["delta_macro_f1_mean"],
        "mid_d_delta_macro_f1_mean": None if mid is None else mid["delta_macro_f1_mean"],
        "high_d_is_negative": False if high is None else high["delta_macro_f1_mean"] < 0,
        "high_d_more_harmful_than_low_d": False
        if high is None or low is None
        else high["delta_macro_f1_mean"] < low["delta_macro_f1_mean"],
        "high_d_more_harmful_than_mid_d": False
        if high is None or mid is None
        else high["delta_macro_f1_mean"] < mid["delta_macro_f1_mean"],
    }
    save_json(path, payload)


def run_one_seed(args: argparse.Namespace, run_root: Path, seed: int, seen: set[Path]) -> Path:
    command = [
        sys.executable,
        "-B",
        str(RUNNER),
        "--dataset",
        args.dataset,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(run_root),
        "--seed",
        str(seed),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--hidden_dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--eta_unimodal",
        str(args.eta_unimodal),
        "--patience",
        str(args.patience),
        "--lambda_align_values",
        *[str(value) for value in args.lambda_align_values],
    ]
    if args.quiet:
        command.append("--quiet")
    print(f"\n=== Running seed {seed} ===")
    subprocess.run(command, check=True)
    run_dir = newest_run_dir(run_root, args.dataset, seen)
    seen.add(run_dir)
    print(f"Seed {seed} output: {run_dir}")
    return run_dir


def main() -> int:
    args = parse_args()
    multi_dir = ensure_dir(
        args.output_root
        / args.dataset
        / f"multi_seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_root = ensure_dir(multi_dir / "runs")
    summary_dir = ensure_dir(multi_dir / "summary")
    save_json(
        summary_dir / "multi_seed_config.json",
        {
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "eta_unimodal": args.eta_unimodal,
            "lambda_align_values": args.lambda_align_values,
            "patience": args.patience,
            "quiet": args.quiet,
        },
    )

    run_dirs: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for seed in args.seeds:
        run_dirs.append((seed, run_one_seed(args, run_root, seed, seen)))

    delta_all = read_seed_csv(run_dirs, "delta_metrics.csv")
    group_all = read_seed_csv(run_dirs, "group_metrics.csv")
    reliability_delta_all = read_seed_csv(run_dirs, "high_d_reliability_delta.csv")
    reliability_metrics_all = read_seed_csv(run_dirs, "high_d_reliability_metrics.csv")
    lambda_delta_all = read_seed_csv(run_dirs, "lambda_test_delta_metrics.csv")
    lambda_reliability_delta_all = read_seed_csv(
        run_dirs,
        "lambda_high_d_reliability_delta.csv",
    )

    delta_all.to_csv(summary_dir / "multi_seed_delta_all.csv", index=False, encoding="utf-8-sig")
    group_all.to_csv(
        summary_dir / "multi_seed_group_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability_delta_all.to_csv(
        summary_dir / "high_d_reliability_delta_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability_metrics_all.to_csv(
        summary_dir / "high_d_reliability_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_delta_all.to_csv(
        summary_dir / "lambda_test_delta_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_reliability_delta_all.to_csv(
        summary_dir / "lambda_high_d_reliability_delta_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_summary = flatten_summary(
        delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "lambda_align"],
    )
    group_summary = flatten_summary(
        group_all,
        ["method", "group"],
        ["acc", "macro_f1", "n"],
    )
    reliability_summary = flatten_summary(
        reliability_delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "lambda_align"],
    )
    lambda_delta_summary = flatten_summary(
        lambda_delta_all,
        ["lambda_align", "group"],
        ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
    )
    lambda_reliability_summary = flatten_summary(
        lambda_reliability_delta_all,
        ["lambda_align", "group"],
        ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
    )

    delta_summary.to_csv(
        summary_dir / "multi_seed_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_summary.to_csv(
        summary_dir / "multi_seed_group_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability_summary.to_csv(
        summary_dir / "high_d_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_delta_summary.to_csv(
        summary_dir / "lambda_test_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lambda_reliability_summary.to_csv(
        summary_dir / "lambda_high_d_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_multi_seed_delta_plot(delta_summary, summary_dir / "multi_seed_delta_macro_f1.png")
    save_reliability_delta_plot(
        reliability_summary,
        summary_dir / "high_d_reliability_delta.png",
    )
    save_lambda_curve_plot(
        lambda_delta_summary,
        summary_dir / "lambda_delta_macro_f1_curve.png",
        title="Multi-seed lambda alignment strength curve",
    )
    write_conclusion(delta_summary, summary_dir / "experiment_one_conclusion.json")

    print("\nMulti-seed delta summary:")
    print(delta_summary.to_string(index=False))
    print("\nHigh-D reliability summary:")
    print(reliability_summary.to_string(index=False))
    print("\nLambda strength delta summary:")
    print(lambda_delta_summary.to_string(index=False))
    print(f"\nSaved multi-seed summary to: {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
