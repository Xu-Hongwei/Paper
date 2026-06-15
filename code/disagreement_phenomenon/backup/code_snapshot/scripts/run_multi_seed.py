from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS  # noqa: E402
from src.disagreement import HIGH_D_RELIABILITY_GROUP_ORDER, RELATION_STATE_GROUP_ORDER  # noqa: E402
from src.plotting import (  # noqa: E402
    save_detailed_delta_plot,
    save_lambda_curve_plot,
    save_method_relation_state_heatmap,
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
        default=[0.001, 0.005, 0.01, 0.05, 0.1],
    )
    parser.add_argument(
        "--direct_add_alpha_values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 1.0],
    )
    parser.add_argument(
        "--run_infonce",
        action="store_true",
        help="Train and aggregate unconditional InfoNCE baselines.",
    )
    parser.add_argument(
        "--lambda_nce_values",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.5],
    )
    parser.add_argument("--nce_temperature", type=float, default=0.1)
    parser.add_argument(
        "--nce_pair_mode",
        choices=("text_anchor", "full_pair"),
        default="text_anchor",
    )
    parser.add_argument(
        "--disagreement_metric",
        choices=("prob_jsd", "kernel_mmd"),
        default="prob_jsd",
    )
    parser.add_argument(
        "--disagreement_pair_mode",
        choices=("text_anchor", "full_pair"),
        default="text_anchor",
    )
    parser.add_argument("--kernel_bandwidth", default="median")
    parser.add_argument(
        "--kernel_pair_mode",
        choices=("text_anchor", "full_pair"),
        default="text_anchor",
    )
    parser.add_argument("--kernel_class_weight", type=float, default=0.5)
    parser.add_argument("--kernel_max_class_samples", type=int, default=1024)
    parser.add_argument(
        "--run_copa",
        action="store_true",
        help="Train and aggregate the label-aware CoPA prototype model.",
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
        help="Temperature for common/residual NCE and full_relation agreement gates.",
    )
    parser.add_argument("--copa_proto_weight", type=float, default=1.0)
    parser.add_argument("--copa_agr_weight", type=float, default=1.0)
    parser.add_argument("--copa_comp_weight", type=float, default=0.5)
    parser.add_argument("--copa_comp_margin", type=float, default=0.2)
    parser.add_argument("--copa_orth_weight", type=float, default=0.01)
    parser.add_argument(
        "--copa_gate_type",
        choices=("label_support", "full_relation"),
        default="label_support",
    )
    parser.add_argument(
        "--copa_gate_metric",
        choices=("prob_jsd", "kernel_mmd"),
        default="prob_jsd",
        help="Distance metric used by --copa_gate_type full_relation.",
    )
    parser.add_argument("--copa_kernel_bandwidth", default="median")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Pass deterministic seeding controls to each single-seed run.",
    )
    parser.add_argument(
        "--error_min_seeds",
        type=int,
        default=5,
        help="Minimum seed count required before a delta passes error control.",
    )
    parser.add_argument(
        "--error_sign_rate",
        type=float,
        default=0.8,
        help="Required same-sign seed ratio for delta error-control checks.",
    )
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


def t_critical_95(count: int) -> float:
    table = {
        1: math.nan,
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


def flatten_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    *,
    sign_cols: list[str] | None = None,
    min_count: int = 5,
    sign_rate_threshold: float = 0.8,
) -> pd.DataFrame:
    sign_cols = sign_cols or []
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_cols, keys))
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
            row[f"{value}_sem"] = sem
            row[f"{value}_ci95_low"] = mean - margin if not math.isnan(margin) else math.nan
            row[f"{value}_ci95_high"] = mean + margin if not math.isnan(margin) else math.nan
            if value in sign_cols:
                positive_rate = float((series > 0).mean()) if count else math.nan
                negative_rate = float((series < 0).mean()) if count else math.nan
                zero_rate = float((series == 0).mean()) if count else math.nan
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
                row[f"{value}_zero_rate"] = zero_rate
                row[f"{value}_sign_consistency"] = sign_consistency
                row[f"{value}_ci95_excludes_zero"] = ci_excludes_zero
                row[f"{value}_passes_error_control"] = bool(
                    count >= min_count
                    and sign_consistency >= sign_rate_threshold
                    and ci_excludes_zero
                )
        rows.append(row)
    return pd.DataFrame(rows)


def write_error_control_report(
    path: Path,
    frames: list[tuple[str, pd.DataFrame, list[str]]],
) -> None:
    rows: list[dict[str, object]] = []
    for source, frame, key_cols in frames:
        for _, record in frame.iterrows():
            key = ",".join(f"{column}={record[column]}" for column in key_cols)
            for metric in ("delta_acc", "delta_macro_f1"):
                pass_col = f"{metric}_passes_error_control"
                if pass_col not in frame.columns:
                    continue
                rows.append(
                    {
                        "source": source,
                        "key": key,
                        "metric": metric,
                        "mean": record.get(f"{metric}_mean"),
                        "std": record.get(f"{metric}_std"),
                        "sem": record.get(f"{metric}_sem"),
                        "ci95_low": record.get(f"{metric}_ci95_low"),
                        "ci95_high": record.get(f"{metric}_ci95_high"),
                        "count": record.get(f"{metric}_count"),
                        "positive_rate": record.get(f"{metric}_positive_rate"),
                        "negative_rate": record.get(f"{metric}_negative_rate"),
                        "sign_consistency": record.get(f"{metric}_sign_consistency"),
                        "ci95_excludes_zero": record.get(f"{metric}_ci95_excludes_zero"),
                        "passes_error_control": record.get(pass_col),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


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
        "--tau_agreement",
        str(args.tau_agreement),
        "--lambda_align_values",
        *[str(value) for value in args.lambda_align_values],
        "--direct_add_alpha_values",
        *[str(value) for value in args.direct_add_alpha_values],
        "--disagreement_metric",
        args.disagreement_metric,
        "--disagreement_pair_mode",
        args.disagreement_pair_mode,
        "--kernel_bandwidth",
        str(args.kernel_bandwidth),
        "--kernel_pair_mode",
        args.kernel_pair_mode,
        "--kernel_class_weight",
        str(args.kernel_class_weight),
        "--kernel_max_class_samples",
        str(args.kernel_max_class_samples),
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.run_infonce:
        command.extend(
            [
                "--run_infonce",
                "--lambda_nce_values",
                *[str(value) for value in args.lambda_nce_values],
                "--nce_temperature",
                str(args.nce_temperature),
                "--nce_pair_mode",
                args.nce_pair_mode,
            ]
        )
    if args.run_copa:
        command.extend(
            [
                "--run_copa",
                "--lambda_copa_values",
                *[str(value) for value in args.lambda_copa_values],
                "--copa_proto_weight",
                str(args.copa_proto_weight),
                "--copa_agr_weight",
                str(args.copa_agr_weight),
                "--copa_comp_weight",
                str(args.copa_comp_weight),
                "--copa_comp_margin",
                str(args.copa_comp_margin),
                "--copa_orth_weight",
                str(args.copa_orth_weight),
                "--copa_gate_type",
                args.copa_gate_type,
                "--copa_gate_metric",
                args.copa_gate_metric,
                "--copa_kernel_bandwidth",
                str(args.copa_kernel_bandwidth),
            ]
        )
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
            "direct_add_alpha_values": args.direct_add_alpha_values,
            "run_infonce": args.run_infonce,
            "lambda_nce_values": args.lambda_nce_values,
            "nce_temperature": args.nce_temperature,
            "nce_pair_mode": args.nce_pair_mode,
            "disagreement_metric": args.disagreement_metric,
            "disagreement_pair_mode": args.disagreement_pair_mode,
            "kernel_bandwidth": args.kernel_bandwidth,
            "kernel_pair_mode": args.kernel_pair_mode,
            "kernel_class_weight": args.kernel_class_weight,
            "kernel_max_class_samples": args.kernel_max_class_samples,
            "run_copa": args.run_copa,
            "lambda_copa_values": args.lambda_copa_values,
            "tau_agreement": args.tau_agreement,
            "copa_proto_weight": args.copa_proto_weight,
            "copa_agr_weight": args.copa_agr_weight,
            "copa_comp_weight": args.copa_comp_weight,
            "copa_comp_margin": args.copa_comp_margin,
            "copa_orth_weight": args.copa_orth_weight,
            "copa_gate_type": args.copa_gate_type,
            "copa_gate_metric": args.copa_gate_metric,
            "copa_kernel_bandwidth": args.copa_kernel_bandwidth,
            "patience": args.patience,
            "deterministic": args.deterministic,
            "error_min_seeds": args.error_min_seeds,
            "error_sign_rate": args.error_sign_rate,
            "quiet": args.quiet,
        },
    )

    run_dirs: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for seed in args.seeds:
        run_dirs.append((seed, run_one_seed(args, run_root, seed, seen)))

    delta_all = read_seed_csv(run_dirs, "delta_metrics.csv")
    group_all = read_seed_csv(run_dirs, "group_metrics.csv")
    direct_add_delta_all = read_seed_csv(run_dirs, "direct_add_delta_metrics.csv")
    direct_add_alpha_delta_all = read_seed_csv(
        run_dirs,
        "direct_add_alpha_test_delta_metrics.csv",
    )
    reliability_delta_all = read_seed_csv(run_dirs, "high_d_reliability_delta.csv")
    reliability_metrics_all = read_seed_csv(run_dirs, "high_d_reliability_metrics.csv")
    relation_state_delta_all = read_seed_csv(run_dirs, "relation_state_delta.csv")
    relation_state_metrics_all = read_seed_csv(run_dirs, "relation_state_metrics.csv")
    direct_add_relation_state_delta_all = read_seed_csv(
        run_dirs,
        "direct_add_relation_state_delta.csv",
    )
    concat_aware_all = read_seed_csv(run_dirs, "concat_aware_motivation.csv")
    feature_consistency_all = read_seed_csv(run_dirs, "feature_consistency_diagnostic.csv")
    residual_diagnostic_all = read_seed_csv(run_dirs, "residual_distribution_diagnostic.csv")
    residual_probe_all = read_seed_csv(run_dirs, "residual_discriminative_probe.csv")
    selective_prototype_all = read_seed_csv(
        run_dirs,
        "selective_agreement_prototype_check.csv",
    )
    label_aware_relation_all = read_seed_csv(run_dirs, "label_aware_relation_summary.csv")
    lambda_delta_all = read_seed_csv(run_dirs, "lambda_test_delta_metrics.csv")
    lambda_reliability_delta_all = read_seed_csv(
        run_dirs,
        "lambda_high_d_reliability_delta.csv",
    )
    infonce_delta_all = pd.DataFrame()
    infonce_reliability_delta_all = pd.DataFrame()
    infonce_relation_state_delta_all = pd.DataFrame()
    infonce_lambda_delta_all = pd.DataFrame()
    infonce_lambda_reliability_delta_all = pd.DataFrame()
    if args.run_infonce:
        infonce_delta_all = read_seed_csv(run_dirs, "infonce_delta_metrics.csv")
        infonce_reliability_delta_all = read_seed_csv(
            run_dirs,
            "infonce_high_d_reliability_delta.csv",
        )
        infonce_relation_state_delta_all = read_seed_csv(
            run_dirs,
            "infonce_relation_state_delta.csv",
        )
        infonce_lambda_delta_all = read_seed_csv(
            run_dirs,
            "infonce_lambda_test_delta_metrics.csv",
        )
        infonce_lambda_reliability_delta_all = read_seed_csv(
            run_dirs,
            "infonce_lambda_high_d_reliability_delta.csv",
        )
    copa_delta_all = pd.DataFrame()
    copa_reliability_delta_all = pd.DataFrame()
    copa_relation_state_delta_all = pd.DataFrame()
    copa_lambda_delta_all = pd.DataFrame()
    copa_lambda_reliability_delta_all = pd.DataFrame()
    if args.run_copa:
        copa_delta_all = read_seed_csv(run_dirs, "copa_delta_metrics.csv")
        copa_reliability_delta_all = read_seed_csv(
            run_dirs,
            "copa_high_d_reliability_delta.csv",
        )
        copa_relation_state_delta_all = read_seed_csv(
            run_dirs,
            "copa_relation_state_delta.csv",
        )
        copa_lambda_delta_all = read_seed_csv(
            run_dirs,
            "copa_lambda_test_delta_metrics.csv",
        )
        copa_lambda_reliability_delta_all = read_seed_csv(
            run_dirs,
            "copa_lambda_high_d_reliability_delta.csv",
        )

    delta_all.to_csv(summary_dir / "multi_seed_delta_all.csv", index=False, encoding="utf-8-sig")
    group_all.to_csv(
        summary_dir / "multi_seed_group_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_delta_all.to_csv(
        summary_dir / "direct_add_delta_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_alpha_delta_all.to_csv(
        summary_dir / "direct_add_alpha_test_delta_all.csv",
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
    relation_state_delta_all.to_csv(
        summary_dir / "relation_state_delta_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    relation_state_metrics_all.to_csv(
        summary_dir / "relation_state_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_relation_state_delta_all.to_csv(
        summary_dir / "direct_add_relation_state_delta_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    concat_aware_all.to_csv(
        summary_dir / "concat_aware_motivation_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_consistency_all.to_csv(
        summary_dir / "feature_consistency_diagnostic_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_diagnostic_all.to_csv(
        summary_dir / "residual_distribution_diagnostic_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_probe_all.to_csv(
        summary_dir / "residual_discriminative_probe_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selective_prototype_all.to_csv(
        summary_dir / "selective_agreement_prototype_check_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    label_aware_relation_all.to_csv(
        summary_dir / "label_aware_relation_summary_all.csv",
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
    if args.run_infonce:
        infonce_delta_all.to_csv(
            summary_dir / "infonce_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_reliability_delta_all.to_csv(
            summary_dir / "infonce_high_d_reliability_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_relation_state_delta_all.to_csv(
            summary_dir / "infonce_relation_state_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_lambda_delta_all.to_csv(
            summary_dir / "infonce_lambda_test_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_lambda_reliability_delta_all.to_csv(
            summary_dir / "infonce_lambda_high_d_reliability_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if args.run_copa:
        copa_delta_all.to_csv(
            summary_dir / "copa_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_reliability_delta_all.to_csv(
            summary_dir / "copa_high_d_reliability_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_relation_state_delta_all.to_csv(
            summary_dir / "copa_relation_state_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_lambda_delta_all.to_csv(
            summary_dir / "copa_lambda_test_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_lambda_reliability_delta_all.to_csv(
            summary_dir / "copa_lambda_high_d_reliability_delta_all.csv",
            index=False,
            encoding="utf-8-sig",
        )

    delta_summary = flatten_summary(
        delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "lambda_align"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    group_summary = flatten_summary(
        group_all,
        ["method", "group"],
        ["acc", "macro_f1", "n"],
    )
    direct_add_delta_summary = flatten_summary(
        direct_add_delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "direct_add_alpha"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    direct_add_alpha_delta_summary = flatten_summary(
        direct_add_alpha_delta_all,
        ["direct_add_alpha", "group"],
        ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    reliability_summary = flatten_summary(
        reliability_delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "lambda_align"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    relation_state_summary = flatten_summary(
        relation_state_delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "lambda_align"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    relation_state_metrics_summary = flatten_summary(
        relation_state_metrics_all,
        ["method", "group"],
        ["acc", "macro_f1", "n"],
    )
    direct_add_relation_state_summary = flatten_summary(
        direct_add_relation_state_delta_all,
        ["group"],
        ["delta_acc", "delta_macro_f1", "direct_add_alpha"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    concat_aware_summary = flatten_summary(
        concat_aware_all,
        ["group"],
        [
            "concat_macro_f1",
            "uncond_align_macro_f1",
            "direct_add_macro_f1",
            "infonce_macro_f1",
            "soft_split_probe_macro_f1",
            "text_anchor_probe_macro_f1",
            "residual_gain_macro_f1",
            "text_anchor_residual_gain_macro_f1",
            "shuffled_residual_only_macro_f1",
            "text_anchor_shuffled_residual_macro_f1",
            "lambda_align",
            "direct_add_alpha",
            "lambda_nce",
        ],
    )
    feature_consistency_summary = flatten_summary(
        feature_consistency_all,
        ["group"],
        ["n", "avg_Dpred", "avg_Dfeat", "spearman_Dpred_Dfeat"],
    )
    residual_diagnostic_summary = flatten_summary(
        residual_diagnostic_all,
        ["group"],
        ["n", "avg_Dpred", "avg_Dfeat", "residual_dist", "residual_sep"],
    )
    residual_probe_summary = flatten_summary(
        residual_probe_all,
        ["group"],
        [
            "train_n",
            "test_n",
            "common_only_macro_f1",
            "residual_only_macro_f1",
            "common_residual_macro_f1",
            "residual_gain_macro_f1",
            "shuffled_residual_only_macro_f1",
            "text_anchor_residual_only_macro_f1",
            "text_anchor_common_residual_macro_f1",
            "text_anchor_residual_gain_macro_f1",
            "text_anchor_shuffled_residual_macro_f1",
        ],
    )
    selective_prototype_summary = flatten_summary(
        selective_prototype_all,
        ["prototype", "eval_group"],
        [
            "train_n",
            "prototype_purity",
            "intra_class_compactness",
            "nearest_proto_acc",
            "nearest_proto_macro_f1",
        ],
    )
    lambda_delta_summary = flatten_summary(
        lambda_delta_all,
        ["lambda_align", "group"],
        ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    lambda_reliability_summary = flatten_summary(
        lambda_reliability_delta_all,
        ["lambda_align", "group"],
        ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
        sign_cols=["delta_acc", "delta_macro_f1"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )
    infonce_delta_summary = pd.DataFrame()
    infonce_reliability_summary = pd.DataFrame()
    infonce_relation_state_summary = pd.DataFrame()
    infonce_lambda_delta_summary = pd.DataFrame()
    infonce_lambda_reliability_summary = pd.DataFrame()
    if args.run_infonce:
        infonce_delta_summary = flatten_summary(
            infonce_delta_all,
            ["group"],
            ["delta_acc", "delta_macro_f1", "lambda_nce"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        infonce_reliability_summary = flatten_summary(
            infonce_reliability_delta_all,
            ["group"],
            ["delta_acc", "delta_macro_f1", "lambda_nce"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        infonce_relation_state_summary = flatten_summary(
            infonce_relation_state_delta_all,
            ["group"],
            ["delta_acc", "delta_macro_f1", "lambda_nce"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        infonce_lambda_delta_summary = flatten_summary(
            infonce_lambda_delta_all,
            ["lambda_nce", "group"],
            ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        infonce_lambda_reliability_summary = flatten_summary(
            infonce_lambda_reliability_delta_all,
            ["lambda_nce", "group"],
            ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
    relation_value_cols = [
        column
        for column in label_aware_relation_all.columns
        if column not in {"seed", "run_dir", "split"}
    ]
    label_aware_relation_summary = flatten_summary(
        label_aware_relation_all,
        ["split"],
        relation_value_cols,
    )
    copa_delta_summary = pd.DataFrame()
    copa_reliability_summary = pd.DataFrame()
    copa_relation_state_summary = pd.DataFrame()
    copa_lambda_delta_summary = pd.DataFrame()
    copa_lambda_reliability_summary = pd.DataFrame()
    if args.run_copa:
        copa_delta_summary = flatten_summary(
            copa_delta_all,
            ["group"],
            ["delta_acc", "delta_macro_f1", "lambda_copa"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        copa_reliability_summary = flatten_summary(
            copa_reliability_delta_all,
            ["group"],
            ["delta_acc", "delta_macro_f1", "lambda_copa"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        copa_relation_state_summary = flatten_summary(
            copa_relation_state_delta_all,
            ["group"],
            ["delta_acc", "delta_macro_f1", "lambda_copa"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        copa_lambda_delta_summary = flatten_summary(
            copa_lambda_delta_all,
            ["lambda_copa", "group"],
            ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
        )
        copa_lambda_reliability_summary = flatten_summary(
            copa_lambda_reliability_delta_all,
            ["lambda_copa", "group"],
            ["delta_acc", "delta_macro_f1", "valid_macro_f1", "valid_acc"],
            sign_cols=["delta_acc", "delta_macro_f1"],
            min_count=args.error_min_seeds,
            sign_rate_threshold=args.error_sign_rate,
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
    direct_add_delta_summary.to_csv(
        summary_dir / "direct_add_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_alpha_delta_summary.to_csv(
        summary_dir / "direct_add_alpha_test_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability_summary.to_csv(
        summary_dir / "high_d_reliability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    relation_state_summary.to_csv(
        summary_dir / "relation_state_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    relation_state_metrics_summary.to_csv(
        summary_dir / "relation_state_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    direct_add_relation_state_summary.to_csv(
        summary_dir / "direct_add_relation_state_delta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    concat_aware_summary.to_csv(
        summary_dir / "concat_aware_motivation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_consistency_summary.to_csv(
        summary_dir / "feature_consistency_diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_diagnostic_summary.to_csv(
        summary_dir / "residual_distribution_diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_probe_summary.to_csv(
        summary_dir / "residual_discriminative_probe_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selective_prototype_summary.to_csv(
        summary_dir / "selective_agreement_prototype_check_summary.csv",
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
    if args.run_infonce:
        infonce_delta_summary.to_csv(
            summary_dir / "infonce_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_reliability_summary.to_csv(
            summary_dir / "infonce_high_d_reliability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_relation_state_summary.to_csv(
            summary_dir / "infonce_relation_state_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_lambda_delta_summary.to_csv(
            summary_dir / "infonce_lambda_test_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        infonce_lambda_reliability_summary.to_csv(
            summary_dir / "infonce_lambda_high_d_reliability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    label_aware_relation_summary.to_csv(
        summary_dir / "label_aware_relation_multi_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if args.run_copa:
        copa_delta_summary.to_csv(
            summary_dir / "copa_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_reliability_summary.to_csv(
            summary_dir / "copa_high_d_reliability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_relation_state_summary.to_csv(
            summary_dir / "copa_relation_state_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_lambda_delta_summary.to_csv(
            summary_dir / "copa_lambda_test_delta_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        copa_lambda_reliability_summary.to_csv(
            summary_dir / "copa_lambda_high_d_reliability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    error_frames: list[tuple[str, pd.DataFrame, list[str]]] = [
        ("uncond_align_delta", delta_summary, ["group"]),
        ("uncond_align_high_d_reliability", reliability_summary, ["group"]),
        ("uncond_align_relation_state", relation_state_summary, ["group"]),
        ("direct_add_delta", direct_add_delta_summary, ["group"]),
        ("direct_add_relation_state", direct_add_relation_state_summary, ["group"]),
        ("lambda_align_strength", lambda_delta_summary, ["lambda_align", "group"]),
        (
            "lambda_align_high_d_reliability",
            lambda_reliability_summary,
            ["lambda_align", "group"],
        ),
    ]
    if args.run_infonce:
        error_frames.extend(
            [
                ("infonce_delta", infonce_delta_summary, ["group"]),
                ("infonce_high_d_reliability", infonce_reliability_summary, ["group"]),
                ("infonce_relation_state", infonce_relation_state_summary, ["group"]),
                ("lambda_nce_strength", infonce_lambda_delta_summary, ["lambda_nce", "group"]),
                (
                    "lambda_nce_high_d_reliability",
                    infonce_lambda_reliability_summary,
                    ["lambda_nce", "group"],
                ),
            ]
        )
    if args.run_copa:
        error_frames.extend(
            [
                ("copa_delta", copa_delta_summary, ["group"]),
                ("copa_high_d_reliability", copa_reliability_summary, ["group"]),
                ("copa_relation_state", copa_relation_state_summary, ["group"]),
                ("lambda_copa_strength", copa_lambda_delta_summary, ["lambda_copa", "group"]),
                (
                    "lambda_copa_high_d_reliability",
                    copa_lambda_reliability_summary,
                    ["lambda_copa", "group"],
                ),
            ]
        )
    write_error_control_report(summary_dir / "error_control_report.csv", error_frames)

    save_multi_seed_delta_plot(delta_summary, summary_dir / "multi_seed_delta_macro_f1.png")
    save_detailed_delta_plot(
        delta_all,
        delta_summary,
        summary_dir / "multi_seed_delta_macro_f1_detailed.png",
        title="Unconditional alignment gain by disagreement group",
        ylabel="Delta Macro-F1 (UncondAlign - Concat)",
    )
    save_reliability_delta_plot(
        reliability_summary,
        summary_dir / "high_d_reliability_delta.png",
    )
    save_detailed_delta_plot(
        reliability_delta_all,
        reliability_summary,
        summary_dir / "high_d_reliability_delta_detailed.png",
        group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
        title="Unconditional alignment on High-D reliability split",
        ylabel="Delta Macro-F1 (UncondAlign - Concat)",
    )
    save_detailed_delta_plot(
        relation_state_delta_all,
        relation_state_summary,
        summary_dir / "relation_state_delta_detailed.png",
        group_order=RELATION_STATE_GROUP_ORDER,
        title="Unconditional alignment by relation state",
        ylabel="Delta Macro-F1 (UncondAlign - Concat)",
    )
    save_detailed_delta_plot(
        direct_add_relation_state_delta_all,
        direct_add_relation_state_summary,
        summary_dir / "direct_add_relation_state_delta_detailed.png",
        group_order=RELATION_STATE_GROUP_ORDER,
        title="DirectAdd by relation state",
        ylabel="Delta Macro-F1 (DirectAdd - Concat)",
    )
    save_lambda_curve_plot(
        lambda_delta_summary,
        summary_dir / "lambda_delta_macro_f1_curve.png",
        title="Multi-seed lambda alignment strength curve",
        raw_frame=lambda_delta_all,
    )
    heatmap_summaries = {
        "UncondAlign": relation_state_summary,
        "DirectAdd": direct_add_relation_state_summary,
    }
    if args.run_infonce:
        save_detailed_delta_plot(
            infonce_delta_all,
            infonce_delta_summary,
            summary_dir / "infonce_delta_macro_f1_detailed.png",
            title="InfoNCE gain by disagreement group",
            ylabel="Delta Macro-F1 (UncondInfoNCE - Concat)",
        )
        save_detailed_delta_plot(
            infonce_reliability_delta_all,
            infonce_reliability_summary,
            summary_dir / "infonce_high_d_reliability_delta_detailed.png",
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            title="InfoNCE on High-D reliability split",
            ylabel="Delta Macro-F1 (UncondInfoNCE - Concat)",
        )
        save_detailed_delta_plot(
            infonce_relation_state_delta_all,
            infonce_relation_state_summary,
            summary_dir / "infonce_relation_state_delta_detailed.png",
            group_order=RELATION_STATE_GROUP_ORDER,
            title="InfoNCE by relation state",
            ylabel="Delta Macro-F1 (UncondInfoNCE - Concat)",
        )
        heatmap_summaries["UncondInfoNCE"] = infonce_relation_state_summary
        save_lambda_curve_plot(
            infonce_lambda_delta_summary,
            summary_dir / "infonce_lambda_delta_macro_f1_curve.png",
            title="Multi-seed InfoNCE strength curve",
            x_col="lambda_nce",
            x_label="lambda_nce",
            y_label="Delta Macro-F1 (UncondInfoNCE - Concat)",
            raw_frame=infonce_lambda_delta_all,
        )
    if args.run_copa:
        save_detailed_delta_plot(
            copa_delta_all,
            copa_delta_summary,
            summary_dir / "copa_delta_macro_f1_detailed.png",
            title="CoPA gain by disagreement group",
            ylabel="Delta Macro-F1 (CoPA - Concat)",
        )
        save_detailed_delta_plot(
            copa_reliability_delta_all,
            copa_reliability_summary,
            summary_dir / "copa_high_d_reliability_delta_detailed.png",
            group_order=HIGH_D_RELIABILITY_GROUP_ORDER,
            title="CoPA on High-D reliability split",
            ylabel="Delta Macro-F1 (CoPA - Concat)",
        )
        save_detailed_delta_plot(
            copa_relation_state_delta_all,
            copa_relation_state_summary,
            summary_dir / "copa_relation_state_delta_detailed.png",
            group_order=RELATION_STATE_GROUP_ORDER,
            title="CoPA by relation state",
            ylabel="Delta Macro-F1 (CoPA - Concat)",
        )
        heatmap_summaries["CoPA"] = copa_relation_state_summary
        save_lambda_curve_plot(
            copa_lambda_delta_summary,
            summary_dir / "copa_lambda_delta_macro_f1_curve.png",
            title="Multi-seed CoPA strength curve",
            x_col="lambda_copa",
            x_label="lambda_copa",
            y_label="Delta Macro-F1 (CoPA - Concat)",
            raw_frame=copa_lambda_delta_all,
        )
    save_method_relation_state_heatmap(
        heatmap_summaries,
        summary_dir / "relation_state_method_comparison_heatmap.png",
    )
    write_conclusion(delta_summary, summary_dir / "experiment_one_conclusion.json")

    print("\nMulti-seed delta summary:")
    print(delta_summary.to_string(index=False))
    print("\nHigh-D reliability summary:")
    print(reliability_summary.to_string(index=False))
    print("\nRelation-state summary:")
    print(relation_state_summary.to_string(index=False))
    print("\nDirectAdd relation-state summary:")
    print(direct_add_relation_state_summary.to_string(index=False))
    print("\nConcat-aware motivation summary:")
    print(concat_aware_summary.to_string(index=False))
    print("\nResidual probe summary:")
    print(residual_probe_summary.to_string(index=False))
    print("\nLambda strength delta summary:")
    print(lambda_delta_summary.to_string(index=False))
    if args.run_infonce:
        print("\nInfoNCE delta summary:")
        print(infonce_delta_summary.to_string(index=False))
        print("\nInfoNCE lambda strength delta summary:")
        print(infonce_lambda_delta_summary.to_string(index=False))
    if args.run_copa:
        print("\nCoPA delta summary:")
        print(copa_delta_summary.to_string(index=False))
        print("\nCoPA lambda strength delta summary:")
        print(copa_lambda_delta_summary.to_string(index=False))
    print(f"\nSaved multi-seed summary to: {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
