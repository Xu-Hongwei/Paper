from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MultimodalSplitDataset, infer_input_dims, load_npz_splits  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.model import MultimodalClassifier  # noqa: E402
from src.train import predict  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


GROUP_SPECS = (
    ("overall", "Overall"),
    ("disagreement", "Low-D"),
    ("disagreement", "Mid-D"),
    ("disagreement", "High-D"),
    ("relation_state", "RA"),
    ("relation_state", "UA"),
    ("relation_state", "Mid-D"),
    ("relation_state", "RD"),
    ("relation_state", "ND"),
)
RELATION_STATE_ORDER = ("RA", "UA", "Mid-D", "RD", "ND")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v6 mechanism diagnostics from completed multi-seed runs."
    )
    parser.add_argument("--dataset", default="mosei")
    parser.add_argument(
        "--multi_seed_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Completed multi_seed_* directories to merge.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--expected_seeds", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
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
    if expected_seeds:
        expected = sorted(set(expected_seeds))
        missing_seeds = sorted(set(expected) - set(unique_seeds))
        unexpected_seeds = sorted(set(unique_seeds) - set(expected))
    else:
        missing_seeds = []
        unexpected_seeds = []
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
            and column != "seed"
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_cols, keys))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else len(group)
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


def load_model_predictions(
    run_dir: Path,
    config: dict,
    *,
    model_filename: str,
    batch_size: int | None,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    splits = load_npz_splits(Path(config["data_path"]), label_mode=config["label_mode"])
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
    try:
        state = torch.load(run_dir / model_filename, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / model_filename, map_location=device)
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


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    eps = 1e-8
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), eps)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), eps)
    return 1.0 - np.sum(left_norm * right_norm, axis=1)


def sample_alignment_distance(pred: dict[str, np.ndarray], pair_mode: str) -> np.ndarray:
    pairs = [
        cosine_distance(pred["h_t"], pred["h_a"]),
        cosine_distance(pred["h_t"], pred["h_v"]),
    ]
    if pair_mode == "full_pair":
        pairs.append(cosine_distance(pred["h_v"], pred["h_a"]))
    elif pair_mode != "text_anchor":
        raise ValueError("pair_mode must be 'text_anchor' or 'full_pair'.")
    return np.stack(pairs, axis=1).mean(axis=1)


def group_mask(groups: pd.DataFrame, group_type: str, group: str) -> np.ndarray:
    if group_type == "overall":
        return np.ones(len(groups), dtype=bool)
    if group_type == "disagreement":
        return groups["group"].to_numpy() == group
    if group_type == "relation_state":
        return groups["relation_state"].to_numpy() == group
    raise ValueError(f"Unknown group type: {group_type}")


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"acc": math.nan, "macro_f1": math.nan}
    return classification_metrics(y_true, y_pred)


def build_alignment_rows(
    seed: int,
    run_dir: Path,
    *,
    batch_size: int | None,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, object]]:
    config = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    groups = pd.read_csv(run_dir / "test_groups.csv", encoding="utf-8-sig").sort_values(
        "index"
    )
    concat_pred = load_model_predictions(
        run_dir,
        config,
        model_filename="concat_model.pt",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    align_pred = load_model_predictions(
        run_dir,
        config,
        model_filename="uncond_align_model.pt",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    y_true = concat_pred["y_true"].astype(np.int64)
    if not np.array_equal(y_true, align_pred["y_true"].astype(np.int64)):
        raise ValueError(f"Concat and UncondAlign labels differ in {run_dir}")
    if not np.array_equal(y_true, groups["label_cls"].to_numpy(dtype=np.int64)):
        raise ValueError(f"Predictions and test_groups labels differ in {run_dir}")

    pair_mode = str(config.get("align_pair_mode", "text_anchor"))
    concat_distance = sample_alignment_distance(concat_pred, pair_mode)
    align_distance = sample_alignment_distance(align_pred, pair_mode)
    concat_y = concat_pred["y_pred"].astype(np.int64)
    align_y = align_pred["y_pred"].astype(np.int64)
    rows = []
    for group_type, group_name in GROUP_SPECS:
        mask = group_mask(groups, group_type, group_name)
        n = int(mask.sum())
        concat_metrics = safe_metrics(y_true[mask], concat_y[mask])
        align_metrics = safe_metrics(y_true[mask], align_y[mask])
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "group_type": group_type,
                "group": group_name,
                "n": n,
                "avg_D_sample": float(groups.loc[mask, "D_sample"].mean()) if n else math.nan,
                "avg_R": float(groups.loc[mask, "R_sample"].mean()) if n else math.nan,
                "concat_hidden_distance": float(concat_distance[mask].mean()) if n else math.nan,
                "uncond_align_hidden_distance": float(align_distance[mask].mean())
                if n
                else math.nan,
                "hidden_distance_delta": float(align_distance[mask].mean() - concat_distance[mask].mean())
                if n
                else math.nan,
                "concat_macro_f1": concat_metrics["macro_f1"],
                "uncond_align_macro_f1": align_metrics["macro_f1"],
                "delta_macro_f1": align_metrics["macro_f1"] - concat_metrics["macro_f1"],
                "concat_acc": concat_metrics["acc"],
                "uncond_align_acc": align_metrics["acc"],
                "delta_acc": align_metrics["acc"] - concat_metrics["acc"],
                "lambda_align": summary.get("best_lambda_align", math.nan),
                "align_pair_mode": pair_mode,
            }
        )
    return rows


def read_summary_csv(multi_seed_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for multi_seed_dir in multi_seed_dirs:
        path = multi_seed_dir / "summary" / filename
        if path.exists():
            frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_dynamic_weight_rows(multi_seed_dirs: list[Path]) -> pd.DataFrame:
    weights = read_summary_csv(multi_seed_dirs, "dynamic_fusion_weight_relation_all.csv")
    deltas = read_summary_csv(multi_seed_dirs, "dynamic_fusion_relation_state_delta_all.csv")
    if weights.empty or deltas.empty:
        return pd.DataFrame()
    keep_delta = deltas[
        [
            "seed",
            "run_dir",
            "group",
            "delta_acc",
            "delta_macro_f1",
            "lambda_dynamic_weight",
            "dynamic_router_temperature",
        ]
    ].copy()
    merged = weights.merge(
        keep_delta,
        on=["seed", "run_dir", "group", "lambda_dynamic_weight", "dynamic_router_temperature"],
        how="left",
    )
    merged = merged[merged["group"].isin(RELATION_STATE_ORDER)].copy()
    return merged


def markdown_table(frame: pd.DataFrame) -> str:
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|")

    columns = [escape(column) for column in frame.columns]
    rows = [[escape(value) for value in record] for record in frame.to_numpy()]
    widths = [
        max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
        for index, column in enumerate(columns)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(columns), separator, *[fmt_row(row) for row in rows]])


def fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def compact_alignment_table(summary: pd.DataFrame) -> pd.DataFrame:
    selector = pd.DataFrame(
        [
            ("overall", "Overall", "overall", 0),
            ("disagreement", "High-D", "D", 1),
            ("relation_state", "RA", "relation", 2),
            ("relation_state", "UA", "relation", 3),
            ("relation_state", "Mid-D", "relation", 4),
            ("relation_state", "RD", "relation", 5),
            ("relation_state", "ND", "relation", 6),
        ],
        columns=["group_type", "group", "scope", "_order"],
    )
    subset = summary.merge(selector, on=["group_type", "group"], how="inner")
    subset = subset.sort_values("_order")
    return pd.DataFrame(
        {
            "Scope": subset["scope"],
            "Group": subset["group"],
            "Concat Dist": subset["concat_hidden_distance_mean"].map(fmt),
            "Align Dist": subset["uncond_align_hidden_distance_mean"].map(fmt),
            "Dist Delta": subset["hidden_distance_delta_mean"].map(fmt),
            "Delta Macro-F1": subset["delta_macro_f1_mean"].map(fmt),
            "F1 EC": subset["delta_macro_f1_passes_error_control"].map(str),
        }
    )


def compact_dynamic_table(summary: pd.DataFrame) -> pd.DataFrame:
    subset = summary[summary["group"].isin(RELATION_STATE_ORDER)].copy()
    order = {group: index for index, group in enumerate(RELATION_STATE_ORDER)}
    subset["_order"] = subset["group"].map(order)
    subset = subset.sort_values("_order")
    return pd.DataFrame(
        {
            "Group": subset["group"],
            "w_text": subset["avg_w_text_mean"].map(fmt),
            "w_vision": subset["avg_w_vision_mean"].map(fmt),
            "w_audio": subset["avg_w_audio_mean"].map(fmt),
            "Entropy": subset["avg_weight_entropy_mean"].map(fmt),
            "Text Dom.": subset["text_dominant_rate_mean"].map(fmt),
            "Delta Macro-F1": subset["delta_macro_f1_mean"].map(fmt),
            "F1 EC": subset["delta_macro_f1_passes_error_control"].map(str),
        }
    )


def write_report(
    output_dir: Path,
    alignment_summary: pd.DataFrame,
    dynamic_summary: pd.DataFrame,
) -> None:
    alignment_table = compact_alignment_table(alignment_summary)
    dynamic_table = compact_dynamic_table(dynamic_summary)
    lines = [
        "# v6 Mechanism Diagnostics",
        "",
        "These diagnostics are explanatory only. They should support the motivation section, not claim final method success.",
        "",
        "## Unconditional Alignment Mechanism",
        "",
        markdown_table(alignment_table),
        "",
        "Interpretation rule: negative distance delta means hidden states moved closer; this is useful only if Macro-F1 improves in the same groups.",
        "",
        "## DynamicFusion Weight Mechanism",
        "",
        markdown_table(dynamic_table),
        "",
        "Interpretation rule: high text-dominant rates with weak RD gain indicate modality selection rather than relation-state scheduling.",
        "",
    ]
    (output_dir / "v6_mechanism_diagnostics.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


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
            / f"v6_mechanism_analysis_{min(seeds)}_{max(seeds)}"
        )
    output_dir = ensure_dir(output_dir)

    device = torch.device(args.device)
    alignment_rows: list[dict[str, object]] = []
    for seed, run_dir in run_dirs:
        alignment_rows.extend(
            build_alignment_rows(
                seed,
                run_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        )
    alignment_all = pd.DataFrame(alignment_rows)
    alignment_summary = aggregate_numeric(
        alignment_all,
        ["group_type", "group"],
        sign_cols=["hidden_distance_delta", "delta_macro_f1", "delta_acc"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )

    dynamic_all = build_dynamic_weight_rows(args.multi_seed_dirs)
    dynamic_summary = aggregate_numeric(
        dynamic_all,
        ["group"],
        sign_cols=["delta_macro_f1", "delta_acc"],
        min_count=args.error_min_seeds,
        sign_rate_threshold=args.error_sign_rate,
    )

    alignment_all.to_csv(
        output_dir / "alignment_mechanism_profile_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    alignment_summary.to_csv(
        output_dir / "alignment_mechanism_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dynamic_all.to_csv(
        output_dir / "dynamic_weight_mechanism_profile_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dynamic_summary.to_csv(
        output_dir / "dynamic_weight_mechanism_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(
        output_dir / "v6_mechanism_analysis_summary.json",
        {
            "dataset": args.dataset,
            "multi_seed_dirs": [str(path) for path in args.multi_seed_dirs],
            "output_dir": str(output_dir),
            "seed_coverage": coverage,
            "alignment_distance": "mean per-sample 1 - cosine over align_pair_mode pairs",
            "dynamic_weight_source": "dynamic_fusion_weight_relation_all.csv merged with dynamic_fusion_relation_state_delta_all.csv",
        },
    )
    write_report(output_dir, alignment_summary, dynamic_summary)
    print(f"Saved v6 mechanism diagnostics to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
