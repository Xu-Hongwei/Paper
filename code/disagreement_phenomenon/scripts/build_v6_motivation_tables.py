from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir  # noqa: E402


METHOD_ORDER = ("UncondAlign", "UncondInfoNCE", "DynamicFusion", "BalancedDirectAdd")
TARGET_ORDER = ("Overall", "High-D", "RD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact paper-ready v6 motivation tables from cause analysis outputs."
    )
    parser.add_argument(
        "--cause_dir",
        type=Path,
        default=ROOT / "outputs" / "mosei" / "v6_cause_analysis_1_15",
        help="Directory produced by analyze_v6_cause.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "outputs" / "mosei" / "v6_motivation_tables_1_15",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def fmt_ci(row: pd.Series, stem: str, digits: int = 3) -> str:
    low = row.get(f"{stem}_ci95_low")
    high = row.get(f"{stem}_ci95_high")
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def write_table(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    frame.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (output_dir / f"{stem}.md").write_text(
        markdown_table(frame),
        encoding="utf-8",
    )


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
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(columns), separator, *[fmt_row(row) for row in rows]])


def build_experiment1_table(
    cause: pd.DataFrame,
    modality: pd.DataFrame,
    summary: dict,
) -> pd.DataFrame:
    rows = []
    d_rows = modality[modality["group_type"] == "disagreement"].copy()
    cause_d = cause[cause["group_type"] == "disagreement"].set_index("group")
    order = {"Low-D": 0, "Mid-D": 1, "High-D": 2}
    d_rows["_order"] = d_rows["group"].map(order)
    d_rows = d_rows.sort_values("_order")
    consistency = summary.get("concat_high_d_order_consistency", {})
    consistency_text = f"{consistency.get('passed', '')}/{consistency.get('total', '')}"
    for _, row in d_rows.iterrows():
        cause_row = cause_d.loc[row["group"]]
        rows.append(
            {
                "Group": row["group"],
                "N": fmt(row["n_mean"], 1),
                "avg D": fmt(cause_row["avg_D_sample_mean"], 4),
                "avg R": fmt(cause_row["avg_R_mean"], 4),
                "Concat Macro-F1": fmt(row["concat_fusion_macro_f1_mean"]),
                "95% CI": fmt_ci(row, "concat_fusion_macro_f1"),
                "Order Consistency": consistency_text,
            }
        )
    return pd.DataFrame(rows)


def build_cause_table(cause: pd.DataFrame, modality: pd.DataFrame) -> pd.DataFrame:
    cause_d = cause[cause["group_type"] == "disagreement"].set_index("group")
    mod_d = modality[modality["group_type"] == "disagreement"].set_index("group")
    rows = []
    for group in ("Low-D", "Mid-D", "High-D"):
        c = cause_d.loc[group]
        m = mod_d.loc[group]
        rows.append(
            {
                "Group": group,
                "N": fmt(c["n_mean"], 1),
                "avg |label|": fmt(c["avg_abs_label_reg_mean"]),
                "Label Entropy": fmt(c["label_entropy_mean"]),
                "Class 0/1/2": (
                    f"{fmt(c['class_0_ratio_mean'])}/"
                    f"{fmt(c['class_1_ratio_mean'])}/"
                    f"{fmt(c['class_2_ratio_mean'])}"
                ),
                "Text Acc": fmt(m["text_acc_mean"]),
                "Audio Acc": fmt(m["audio_acc_mean"]),
                "Vision Acc": fmt(m["vision_acc_mean"]),
                "Concat Macro-F1": fmt(m["concat_fusion_macro_f1_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_rd_nd_table(oracle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oracle = oracle.set_index("group")
    for group in ("RD", "ND"):
        row = oracle.loc[group]
        rows.append(
            {
                "Group": group,
                "N": fmt(row["n_mean"], 1),
                "avg D": fmt(row["avg_D_sample_mean"], 4),
                "avg R": fmt(row["avg_R_mean"], 4),
                "Text Macro-F1": fmt(row["text_macro_f1_mean"]),
                "Audio Macro-F1": fmt(row["audio_macro_f1_mean"]),
                "Vision Macro-F1": fmt(row["vision_macro_f1_mean"]),
                "Fusion Macro-F1": fmt(row["fusion_macro_f1_mean"]),
                "Concat Macro-F1": fmt(row["concat_fusion_macro_f1_mean"]),
                "Oracle Macro-F1": fmt(row["oracle_macro_f1_mean"]),
                "Oracle - Fusion": fmt(row["oracle_minus_fusion_macro_f1_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_method_table(method: pd.DataFrame) -> pd.DataFrame:
    method = method.copy()
    method["_method_order"] = method["method"].map(
        {name: index for index, name in enumerate(METHOD_ORDER)}
    )
    rows = []
    for method_name in METHOD_ORDER:
        method_rows = method[method["method"] == method_name].set_index("target")
        row: dict[str, object] = {"Method": method_name}
        for target in TARGET_ORDER:
            if target not in method_rows.index:
                row[f"{target} Delta Macro-F1"] = ""
                row[f"{target} 95% CI"] = ""
                row[f"{target} EC"] = ""
                continue
            record = method_rows.loc[target]
            row[f"{target} Delta Macro-F1"] = fmt(record["delta_macro_f1_mean"], 4)
            row[f"{target} 95% CI"] = fmt_ci(record, "delta_macro_f1", digits=4)
            row[f"{target} EC"] = str(bool(record["delta_macro_f1_passes_error_control"]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_class_prior_table(class_prior: pd.DataFrame, modality: pd.DataFrame) -> pd.DataFrame:
    prior_d = class_prior[class_prior["group_type"] == "disagreement"].set_index("group")
    mod_d = modality[modality["group_type"] == "disagreement"].set_index("group")
    class_cols = sorted(
        column
        for column in prior_d.columns
        if column.startswith("class_") and column.endswith("_ratio_mean")
    )
    rows = []
    for group in ("Low-D", "Mid-D", "High-D"):
        p = prior_d.loc[group]
        m = mod_d.loc[group]
        rows.append(
            {
                "Group": group,
                "N": fmt(p["n_mean"], 1),
                "Class Ratios": "/".join(fmt(p[column]) for column in class_cols),
                "Majority Acc": fmt(p["class_prior_majority_acc_mean"]),
                "Concat Acc": fmt(m["concat_fusion_acc_mean"]),
                "Concat Macro-F1": fmt(m["concat_fusion_macro_f1_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_class_wise_table(class_wise: pd.DataFrame) -> pd.DataFrame:
    d_rows = class_wise[class_wise["group_type"] == "disagreement"].copy()
    rows = []
    for class_id in sorted(d_rows["class_id"].dropna().unique()):
        class_rows = d_rows[d_rows["class_id"] == class_id].set_index("group")
        row: dict[str, object] = {"Class": int(class_id)}
        acc_values: dict[str, float] = {}
        for group in ("Low-D", "Mid-D", "High-D"):
            if group not in class_rows.index:
                row[f"{group} N"] = ""
                row[f"{group} Concat Acc"] = ""
                continue
            record = class_rows.loc[group]
            acc_values[group] = float(record["concat_acc_mean"])
            row[f"{group} N"] = fmt(record["n_mean"], 1)
            row[f"{group} Concat Acc"] = fmt(acc_values[group])
        row["High-Low"] = fmt(
            acc_values.get("High-D", float("nan")) - acc_values.get("Low-D", float("nan"))
        )
        row["High-Mid"] = fmt(
            acc_values.get("High-D", float("nan")) - acc_values.get("Mid-D", float("nan"))
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_polarity_correlation_table(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    signal_order = [
        "label_abs_polarity",
        "pred_polarity_conf",
        "pred_confidence",
        "pred_margin",
        "R_sample",
    ]
    for signal in signal_order:
        signal_rows = correlation[
            (correlation["polarity_signal"] == signal)
            & (correlation["d_metric"] == "D_pred")
        ].set_index("correlation_type")
        if signal_rows.empty:
            continue
        rows.append(
            {
                "Signal": signal,
                "Pearson": fmt(
                    signal_rows.loc["pearson", "correlation_mean"]
                    if "pearson" in signal_rows.index
                    else None,
                    4,
                ),
                "Spearman": fmt(
                    signal_rows.loc["spearman", "correlation_mean"]
                    if "spearman" in signal_rows.index
                    else None,
                    4,
                ),
                "N": fmt(
                    signal_rows["n_mean"].iloc[0]
                    if "n_mean" in signal_rows.columns
                    else None,
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_polarity_bin_table(polarity_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_name in ("Low-P", "Mid-P", "High-P"):
        bin_rows = polarity_bin[polarity_bin["polarity_bin"] == bin_name].set_index("group")
        if bin_rows.empty:
            continue
        low = bin_rows.loc["Low-D"] if "Low-D" in bin_rows.index else None
        mid = bin_rows.loc["Mid-D"] if "Mid-D" in bin_rows.index else None
        high = bin_rows.loc["High-D"] if "High-D" in bin_rows.index else None
        low_acc = float(low["concat_acc_mean"]) if low is not None else float("nan")
        mid_acc = float(mid["concat_acc_mean"]) if mid is not None else float("nan")
        high_acc = float(high["concat_acc_mean"]) if high is not None else float("nan")
        rows.append(
            {
                "Polarity Bin": bin_name,
                "Low-D Acc": fmt(low_acc),
                "Mid-D Acc": fmt(mid_acc),
                "High-D Acc": fmt(high_acc),
                "High-Low": fmt(high_acc - low_acc),
                "High-Mid": fmt(high_acc - mid_acc),
                "High-D N": fmt(high["n_mean"] if high is not None else None, 1),
            }
        )
    return pd.DataFrame(rows)


def build_decoupled_table(
    decoupled_eval: pd.DataFrame,
    decoupled_contrast: pd.DataFrame,
) -> pd.DataFrame:
    eval_rows = decoupled_eval.set_index("group")
    contrast = decoupled_contrast.iloc[0] if not decoupled_contrast.empty else {}
    rows = []
    for group in ("Low-D", "Mid-D", "High-D"):
        if group not in eval_rows.index:
            continue
        record = eval_rows.loc[group]
        rows.append(
            {
                "Group": group,
                "N": fmt(record["n_mean"], 1),
                "Cross-seed Concat Acc": fmt(record["concat_acc_mean"]),
                "Cross-seed Macro-F1": fmt(record["concat_macro_f1_mean"]),
            }
        )
    rows.append(
        {
            "Group": "High-Low",
            "N": "",
            "Cross-seed Concat Acc": fmt(contrast.get("high_minus_low_acc_mean")),
            "Cross-seed Macro-F1": fmt(contrast.get("high_minus_low_macro_f1_mean")),
        }
    )
    rows.append(
        {
            "Group": "High-Mid",
            "N": "",
            "Cross-seed Concat Acc": fmt(contrast.get("high_minus_mid_acc_mean")),
            "Cross-seed Macro-F1": fmt(contrast.get("high_minus_mid_macro_f1_mean")),
        }
    )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    experiment1: pd.DataFrame,
    cause: pd.DataFrame,
    rd_nd: pd.DataFrame,
    method: pd.DataFrame,
    class_prior: pd.DataFrame,
    class_wise: pd.DataFrame,
    polarity_corr: pd.DataFrame,
    polarity_bin: pd.DataFrame,
    decoupled: pd.DataFrame,
) -> None:
    lines = [
        "# v6 Motivation Evidence Tables",
        "",
        "Generated from completed 15-seed MOSEI outputs. D is a supervised model-induced, task-aware prediction disagreement; R is a prediction-certainty diagnostic, not a ground-truth quality label.",
        "",
        "## Experiment 1: High-D Is Not Difficulty",
        "",
        markdown_table(experiment1),
        "",
        "## Cause Profile: Why High-D Looks Easier",
        "",
        markdown_table(cause),
        "",
        "## RD / ND Oracle Boundary",
        "",
        markdown_table(rd_nd),
        "",
        "## Method Insufficiency and Minimal Positive Clue",
        "",
        markdown_table(method),
        "",
        "## Class-Prior Control",
        "",
        markdown_table(class_prior),
        "",
        "Majority Acc is an offline label-composition diagnostic within each already-defined group, not a deployable no-label baseline.",
        "",
        "## Class-wise Concat Accuracy",
        "",
        markdown_table(class_wise),
        "",
        "Class-wise accuracy fixes the true class and reports recall-like Concat accuracy inside each D group.",
        "",
        "## D vs Polarity Correlation",
        "",
        markdown_table(polarity_corr),
        "",
        "`label_abs_polarity` is an offline label-aware diagnostic; prediction-based signals are model-induced proxies.",
        "",
        "## Polarity-bin Controlled D Analysis",
        "",
        markdown_table(polarity_bin),
        "",
        "Rows compare Low/Mid/High-D within the same |label| polarity tertile.",
        "",
        "## Cross-seed Decoupled D Diagnostic",
        "",
        markdown_table(decoupled),
        "",
        "D groups come from one diagnostic seed and Concat evaluation comes from another seed; same-seed pairs are excluded.",
        "",
        "Recommended wording: BalancedDirectAdd is a minimal positive clue, not a final method. It supplies the first error-controlled RD gain among these tested controls.",
        "",
    ]
    (output_dir / "v6_motivation_evidence_tables.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    cause_dir = args.cause_dir
    required = [
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
    ]
    missing = [name for name in required if not (cause_dir / name).exists()]
    if missing:
        print(f"Missing cause-analysis outputs in {cause_dir}: {missing}", file=sys.stderr)
        return 2

    output_dir = ensure_dir(args.output_dir)
    cause = pd.read_csv(cause_dir / "group_cause_profile.csv", encoding="utf-8-sig")
    modality = pd.read_csv(cause_dir / "group_unimodal_fusion_profile.csv", encoding="utf-8-sig")
    oracle = pd.read_csv(cause_dir / "rd_nd_oracle_profile.csv", encoding="utf-8-sig")
    class_prior = pd.read_csv(cause_dir / "group_class_prior_control.csv", encoding="utf-8-sig")
    class_wise = pd.read_csv(cause_dir / "class_wise_accuracy.csv", encoding="utf-8-sig")
    polarity_corr = pd.read_csv(cause_dir / "d_polarity_correlation.csv", encoding="utf-8-sig")
    polarity_bin = pd.read_csv(cause_dir / "polarity_bin_d_control.csv", encoding="utf-8-sig")
    decoupled_eval = pd.read_csv(cause_dir / "decoupled_d_group_eval.csv", encoding="utf-8-sig")
    decoupled_contrast = pd.read_csv(
        cause_dir / "decoupled_high_d_contrast.csv",
        encoding="utf-8-sig",
    )
    method = pd.read_csv(cause_dir / "method_insufficiency_1_15.csv", encoding="utf-8-sig")
    summary = load_json(cause_dir / "v6_cause_analysis_summary.json")

    experiment1 = build_experiment1_table(cause, modality, summary)
    cause_table = build_cause_table(cause, modality)
    rd_nd_table = build_rd_nd_table(oracle)
    method_table = build_method_table(method)
    class_prior_table = build_class_prior_table(class_prior, modality)
    class_wise_table = build_class_wise_table(class_wise)
    polarity_corr_table = build_polarity_correlation_table(polarity_corr)
    polarity_bin_table = build_polarity_bin_table(polarity_bin)
    decoupled_table = build_decoupled_table(decoupled_eval, decoupled_contrast)

    write_table(experiment1, output_dir, "table1_experiment1_concat_disagreement")
    write_table(cause_table, output_dir, "table2_high_d_cause_profile")
    write_table(rd_nd_table, output_dir, "table3_rd_nd_oracle_profile")
    write_table(method_table, output_dir, "table4_method_insufficiency")
    write_table(class_prior_table, output_dir, "table5_class_prior_control")
    write_table(class_wise_table, output_dir, "table6_class_wise_accuracy")
    write_table(polarity_corr_table, output_dir, "table7_d_polarity_correlation")
    write_table(polarity_bin_table, output_dir, "table8_polarity_bin_d_control")
    write_table(decoupled_table, output_dir, "table9_cross_seed_decoupled_d")
    write_report(
        output_dir,
        experiment1,
        cause_table,
        rd_nd_table,
        method_table,
        class_prior_table,
        class_wise_table,
        polarity_corr_table,
        polarity_bin_table,
        decoupled_table,
    )

    print(f"Saved v6 motivation tables to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
