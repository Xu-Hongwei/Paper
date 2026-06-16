from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .disagreement import GROUP_ORDER, HIGH_D_RELIABILITY_GROUP_ORDER, RELATION_STATE_GROUP_ORDER


def save_delta_plot(delta_df: pd.DataFrame, path: Path) -> None:
    """保存按分歧分组的分组柱状图（单次实验版）。

    Args:
        delta_df: 包含 "group" 和 "delta_macro_f1" 列的 DataFrame。
        path: 图片保存路径。
    """
    plot_df = delta_df[delta_df["group"].isin(GROUP_ORDER)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values("group")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(plot_df["group"].astype(str), plot_df["delta_macro_f1"], color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta Macro-F1 (UncondAlign - Concat)")
    ax.set_xlabel("Disagreement group")
    ax.set_title("Unconditional alignment gain by disagreement group")
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


def save_multi_seed_delta_plot(summary_df: pd.DataFrame, path: Path) -> None:
    """保存按分歧分组并带误差棒的柱状图（多 seed 汇总版）。

    优先使用 95% 置信区间作为误差棒，回退到标准差。

    Args:
        summary_df: 多 seed 汇总 DataFrame，需含 "group" 和 "delta_macro_f1_mean"
                    及 "delta_macro_f1_ci95_high" 或 "delta_macro_f1_std"。
        path: 图片保存路径。
    """
    plot_df = summary_df[summary_df["group"].isin(GROUP_ORDER)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values("group")

    err_col = (
        "delta_macro_f1_ci95_high"
        if "delta_macro_f1_ci95_high" in plot_df.columns
        else "delta_macro_f1_std"
    )
    if err_col == "delta_macro_f1_ci95_high":
        yerr = (
            plot_df["delta_macro_f1_ci95_high"] - plot_df["delta_macro_f1_mean"]
        ).fillna(0.0)
        ylabel = "Delta Macro-F1 mean +/- 95% CI"
    else:
        yerr = plot_df["delta_macro_f1_std"].fillna(0.0)
        ylabel = "Delta Macro-F1 mean +/- std"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        plot_df["group"].astype(str),
        plot_df["delta_macro_f1_mean"],
        yerr=yerr,
        capsize=4,
        color="#4C78A8",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Disagreement group")
    ax.set_title("Multi-seed unconditional alignment gain")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reliability_delta_plot(summary_df: pd.DataFrame, path: Path) -> None:
    """保存 High-D 可靠性子组的柱状图（High-D+Low-R vs High-D+High-R）。

    Args:
        summary_df: 汇总 DataFrame，需含 "group" 和 "delta_macro_f1_mean"
                    及误差列。
        path: 图片保存路径。
    """
    if "delta_macro_f1_ci95_high" in summary_df.columns:
        yerr = (
            summary_df["delta_macro_f1_ci95_high"] - summary_df["delta_macro_f1_mean"]
        ).fillna(0.0)
        ylabel = "Delta Macro-F1 mean +/- 95% CI"
    else:
        yerr = summary_df["delta_macro_f1_std"].fillna(0.0)
        ylabel = "Delta Macro-F1 mean +/- std"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary_df["group"].astype(str),
        summary_df["delta_macro_f1_mean"],
        yerr=yerr,
        capsize=4,
        color="#F58518",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("High-D reliability group")
    ax.set_title("High-D reliability split")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _summary_error(summary_df: pd.DataFrame, value_col: str) -> pd.Series:
    """从汇总 DataFrame 中提取误差棒值。

    优先使用 95% CI 半宽，回退到标准差，否则返回 0。

    Args:
        summary_df: 汇总 DataFrame。
        value_col: 指标列名前缀（如 "delta_macro_f1"）。

    Returns:
        误差值 Series。
    """
    mean_col = f"{value_col}_mean"
    ci_high_col = f"{value_col}_ci95_high"
    std_col = f"{value_col}_std"
    if ci_high_col in summary_df.columns:
        return (summary_df[ci_high_col] - summary_df[mean_col]).abs().fillna(0.0)
    if std_col in summary_df.columns:
        return summary_df[std_col].fillna(0.0)
    return pd.Series(0.0, index=summary_df.index)


def save_detailed_delta_plot(
    all_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
    *,
    group_order: tuple[str, ...] = GROUP_ORDER,
    value_col: str = "delta_macro_f1",
    title: str,
    ylabel: str = "Delta Macro-F1",
) -> None:
    """保存详细的分组柱状图，含误差棒、散点、标注（多 seed 综合版）。

    柱色：绿色 = 通过 error control，蓝色 = 未通过。
    每个柱上标注均值、seed 数、方向一致性比例，以及散点表示各 seed 值。

    Args:
        all_df: 所有 seed 的明细 DataFrame，每行一个 seed×group。
        summary_df: 跨 seed 汇总 DataFrame。
        path: 图片保存路径。
        group_order: 分组显示顺序。
        value_col: 指标列名前缀。
        title: 图表标题。
        ylabel: Y 轴标签。
    """
    plot_all = all_df[all_df["group"].isin(group_order)].copy()
    plot_summary = summary_df[summary_df["group"].isin(group_order)].copy()
    plot_summary["group"] = pd.Categorical(plot_summary["group"], group_order, ordered=True)
    plot_summary = plot_summary.sort_values("group")
    if plot_summary.empty:
        return

    mean_col = f"{value_col}_mean"
    count_col = f"{value_col}_count"
    sign_col = f"{value_col}_sign_consistency"
    pass_col = f"{value_col}_passes_error_control"
    pos_col = f"{value_col}_positive_rate"
    neg_col = f"{value_col}_negative_rate"

    x_lookup = {group: idx for idx, group in enumerate(plot_summary["group"].astype(str))}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [
        "#54A24B" if bool(row.get(pass_col, False)) else "#4C78A8"
        for _, row in plot_summary.iterrows()
    ]
    yerr = _summary_error(plot_summary, value_col)
    bars = ax.bar(
        range(len(plot_summary)),
        plot_summary[mean_col],
        yerr=yerr,
        capsize=5,
        color=colors,
        alpha=0.82,
        edgecolor="#222222",
        linewidth=0.8,
        label="mean +/- 95% CI",
    )

    if not plot_all.empty:
        for _, row in plot_all.iterrows():
            group = row["group"]
            if group not in x_lookup or pd.isna(row.get(value_col)):
                continue
            seed = int(row["seed"]) if "seed" in row and not pd.isna(row["seed"]) else 0
            jitter = ((seed * 37) % 17 - 8) / 55.0
            ax.scatter(
                x_lookup[group] + jitter,
                row[value_col],
                color="#222222",
                alpha=0.58,
                s=26,
                zorder=3,
            )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(range(len(plot_summary)))
    ax.set_xticklabels(plot_summary["group"].astype(str))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.28)

    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    for bar, (_, row) in zip(bars, plot_summary.iterrows()):
        mean = row.get(mean_col)
        count = int(row.get(count_col, 0)) if not pd.isna(row.get(count_col)) else 0
        sign = row.get(sign_col, np.nan)
        pos_rate = row.get(pos_col, np.nan)
        neg_rate = row.get(neg_col, np.nan)
        passed = bool(row.get(pass_col, False))
        direction = "+"
        if not pd.isna(pos_rate) and not pd.isna(neg_rate):
            direction = "+" if pos_rate >= neg_rate else "-"
        label = f"{mean:+.3f}\nn={count}, {direction}{sign:.0%}" if not pd.isna(sign) else f"{mean:+.3f}\nn={count}"
        if passed:
            label += "\nEC"
        y = bar.get_height()
        offset = 0.025 * span if y >= 0 else -0.025 * span
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            label,
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
        )

    ax.text(
        0.99,
        0.02,
        "black dots = individual seeds; EC = passes error control",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_method_relation_state_heatmap(
    summaries: dict[str, pd.DataFrame],
    path: Path,
    *,
    group_order: tuple[str, ...] = RELATION_STATE_GROUP_ORDER,
    value_col: str = "delta_macro_f1",
    title: str = "Relation-state delta comparison",
) -> None:
    """保存多方法×关系状态的热力图。

    行 = 方法，列 = 关系状态（RA/UA/Mid-D/RD/ND），颜色用 RdBu_r 表示 delta 值。
    每个单元格标注均值、95% CI、符号一致性比例。

    Args:
        summaries: {方法名: 汇总 DataFrame} 的字典。
        path: 图片保存路径。
        group_order: 列（关系状态）的显示顺序。
        value_col: 指标列名前缀。
        title: 图表标题。
    """
    methods = [method for method, frame in summaries.items() if frame is not None and not frame.empty]
    if not methods:
        return

    matrix = np.full((len(methods), len(group_order)), np.nan, dtype=float)
    annotations: list[list[str]] = [["" for _ in group_order] for _ in methods]
    for row_idx, method in enumerate(methods):
        frame = summaries[method]
        by_group = {row["group"]: row for _, row in frame.iterrows()}
        for col_idx, group in enumerate(group_order):
            row = by_group.get(group)
            if row is None:
                continue
            mean = row.get(f"{value_col}_mean", np.nan)
            ci_low = row.get(f"{value_col}_ci95_low", np.nan)
            ci_high = row.get(f"{value_col}_ci95_high", np.nan)
            sign = row.get(f"{value_col}_sign_consistency", np.nan)
            passed = bool(row.get(f"{value_col}_passes_error_control", False))
            matrix[row_idx, col_idx] = mean
            if pd.isna(mean):
                annotations[row_idx][col_idx] = "NA"
            else:
                ci_text = ""
                if not pd.isna(ci_low) and not pd.isna(ci_high):
                    ci_text = f"\n[{ci_low:+.2f},{ci_high:+.2f}]"
                sign_text = f"\n{sign:.0%}" if not pd.isna(sign) else ""
                annotations[row_idx][col_idx] = f"{mean:+.3f}{ci_text}{sign_text}{'*' if passed else ''}"

    finite = matrix[np.isfinite(matrix)]
    vmax = max(abs(finite).max(), 0.01) if finite.size else 0.01
    fig, ax = plt.subplots(figsize=(1.35 * len(group_order) + 2.5, 1.05 * len(methods) + 2.4))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(title)

    for row_idx in range(len(methods)):
        for col_idx in range(len(group_order)):
            value = matrix[row_idx, col_idx]
            color = "white" if np.isfinite(value) and abs(value) > vmax * 0.55 else "black"
            ax.text(
                col_idx,
                row_idx,
                annotations[row_idx][col_idx],
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Delta Macro-F1 mean")
    ax.text(
        0.99,
        -0.18,
        "* = passes error control; cell lines show mean, 95% CI, sign consistency",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_lambda_curve_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str = "Lambda alignment strength curve",
    x_col: str = "lambda_align",
    x_label: str = "lambda_align",
    y_label: str = "Delta Macro-F1 (UncondAlign - Concat)",
    raw_frame: pd.DataFrame | None = None,
) -> None:
    """保存 λ 对齐强度曲线图（X 轴对数刻度）。

    每条线对应一个分歧分组，可叠加各 seed 的原始折线作为背景。

    Args:
        frame: 汇总 DataFrame，需含 group、x_col、value_col 及误差列。
        path: 图片保存路径。
        title: 图表标题。
        x_col: X 轴列名。
        x_label: X 轴标签。
        y_label: Y 轴标签。
        raw_frame: 可选，各 seed 的原始数据，用于绘制背景灰线。
    """
    plot_df = frame[frame["group"].isin(GROUP_ORDER)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["group", x_col])

    value_col = (
        "delta_macro_f1_mean"
        if "delta_macro_f1_mean" in plot_df.columns
        else "delta_macro_f1"
    )
    if "delta_macro_f1_ci95_high" in plot_df.columns:
        err_col = "ci95"
    elif "delta_macro_f1_std" in plot_df.columns:
        err_col = "std"
    else:
        err_col = None

    fig, ax = plt.subplots(figsize=(7, 4))
    for group in GROUP_ORDER:
        group_df = plot_df[plot_df["group"] == group].sort_values(x_col)
        if group_df.empty:
            continue
        if raw_frame is not None and not raw_frame.empty:
            raw_group = raw_frame[raw_frame["group"] == group].copy()
            if not raw_group.empty and {"seed", x_col, "delta_macro_f1"}.issubset(raw_group.columns):
                for _, seed_df in raw_group.groupby("seed"):
                    seed_df = seed_df.sort_values(x_col)
                    ax.plot(
                        seed_df[x_col],
                        seed_df["delta_macro_f1"],
                        color="#999999",
                        alpha=0.22,
                        linewidth=0.9,
                        zorder=1,
                    )
        if err_col == "ci95":
            yerr = (group_df["delta_macro_f1_ci95_high"] - group_df[value_col]).fillna(0.0)
        elif err_col == "std":
            yerr = group_df["delta_macro_f1_std"].fillna(0.0)
        else:
            yerr = None
        ax.errorbar(
            group_df[x_col],
            group_df[value_col],
            yerr=yerr,
            marker="o",
            capsize=3 if err_col else 0,
            label=group,
            zorder=3,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
