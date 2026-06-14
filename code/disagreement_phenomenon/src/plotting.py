from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .disagreement import GROUP_ORDER


def save_delta_plot(delta_df: pd.DataFrame, path: Path) -> None:
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
    plot_df = summary_df[summary_df["group"].isin(GROUP_ORDER)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values("group")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        plot_df["group"].astype(str),
        plot_df["delta_macro_f1_mean"],
        yerr=plot_df["delta_macro_f1_std"].fillna(0.0),
        capsize=4,
        color="#4C78A8",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta Macro-F1 mean +/- std")
    ax.set_xlabel("Disagreement group")
    ax.set_title("Multi-seed unconditional alignment gain")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_reliability_delta_plot(summary_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary_df["group"].astype(str),
        summary_df["delta_macro_f1_mean"],
        yerr=summary_df["delta_macro_f1_std"].fillna(0.0),
        capsize=4,
        color="#F58518",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta Macro-F1 mean +/- std")
    ax.set_xlabel("High-D reliability group")
    ax.set_title("High-D reliability split")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_lambda_curve_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str = "Lambda alignment strength curve",
) -> None:
    plot_df = frame[frame["group"].isin(GROUP_ORDER)].copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["group", "lambda_align"])

    value_col = (
        "delta_macro_f1_mean"
        if "delta_macro_f1_mean" in plot_df.columns
        else "delta_macro_f1"
    )
    std_col = "delta_macro_f1_std" if "delta_macro_f1_std" in plot_df.columns else None

    fig, ax = plt.subplots(figsize=(7, 4))
    for group in GROUP_ORDER:
        group_df = plot_df[plot_df["group"] == group].sort_values("lambda_align")
        if group_df.empty:
            continue
        yerr = group_df[std_col].fillna(0.0) if std_col else None
        ax.errorbar(
            group_df["lambda_align"],
            group_df[value_col],
            yerr=yerr,
            marker="o",
            capsize=3 if std_col else 0,
            label=group,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("lambda_align")
    ax.set_ylabel("Delta Macro-F1 (UncondAlign - Concat)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
