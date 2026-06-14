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
