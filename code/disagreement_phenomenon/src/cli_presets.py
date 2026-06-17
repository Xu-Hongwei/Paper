from __future__ import annotations

import argparse
from collections.abc import Sequence


PRESET_CHOICES = ("none", "v6_motivation", "v6_pilot", "appendix_full", "smoke")


def requested_preset(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="none")
    args, _ = parser.parse_known_args(argv)
    return args.preset


def preset_defaults(preset: str) -> dict[str, object]:
    if preset == "none":
        return {}
    if preset == "v6_motivation":
        return {
            "batch_size": 1024,
            "num_workers": 0,
            "epochs": 25,
            "patience": 6,
            "quiet": True,
            "deterministic": True,
            "run_infonce": True,
            "run_dynamic_fusion": True,
            "run_rc_balanced_add": False,
            "run_residual_probe": False,
            "run_kernel_dist_diagnostic": False,
            "pair_mode": "text_anchor",
            "relation_split": "balanced_within_d",
        }
    if preset == "v6_pilot":
        values = preset_defaults("v6_motivation")
        values.update(
            {
                "run_rc_balanced_add": True,
                "rc_balanced_modes": ["rd_only", "hard"],
            }
        )
        return values
    if preset == "appendix_full":
        values = preset_defaults("v6_pilot")
        values.update(
            {
                "run_residual_probe": True,
                "run_kernel_dist_diagnostic": True,
            }
        )
        return values
    if preset == "smoke":
        return {
            "batch_size": 64,
            "num_workers": 0,
            "epochs": 1,
            "patience": 1,
            "quiet": True,
            "run_infonce": True,
            "run_dynamic_fusion": True,
            "run_rc_balanced_add": True,
            "rc_balanced_modes": ["rd_only", "hard"],
            "run_residual_probe": True,
            "run_kernel_dist_diagnostic": True,
            "kernel_dist_min_group_size": 4,
            "lambda_align_values": [0.01],
            "lambda_nce_values": [0.01],
            "lambda_dynamic_weight_values": [0.01],
            "direct_add_alpha_values": [0.1],
            "pair_mode": "text_anchor",
            "relation_split": "balanced_within_d",
        }
    raise ValueError(f"Unknown preset: {preset}")
