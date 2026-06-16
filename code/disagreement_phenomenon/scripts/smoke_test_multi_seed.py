from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_multi_seed.py"


def make_fixture(path: Path) -> None:
    rng = np.random.default_rng(11)

    def split(n: int) -> dict[str, np.ndarray]:
        text = rng.normal(size=(n, 5, 12)).astype("float32")
        vision = rng.normal(size=(n, 4, 8)).astype("float32")
        audio = rng.normal(size=(n, 6)).astype("float32")
        signal = text.mean(axis=(1, 2)) + 0.7 * vision.mean(axis=(1, 2)) + 0.4 * audio.mean(axis=1)
        label = np.clip(signal * 2.0, -3.0, 3.0).astype("float32")
        return {
            "text": text,
            "vision": vision,
            "audio": audio,
            "label": label,
        }

    train = split(72)
    valid = split(36)
    test = split(36)
    np.savez(
        path,
        train_text=train["text"],
        train_vision=train["vision"],
        train_audio=train["audio"],
        train_label=train["label"],
        valid_text=valid["text"],
        valid_vision=valid["vision"],
        valid_audio=valid["audio"],
        valid_label=valid["label"],
        test_text=test["text"],
        test_vision=test["vision"],
        test_audio=test["audio"],
        test_label=test["label"],
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "mosi"
        data_dir.mkdir(parents=True)
        make_fixture(data_dir / "mosi_aligned.npz")
        output_root = root / "outputs"

        command = [
            sys.executable,
            "-B",
            str(RUNNER),
            "--dataset",
            "mosi",
            "--data_root",
            str(root),
            "--output_root",
            str(output_root),
            "--seeds",
            "1",
            "2",
            "--epochs",
            "1",
            "--batch_size",
            "16",
            "--hidden_dim",
            "16",
            "--lambda_align_values",
            "0.01",
            "--direct_add_alpha_values",
            "0.1",
            "--run_infonce",
            "--lambda_nce_values",
            "0.01",
            "--pair_mode",
            "text_anchor",
            "--relation_split",
            "balanced_within_d",
            "--disagreement_metric",
            "kernel_mmd",
            "--kernel_max_class_samples",
            "32",
            "--run_kernel_dist_diagnostic",
            "--kernel_dist_min_group_size",
            "4",
            "--patience",
            "1",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode

        runs = list((output_root / "mosi").glob("multi_seed_*"))
        if not runs:
            print("Multi-seed smoke test failed: no multi_seed output.", file=sys.stderr)
            return 1
        latest = max(runs, key=lambda path: path.stat().st_mtime)
        summary = latest / "summary"
        required = [
            "multi_seed_delta_summary.csv",
            "multi_seed_group_metrics_summary.csv",
            "direct_add_delta_all.csv",
            "direct_add_delta_summary.csv",
            "direct_add_alpha_test_delta_all.csv",
            "direct_add_alpha_test_delta_summary.csv",
            "balanced_direct_add_delta_all.csv",
            "balanced_direct_add_delta_summary.csv",
            "balanced_direct_add_alpha_test_delta_all.csv",
            "balanced_direct_add_alpha_test_delta_summary.csv",
            "high_d_reliability_summary.csv",
            "relation_state_delta_all.csv",
            "relation_state_delta_summary.csv",
            "uncond_align_relation_delta_all.csv",
            "uncond_align_relation_delta_summary.csv",
            "relation_state_metrics_all.csv",
            "relation_state_metrics_summary.csv",
            "relation_state_distribution_calibration_all.csv",
            "relation_state_distribution_calibration_summary.csv",
            "kernel_distribution_relation_metrics_all.csv",
            "kernel_distribution_relation_summary_all.csv",
            "kernel_distribution_relation_summary.csv",
            "direct_add_relation_state_delta_all.csv",
            "direct_add_relation_state_delta_summary.csv",
            "balanced_direct_add_relation_state_delta_all.csv",
            "balanced_direct_add_relation_state_delta_summary.csv",
            "concat_aware_motivation_all.csv",
            "concat_aware_motivation_summary.csv",
            "residual_discriminative_probe_all.csv",
            "residual_discriminative_probe_summary.csv",
            "residual_probe_by_mode_all.csv",
            "residual_probe_by_mode_summary.csv",
            "lambda_test_delta_all.csv",
            "lambda_test_delta_summary.csv",
            "lambda_high_d_reliability_delta_all.csv",
            "lambda_high_d_reliability_summary.csv",
            "infonce_delta_all.csv",
            "infonce_delta_summary.csv",
            "infonce_high_d_reliability_delta_all.csv",
            "infonce_high_d_reliability_summary.csv",
            "infonce_relation_state_delta_all.csv",
            "infonce_relation_state_delta_summary.csv",
            "infonce_relation_delta_all.csv",
            "infonce_relation_delta_summary.csv",
            "infonce_lambda_test_delta_all.csv",
            "infonce_lambda_test_delta_summary.csv",
            "infonce_lambda_high_d_reliability_delta_all.csv",
            "infonce_lambda_high_d_reliability_summary.csv",
            "multi_seed_delta_macro_f1.png",
            "multi_seed_delta_macro_f1_detailed.png",
            "high_d_reliability_delta.png",
            "high_d_reliability_delta_detailed.png",
            "relation_state_delta_detailed.png",
            "direct_add_relation_state_delta_detailed.png",
            "balanced_direct_add_relation_state_delta_detailed.png",
            "lambda_delta_macro_f1_curve.png",
            "infonce_lambda_delta_macro_f1_curve.png",
            "infonce_delta_macro_f1_detailed.png",
            "infonce_high_d_reliability_delta_detailed.png",
            "infonce_relation_state_delta_detailed.png",
            "relation_state_method_comparison_heatmap.png",
            "experiment_one_conclusion.json",
            "error_control_report.csv",
        ]
        missing = [name for name in required if not (summary / name).exists()]
        if missing:
            print(f"Multi-seed smoke test failed: missing outputs {missing}", file=sys.stderr)
            return 1
        delta_summary = pd.read_csv(summary / "multi_seed_delta_summary.csv")
        required_columns = {
            "delta_macro_f1_sem",
            "delta_macro_f1_ci95_low",
            "delta_macro_f1_ci95_high",
            "delta_macro_f1_positive_rate",
            "delta_macro_f1_negative_rate",
            "delta_macro_f1_sign_consistency",
            "delta_macro_f1_passes_error_control",
        }
        missing_columns = sorted(required_columns - set(delta_summary.columns))
        if missing_columns:
            print(
                f"Multi-seed smoke test failed: missing error-control columns {missing_columns}",
                file=sys.stderr,
            )
            return 1
        residual_summary = pd.read_csv(summary / "residual_discriminative_probe_summary.csv")
        residual_columns = {
            "text_anchor_residual_only_macro_f1_mean",
            "text_anchor_common_residual_macro_f1_mean",
            "text_anchor_residual_gain_macro_f1_mean",
            "text_anchor_shuffled_residual_macro_f1_mean",
            "common_shuffled_residual_macro_f1_mean",
            "residual_gain_vs_feature_shuffle_macro_f1_mean",
            "text_anchor_common_shuffled_residual_macro_f1_mean",
            "text_anchor_residual_gain_vs_feature_shuffle_macro_f1_mean",
        }
        missing_residual = sorted(residual_columns - set(residual_summary.columns))
        if missing_residual:
            print(
                f"Multi-seed smoke test failed: missing text-anchor residual summary columns {missing_residual}",
                file=sys.stderr,
            )
            return 1
        infonce_summary = pd.read_csv(summary / "infonce_lambda_test_delta_summary.csv")
        if infonce_summary.empty or "lambda_nce" not in infonce_summary.columns:
            print("Multi-seed smoke test failed: InfoNCE summary is empty.", file=sys.stderr)
            return 1
        infonce_config_columns = {
            "nce_pair_mode",
            "nce_temperature",
            "use_nce_projection",
            "nce_proj_dim",
        }
        missing_infonce_config = sorted(infonce_config_columns - set(infonce_summary.columns))
        if missing_infonce_config:
            print(
                f"Multi-seed smoke test failed: missing InfoNCE config columns {missing_infonce_config}",
                file=sys.stderr,
            )
            return 1
        calibration_summary = pd.read_csv(
            summary / "relation_state_distribution_calibration_summary.csv"
        )
        calibration_columns = {
            "avg_R_mean",
            "text_acc_mean",
            "audio_acc_mean",
            "vision_acc_mean",
            "fusion_acc_mean",
        }
        missing_calibration = sorted(calibration_columns - set(calibration_summary.columns))
        if missing_calibration:
            print(
                f"Multi-seed smoke test failed: missing calibration summary columns {missing_calibration}",
                file=sys.stderr,
            )
            return 1
        kernel_summary = pd.read_csv(summary / "kernel_distribution_relation_summary.csv")
        kernel_columns = {
            "D_dist_text_anchor_mean",
            "D_dist_full_pair_mean",
            "mmd_ta_mean",
            "mmd_tv_mean",
            "avg_R_mean",
            "avg_D_sample_mean",
        }
        missing_kernel = sorted(kernel_columns - set(kernel_summary.columns))
        if missing_kernel:
            print(
                f"Multi-seed smoke test failed: missing kernel distribution columns {missing_kernel}",
                file=sys.stderr,
            )
            return 1
        kernel_all = pd.read_csv(summary / "kernel_distribution_relation_summary_all.csv")
        if kernel_all.empty or set(kernel_all["pair_mode"]) != {"text_anchor"}:
            print(
                "Multi-seed smoke test failed: kernel distribution all-seed pair mode is wrong.",
                file=sys.stderr,
            )
            return 1
        residual_by_mode = pd.read_csv(summary / "residual_probe_by_mode_summary.csv")
        if residual_by_mode.empty or "matched_residual_gain_macro_f1_mean" not in residual_by_mode:
            print(
                "Multi-seed smoke test failed: residual by-mode summary is incomplete.",
                file=sys.stderr,
            )
            return 1
        align_all = pd.read_csv(summary / "lambda_test_delta_all.csv")
        if align_all.empty or set(align_all["align_pair_mode"]) != {"text_anchor"}:
            print(
                "Multi-seed smoke test failed: UncondAlign all-seed pair mode is wrong.",
                file=sys.stderr,
            )
            return 1
        direct_all = pd.read_csv(summary / "direct_add_alpha_test_delta_all.csv")
        if direct_all.empty or set(direct_all["direct_add_pair_mode"]) != {"text_anchor"}:
            print(
                "Multi-seed smoke test failed: DirectAdd all-seed pair mode is wrong.",
                file=sys.stderr,
            )
            return 1
        balanced_all = pd.read_csv(summary / "balanced_direct_add_alpha_test_delta_all.csv")
        if balanced_all.empty or set(balanced_all["direct_add_pair_mode"]) != {"balanced"}:
            print(
                "Multi-seed smoke test failed: BalancedDirectAdd all-seed mode is wrong.",
                file=sys.stderr,
            )
            return 1
        group_summary = pd.read_csv(summary / "multi_seed_group_metrics_summary.csv")
        if "BalancedDirectAdd" not in set(group_summary["method"]):
            print(
                "Multi-seed smoke test failed: BalancedDirectAdd summary row is missing.",
                file=sys.stderr,
            )
            return 1
        print(f"Multi-seed smoke test passed. Outputs checked in {summary}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
