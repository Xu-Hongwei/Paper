from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "scripts" / "run_phenomenon.py"

from src.model import unconditional_alignment_loss, unconditional_infonce_loss  # noqa: E402


def make_fixture(path: Path) -> None:
    rng = np.random.default_rng(7)

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


def check_projection_loss_boundary() -> None:
    torch.manual_seed(3)
    outputs = {
        "h_t": torch.randn(6, 5),
        "h_a": torch.randn(6, 5),
        "h_v": torch.randn(6, 5),
    }
    outputs["z_t"] = torch.randn(6, 7)
    outputs["z_a"] = torch.randn(6, 7)
    outputs["z_v"] = torch.randn(6, 7)
    align_loss = unconditional_alignment_loss(outputs, pair_mode="text_anchor")
    changed_z = dict(outputs)
    changed_z["z_t"] = outputs["z_t"] * 13.0
    changed_z["z_a"] = outputs["z_a"] * -5.0
    changed_z["z_v"] = outputs["z_v"] + 9.0
    changed_align_loss = unconditional_alignment_loss(changed_z, pair_mode="text_anchor")
    if not torch.allclose(align_loss, changed_align_loss):
        raise AssertionError("UncondAlign must use h_* hidden states, not z_* projections.")
    infonce_loss = unconditional_infonce_loss(outputs, temperature=0.1, pair_mode="text_anchor")
    changed_infonce_loss = unconditional_infonce_loss(
        changed_z,
        temperature=0.1,
        pair_mode="text_anchor",
    )
    if torch.allclose(infonce_loss, changed_infonce_loss):
        raise AssertionError("UncondInfoNCE must use z_* projections when they are present.")


def main() -> int:
    check_projection_loss_boundary()
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
            "--epochs",
            "2",
            "--batch_size",
            "16",
            "--hidden_dim",
            "16",
            "--lambda_align_values",
            "0.01",
            "0.05",
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
            "--patience",
            "2",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return result.returncode

        runs = list((output_root / "mosi").glob("*"))
        if not runs:
            print("Smoke test failed: no output run directory was created.", file=sys.stderr)
            return 1
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        required = [
            "test_groups.csv",
            "group_metrics.csv",
            "delta_metrics.csv",
            "high_d_reliability_metrics.csv",
            "high_d_reliability_delta.csv",
            "relation_state_metrics.csv",
            "relation_state_delta.csv",
            "relation_state_distribution_calibration.csv",
            "uncond_align_relation_delta.csv",
            "direct_add_alpha_sweep_valid.csv",
            "direct_add_alpha_test_delta_metrics.csv",
            "direct_add_delta_metrics.csv",
            "direct_add_relation_state_delta.csv",
            "balanced_direct_add_alpha_sweep_valid.csv",
            "balanced_direct_add_alpha_test_delta_metrics.csv",
            "balanced_direct_add_delta_metrics.csv",
            "balanced_direct_add_relation_state_delta.csv",
            "balanced_direct_add_model.pt",
            "infonce_lambda_sweep_valid.csv",
            "infonce_lambda_test_delta_metrics.csv",
            "infonce_delta_metrics.csv",
            "infonce_high_d_reliability_delta.csv",
            "infonce_lambda_high_d_reliability_delta.csv",
            "infonce_relation_state_delta.csv",
            "infonce_relation_delta.csv",
            "concat_aware_motivation.csv",
            "residual_discriminative_probe.csv",
            "residual_probe_by_mode.csv",
            "lambda_test_delta_metrics.csv",
            "lambda_high_d_reliability_delta.csv",
            "delta_macro_f1.png",
            "lambda_delta_macro_f1_curve.png",
            "summary.json",
        ]
        missing = [name for name in required if not (latest / name).exists()]
        if missing:
            print(f"Smoke test failed: missing outputs {missing}", file=sys.stderr)
            return 1
        groups = pd.read_csv(latest / "test_groups.csv")
        required_columns = {
            "R_text",
            "R_vision",
            "R_audio",
            "R_tv",
            "R_ta",
            "R_va",
            "R_sample",
            "D_tv",
            "D_ta",
            "D_va",
            "A_tv",
            "A_ta",
            "A_va",
            "g_tv_agr",
            "g_tv_comp",
            "g_tv_noise",
            "high_d_reliability_group",
            "relation_state",
            "relation_state_desc",
            "pair_mode",
            "disagreement_pair_mode",
            "kernel_pair_mode",
            "label_mode",
            "relation_split",
        }
        missing_columns = sorted(required_columns - set(groups.columns))
        if missing_columns:
            print(
                f"Smoke test failed: missing test_groups columns {missing_columns}",
                file=sys.stderr,
            )
            return 1
        high_d_groups = set(groups.loc[groups["group"] == "High-D", "high_d_reliability_group"])
        if not high_d_groups.intersection({"High-D+Low-R", "High-D+High-R"}):
            print("Smoke test failed: High-D reliability split is empty.", file=sys.stderr)
            return 1
        relation_states = set(groups["relation_state"])
        if not relation_states.intersection({"RA", "UA", "RD", "ND"}):
            print("Smoke test failed: relation-state split is empty.", file=sys.stderr)
            return 1
        if set(groups["disagreement_pair_mode"]) != {"text_anchor"}:
            print(
                "Smoke test failed: disagreement_pair_mode was not recorded as text_anchor.",
                file=sys.stderr,
            )
            return 1
        if set(groups["pair_mode"]) != {"text_anchor"}:
            print("Smoke test failed: pair_mode was not recorded as text_anchor.", file=sys.stderr)
            return 1
        infonce_sweep = pd.read_csv(latest / "infonce_lambda_sweep_valid.csv")
        if infonce_sweep.empty or set(infonce_sweep["nce_pair_mode"]) != {"text_anchor"}:
            print(
                "Smoke test failed: InfoNCE text_anchor sweep is empty or has wrong pair mode.",
                file=sys.stderr,
            )
            return 1
        infonce_projection_columns = {"use_nce_projection", "nce_proj_dim"}
        missing_infonce_projection = sorted(
            infonce_projection_columns - set(infonce_sweep.columns)
        )
        if missing_infonce_projection:
            print(
                f"Smoke test failed: missing InfoNCE projection columns {missing_infonce_projection}",
                file=sys.stderr,
            )
            return 1
        align_sweep = pd.read_csv(latest / "lambda_sweep_valid.csv")
        if align_sweep.empty or set(align_sweep["align_pair_mode"]) != {"text_anchor"}:
            print(
                "Smoke test failed: UncondAlign sweep is empty or has wrong pair mode.",
                file=sys.stderr,
            )
            return 1
        direct_sweep = pd.read_csv(latest / "direct_add_alpha_sweep_valid.csv")
        if direct_sweep.empty or set(direct_sweep["direct_add_pair_mode"]) != {"text_anchor"}:
            print(
                "Smoke test failed: DirectAdd sweep is empty or has wrong pair mode.",
                file=sys.stderr,
            )
            return 1
        balanced_sweep = pd.read_csv(latest / "balanced_direct_add_alpha_sweep_valid.csv")
        if balanced_sweep.empty or set(balanced_sweep["direct_add_pair_mode"]) != {"balanced"}:
            print(
                "Smoke test failed: BalancedDirectAdd sweep is empty or has wrong mode.",
                file=sys.stderr,
            )
            return 1
        group_metrics = pd.read_csv(latest / "group_metrics.csv")
        if "BalancedDirectAdd" not in set(group_metrics["method"]):
            print("Smoke test failed: BalancedDirectAdd method row is missing.", file=sys.stderr)
            return 1
        residual_probe = pd.read_csv(latest / "residual_discriminative_probe.csv")
        residual_columns = {
            "text_anchor_residual_only_macro_f1",
            "text_anchor_common_residual_macro_f1",
            "text_anchor_residual_gain_macro_f1",
            "text_anchor_shuffled_residual_macro_f1",
            "common_shuffled_residual_macro_f1",
            "residual_gain_vs_feature_shuffle_macro_f1",
            "text_anchor_common_shuffled_residual_macro_f1",
            "text_anchor_residual_gain_vs_feature_shuffle_macro_f1",
        }
        missing_residual = sorted(residual_columns - set(residual_probe.columns))
        if missing_residual:
            print(
                f"Smoke test failed: missing text-anchor residual columns {missing_residual}",
                file=sys.stderr,
            )
            return 1
        calibration = pd.read_csv(latest / "relation_state_distribution_calibration.csv")
        calibration_columns = {
            "group",
            "n",
            "label_mode",
            "relation_split",
            "text_acc",
            "audio_acc",
            "vision_acc",
            "fusion_acc",
            "avg_R",
        }
        missing_calibration = sorted(calibration_columns - set(calibration.columns))
        if missing_calibration:
            print(
                f"Smoke test failed: missing calibration columns {missing_calibration}",
                file=sys.stderr,
            )
            return 1
        residual_by_mode = pd.read_csv(latest / "residual_probe_by_mode.csv")
        if residual_by_mode.empty or not {"abs", "signed", "prod", "all"}.issubset(
            set(residual_by_mode["residual_mode"])
        ):
            print("Smoke test failed: residual by-mode table is incomplete.", file=sys.stderr)
            return 1
        binary_command = [
            *command,
            "--label_mode",
            "binary",
        ]
        binary_result = subprocess.run(binary_command, text=True, capture_output=True)
        print(binary_result.stdout)
        if binary_result.returncode != 0:
            print(binary_result.stderr, file=sys.stderr)
            return binary_result.returncode
        binary_runs = list((output_root / "mosi").glob("*"))
        binary_latest = max(binary_runs, key=lambda p: p.stat().st_mtime)
        binary_calibration = pd.read_csv(
            binary_latest / "relation_state_distribution_calibration.csv"
        )
        if "class_1_ratio" not in binary_calibration.columns:
            print("Smoke test failed: binary calibration lacks class_1_ratio.", file=sys.stderr)
            return 1
        print(f"Smoke test passed. Outputs checked in {latest}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
