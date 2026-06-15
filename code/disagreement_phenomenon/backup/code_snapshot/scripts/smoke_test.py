from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_phenomenon.py"


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
            "--nce_pair_mode",
            "text_anchor",
            "--disagreement_metric",
            "kernel_mmd",
            "--kernel_pair_mode",
            "text_anchor",
            "--kernel_max_class_samples",
            "32",
            "--run_copa",
            "--lambda_copa_values",
            "0.01",
            "--copa_gate_type",
            "label_support",
            "--copa_orth_weight",
            "0.01",
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
            "train_label_aware_relations.csv",
            "valid_label_aware_relations.csv",
            "label_aware_relation_summary.csv",
            "copa_delta_metrics.csv",
            "copa_high_d_reliability_delta.csv",
            "copa_lambda_sweep_valid.csv",
            "copa_lambda_test_delta_metrics.csv",
            "copa_lambda_high_d_reliability_delta.csv",
            "high_d_reliability_metrics.csv",
            "high_d_reliability_delta.csv",
            "relation_state_metrics.csv",
            "relation_state_delta.csv",
            "direct_add_alpha_sweep_valid.csv",
            "direct_add_alpha_test_delta_metrics.csv",
            "direct_add_delta_metrics.csv",
            "direct_add_relation_state_delta.csv",
            "infonce_lambda_sweep_valid.csv",
            "infonce_lambda_test_delta_metrics.csv",
            "infonce_delta_metrics.csv",
            "infonce_high_d_reliability_delta.csv",
            "infonce_lambda_high_d_reliability_delta.csv",
            "infonce_relation_state_delta.csv",
            "concat_aware_motivation.csv",
            "feature_consistency_diagnostic.csv",
            "residual_distribution_diagnostic.csv",
            "residual_discriminative_probe.csv",
            "selective_agreement_prototype_check.csv",
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
        label_aware = pd.read_csv(latest / "train_label_aware_relations.csv")
        label_aware_columns = {
            "C_text",
            "S_text",
            "R_label_text",
            "R_label_sample",
            "g_tv_agr",
            "g_tv_comp",
            "g_tv_noise",
            "disagreement_metric",
            "kernel_pair_mode",
        }
        missing_label_aware = sorted(label_aware_columns - set(label_aware.columns))
        if missing_label_aware:
            print(
                f"Smoke test failed: missing label-aware columns {missing_label_aware}",
                file=sys.stderr,
            )
            return 1
        if set(label_aware["disagreement_metric"]) != {"kernel_mmd"}:
            print(
                "Smoke test failed: label-aware relation frame did not use kernel_mmd.",
                file=sys.stderr,
            )
            return 1
        infonce_sweep = pd.read_csv(latest / "infonce_lambda_sweep_valid.csv")
        if infonce_sweep.empty or set(infonce_sweep["nce_pair_mode"]) != {"text_anchor"}:
            print(
                "Smoke test failed: InfoNCE text_anchor sweep is empty or has wrong pair mode.",
                file=sys.stderr,
            )
            return 1
        residual_probe = pd.read_csv(latest / "residual_discriminative_probe.csv")
        residual_columns = {
            "text_anchor_residual_only_macro_f1",
            "text_anchor_common_residual_macro_f1",
            "text_anchor_residual_gain_macro_f1",
            "text_anchor_shuffled_residual_macro_f1",
        }
        missing_residual = sorted(residual_columns - set(residual_probe.columns))
        if missing_residual:
            print(
                f"Smoke test failed: missing text-anchor residual columns {missing_residual}",
                file=sys.stderr,
            )
            return 1
        print(f"Smoke test passed. Outputs checked in {latest}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
