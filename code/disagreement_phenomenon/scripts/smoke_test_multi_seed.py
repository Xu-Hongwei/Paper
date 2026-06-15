from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


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
            "--run_copa",
            "--lambda_copa_values",
            "0.01",
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
            "high_d_reliability_summary.csv",
            "relation_state_delta_all.csv",
            "relation_state_delta_summary.csv",
            "relation_state_metrics_all.csv",
            "relation_state_metrics_summary.csv",
            "direct_add_relation_state_delta_all.csv",
            "direct_add_relation_state_delta_summary.csv",
            "concat_aware_motivation_all.csv",
            "concat_aware_motivation_summary.csv",
            "feature_consistency_diagnostic_all.csv",
            "feature_consistency_diagnostic_summary.csv",
            "residual_distribution_diagnostic_all.csv",
            "residual_distribution_diagnostic_summary.csv",
            "residual_discriminative_probe_all.csv",
            "residual_discriminative_probe_summary.csv",
            "selective_agreement_prototype_check_all.csv",
            "selective_agreement_prototype_check_summary.csv",
            "label_aware_relation_summary_all.csv",
            "label_aware_relation_multi_seed_summary.csv",
            "lambda_test_delta_all.csv",
            "lambda_test_delta_summary.csv",
            "lambda_high_d_reliability_delta_all.csv",
            "lambda_high_d_reliability_summary.csv",
            "copa_delta_all.csv",
            "copa_delta_summary.csv",
            "copa_high_d_reliability_delta_all.csv",
            "copa_high_d_reliability_summary.csv",
            "copa_relation_state_delta_all.csv",
            "copa_relation_state_delta_summary.csv",
            "copa_lambda_test_delta_all.csv",
            "copa_lambda_test_delta_summary.csv",
            "copa_lambda_high_d_reliability_delta_all.csv",
            "copa_lambda_high_d_reliability_summary.csv",
            "multi_seed_delta_macro_f1.png",
            "high_d_reliability_delta.png",
            "lambda_delta_macro_f1_curve.png",
            "copa_lambda_delta_macro_f1_curve.png",
            "experiment_one_conclusion.json",
        ]
        missing = [name for name in required if not (summary / name).exists()]
        if missing:
            print(f"Multi-seed smoke test failed: missing outputs {missing}", file=sys.stderr)
            return 1
        print(f"Multi-seed smoke test passed. Outputs checked in {summary}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
