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
            "high_d_reliability_group",
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
        print(f"Smoke test passed. Outputs checked in {latest}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
