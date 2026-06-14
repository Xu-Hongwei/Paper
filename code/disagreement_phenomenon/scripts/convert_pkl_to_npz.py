from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_OUTPUTS = {
    "mosi": Path(r"E:\Xu\data\MultiBench\mosi\mosi_aligned.npz"),
    "mosei": Path(r"E:\Xu\data\MultiBench\mosei\mosei_aligned.npz"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a processed CMU-MOSI/MOSEI pickle with train/valid/test "
            "splits into the .npz format expected by run_phenomenon.py."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .pkl file.")
    parser.add_argument(
        "--dataset",
        choices=["mosi", "mosei"],
        required=True,
        help="Used only to choose the default output path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to E:\\Xu\\data\\MultiBench\\<dataset>\\*_aligned.npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output .npz if it already exists.",
    )
    parser.add_argument(
        "--label_index",
        type=int,
        default=0,
        help="Label column to use when labels are multi-dimensional. Default: 0.",
    )
    return parser.parse_args()


def as_float32(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(array, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def labels_1d(array: np.ndarray, label_index: int = 0) -> np.ndarray:
    labels = np.squeeze(as_float32(array))
    if labels.ndim == 0:
        return labels.reshape(1)
    if labels.ndim == 1:
        return labels
    return labels[:, label_index]


def require_keys(split_name: str, split: dict) -> None:
    required = ["text", "vision", "audio", "labels"]
    missing = [key for key in required if key not in split]
    if missing:
        raise KeyError(f"Split '{split_name}' is missing keys: {', '.join(missing)}")


def convert(
    input_path: Path,
    output_path: Path,
    overwrite: bool,
    label_index: int = 0,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to rebuild it.")

    with input_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level dict, got {type(data)!r}")

    exported: dict[str, np.ndarray] = {}
    for split_name in ("train", "valid", "test"):
        if split_name not in data:
            raise KeyError(f"Input pickle is missing split '{split_name}'")
        split = data[split_name]
        if not isinstance(split, dict):
            raise TypeError(f"Split '{split_name}' should be dict, got {type(split)!r}")
        require_keys(split_name, split)

        exported[f"{split_name}_text"] = as_float32(split["text"])
        exported[f"{split_name}_vision"] = as_float32(split["vision"])
        exported[f"{split_name}_audio"] = as_float32(split["audio"])
        exported[f"{split_name}_label"] = labels_1d(
            split["labels"],
            label_index=label_index,
        )

        n = exported[f"{split_name}_label"].shape[0]
        for modality in ("text", "vision", "audio"):
            key = f"{split_name}_{modality}"
            if exported[key].shape[0] != n:
                raise ValueError(
                    f"{key} has {exported[key].shape[0]} samples but "
                    f"{split_name}_label has {n} samples."
                )
        print(
            f"{split_name}: n={n}, "
            f"text={exported[f'{split_name}_text'].shape}, "
            f"vision={exported[f'{split_name}_vision'].shape}, "
            f"audio={exported[f'{split_name}_audio'].shape}, "
            f"label={exported[f'{split_name}_label'].shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **exported)
    print(f"Saved: {output_path}")


def main() -> int:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.dataset]
    convert(
        args.input,
        output,
        overwrite=args.overwrite,
        label_index=args.label_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
