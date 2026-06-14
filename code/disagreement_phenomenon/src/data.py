from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import REQUIRED_NPZ_KEYS


def sentiment_to_three_class(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    classes = np.ones_like(labels, dtype=np.int64)
    classes[labels < -0.5] = 0
    classes[labels > 0.5] = 2
    return classes


def describe_expected_npz(path: Path) -> str:
    keys = "\n  - ".join(REQUIRED_NPZ_KEYS)
    return (
        f"Missing data file: {path}\n\n"
        "Expected one aligned .npz feature file with these arrays:\n"
        f"  - {keys}\n\n"
        "Labels should be continuous sentiment scores in [-3, 3]. "
        "The code will convert them to negative/neutral/positive classes."
    )


def validate_npz(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(describe_expected_npz(path))
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
    if missing:
        raise KeyError(
            f"{path} is missing required arrays: {', '.join(missing)}\n"
            f"Required arrays: {', '.join(REQUIRED_NPZ_KEYS)}"
        )


@dataclass
class SplitArrays:
    text: np.ndarray
    vision: np.ndarray
    audio: np.ndarray
    label_reg: np.ndarray
    label_cls: np.ndarray


class MultimodalSplitDataset(Dataset):
    def __init__(self, split: SplitArrays) -> None:
        self.text = torch.as_tensor(split.text, dtype=torch.float32)
        self.vision = torch.as_tensor(split.vision, dtype=torch.float32)
        self.audio = torch.as_tensor(split.audio, dtype=torch.float32)
        self.label_reg = torch.as_tensor(split.label_reg, dtype=torch.float32).view(-1)
        self.label_cls = torch.as_tensor(split.label_cls, dtype=torch.long).view(-1)

    def __len__(self) -> int:
        return int(self.label_cls.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "text": self.text[index],
            "vision": self.vision[index],
            "audio": self.audio[index],
            "label_reg": self.label_reg[index],
            "label_cls": self.label_cls[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def _load_split(data: np.lib.npyio.NpzFile, split: str) -> SplitArrays:
    text = np.asarray(data[f"{split}_text"], dtype=np.float32)
    vision = np.asarray(data[f"{split}_vision"], dtype=np.float32)
    audio = np.asarray(data[f"{split}_audio"], dtype=np.float32)
    label_reg = np.asarray(data[f"{split}_label"], dtype=np.float32).reshape(-1)
    label_cls = sentiment_to_three_class(label_reg)

    n = label_reg.shape[0]
    for name, array in (("text", text), ("vision", vision), ("audio", audio)):
        if array.shape[0] != n:
            raise ValueError(
                f"{split}_{name} has {array.shape[0]} samples but "
                f"{split}_label has {n} samples."
            )
    return SplitArrays(text, vision, audio, label_reg, label_cls)


def load_npz_splits(path: Path) -> dict[str, SplitArrays]:
    validate_npz(path)
    with np.load(path, allow_pickle=False) as data:
        return {split: _load_split(data, split) for split in ("train", "valid", "test")}


def infer_input_dims(split: SplitArrays) -> dict[str, int]:
    def last_dim(array: np.ndarray) -> int:
        if array.ndim < 2:
            raise ValueError(f"Feature array must be [N, D] or [N, T, D], got {array.shape}")
        return int(array.shape[-1])

    return {
        "text": last_dim(split.text),
        "vision": last_dim(split.vision),
        "audio": last_dim(split.audio),
    }
