from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import REQUIRED_NPZ_KEYS


def sentiment_to_three_class(labels: np.ndarray) -> np.ndarray:
    """将连续情感分数映射为三分类标签。

    映射规则：< -0.5 → 0 (negative)；[-0.5, 0.5] → 1 (neutral)；> 0.5 → 2 (positive)。

    Args:
        labels: 连续情感分数，shape (N,)，典型范围为 [-3, 3]。

    Returns:
        三分类标签数组，dtype int64，shape (N,)。
    """
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    classes = np.ones_like(labels, dtype=np.int64)
    classes[labels < -0.5] = 0
    classes[labels > 0.5] = 2
    return classes


def sentiment_to_binary(labels: np.ndarray) -> np.ndarray:
    """将连续情感分数映射为二分类标签。

    映射规则：<= 0 → 0 (non-positive)；> 0 → 1 (positive)。

    Args:
        labels: 连续情感分数，shape (N,)。

    Returns:
        二分类标签数组，dtype int64，shape (N,)。
    """
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    return (labels > 0.0).astype(np.int64)


def sentiment_to_class(labels: np.ndarray, label_mode: str) -> np.ndarray:
    """根据 label_mode 将连续情感分数转换为分类标签。

    Args:
        labels: 连续情感分数。
        label_mode: "three_class" 或 "binary"。

    Returns:
        分类标签数组。

    Raises:
        ValueError: label_mode 不合法时抛出。
    """
    if label_mode == "three_class":
        return sentiment_to_three_class(labels)
    if label_mode == "binary":
        return sentiment_to_binary(labels)
    raise ValueError("label_mode must be 'three_class' or 'binary'.")


def describe_expected_npz(path: Path) -> str:
    """生成描述预期 npz 文件格式的帮助信息。

    Args:
        path: 期望的数据文件路径。

    Returns:
        格式化的帮助字符串。
    """
    keys = "\n  - ".join(REQUIRED_NPZ_KEYS)
    return (
        f"Missing data file: {path}\n\n"
        "Expected one aligned .npz feature file with these arrays:\n"
        f"  - {keys}\n\n"
        "Labels should be continuous sentiment scores in [-3, 3]. "
        "The code will convert them to negative/neutral/positive classes."
    )


def validate_npz(path: Path) -> None:
    """验证 npz 文件是否存在且包含所有必需的数组键。

    Args:
        path: 数据文件路径。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        KeyError: 缺少必需数组时抛出。
    """
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
    """单个数据划分的 numpy 数组容器。

    Attributes:
        text: 文本特征，shape (N, D_t) 或 (N, T_t, D_t)。
        vision: 视觉特征，shape (N, D_v) 或 (N, T_v, D_v)。
        audio: 音频特征，shape (N, D_a) 或 (N, T_a, D_a)。
        label_reg: 连续情感回归标签，shape (N,)。
        label_cls: 离散分类标签，shape (N,)。
    """

    text: np.ndarray
    vision: np.ndarray
    audio: np.ndarray
    label_reg: np.ndarray
    label_cls: np.ndarray


class MultimodalSplitDataset(Dataset):
    """PyTorch Dataset，将 SplitArrays 转为 tensor 格式。

    每个样本返回包含 text/vision/audio 特征、回归标签、分类标签及索引的字典。
    """

    def __init__(self, split: SplitArrays) -> None:
        """初始化 Dataset。

        Args:
            split: 包含各模态 numpy 数组的 SplitArrays 对象。
        """
        self.text = torch.as_tensor(split.text, dtype=torch.float32)
        self.vision = torch.as_tensor(split.vision, dtype=torch.float32)
        self.audio = torch.as_tensor(split.audio, dtype=torch.float32)
        self.label_reg = torch.as_tensor(split.label_reg, dtype=torch.float32).view(-1)
        self.label_cls = torch.as_tensor(split.label_cls, dtype=torch.long).view(-1)

    def __len__(self) -> int:
        """返回数据集样本数。"""
        return int(self.label_cls.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """获取单个样本。

        Args:
            index: 样本索引。

        Returns:
            dict，包含 text, vision, audio, label_reg, label_cls, index。
        """
        return {
            "text": self.text[index],
            "vision": self.vision[index],
            "audio": self.audio[index],
            "label_reg": self.label_reg[index],
            "label_cls": self.label_cls[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def _load_split(data: np.lib.npyio.NpzFile, split: str, label_mode: str) -> SplitArrays:
    """从 npz 文件中加载单个数据划分（train/valid/test）。

    验证各模态样本数与标签样本数一致。

    Args:
        data: 已打开的 npz 文件对象。
        split: 划分名称（"train"/"valid"/"test"）。
        label_mode: 标签模式。

    Returns:
        SplitArrays 对象。

    Raises:
        ValueError: 模态样本数与标签数不匹配时抛出。
    """
    text = np.asarray(data[f"{split}_text"], dtype=np.float32)
    vision = np.asarray(data[f"{split}_vision"], dtype=np.float32)
    audio = np.asarray(data[f"{split}_audio"], dtype=np.float32)
    label_reg = np.asarray(data[f"{split}_label"], dtype=np.float32).reshape(-1)
    label_cls = sentiment_to_class(label_reg, label_mode)

    n = label_reg.shape[0]
    for name, array in (("text", text), ("vision", vision), ("audio", audio)):
        if array.shape[0] != n:
            raise ValueError(
                f"{split}_{name} has {array.shape[0]} samples but "
                f"{split}_label has {n} samples."
            )
    return SplitArrays(text, vision, audio, label_reg, label_cls)


def load_npz_splits(path: Path, *, label_mode: str = "three_class") -> dict[str, SplitArrays]:
    """加载并验证 npz 文件，返回 train/valid/test 三个划分。

    Args:
        path: npz 文件路径。
        label_mode: 标签模式。

    Returns:
        dict，键为 "train"/"valid"/"test"，值为对应的 SplitArrays。
    """
    validate_npz(path)
    with np.load(path, allow_pickle=False) as data:
        return {
            split: _load_split(data, split, label_mode)
            for split in ("train", "valid", "test")
        }


def infer_input_dims(split: SplitArrays) -> dict[str, int]:
    """从 SplitArrays 推断三模态的输入维度（取最后一维）。

    Args:
        split: 已加载的 SplitArrays。

    Returns:
        dict，如 {"text": 300, "vision": 35, "audio": 74}。

    Raises:
        ValueError: 特征数组维度不足 2 时抛出。
    """
    def last_dim(array: np.ndarray) -> int:
        if array.ndim < 2:
            raise ValueError(f"Feature array must be [N, D] or [N, T, D], got {array.shape}")
        return int(array.shape[-1])

    return {
        "text": last_dim(split.text),
        "vision": last_dim(split.vision),
        "audio": last_dim(split.audio),
    }
