from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_NPZ_KEYS = (
    "train_text",
    "train_vision",
    "train_audio",
    "train_label",
    "valid_text",
    "valid_vision",
    "valid_audio",
    "valid_label",
    "test_text",
    "test_vision",
    "test_audio",
    "test_label",
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    folder: str
    filename: str


DATASETS = {
    "mosi": DatasetConfig(
        name="mosi",
        folder="CMU_MOSI",
        filename="mosi_aligned.npz",
    ),
    "mosei": DatasetConfig(
        name="mosei",
        folder="CMU_MOSEI",
        filename="mosei_aligned.npz",
    ),
}


@dataclass
class ExperimentConfig:
    dataset: str
    data_root: Path
    output_root: Path
    seed: int = 42
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.2
    eta_unimodal: float = 0.3
    lambda_align_values: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.5]
    )
    patience: int = 8
    num_workers: int = 0

    @property
    def dataset_config(self) -> DatasetConfig:
        if self.dataset not in DATASETS:
            valid = ", ".join(sorted(DATASETS))
            raise ValueError(f"Unknown dataset '{self.dataset}'. Valid choices: {valid}")
        return DATASETS[self.dataset]

    @property
    def data_path(self) -> Path:
        ds = self.dataset_config
        return self.data_root / ds.folder / ds.filename

