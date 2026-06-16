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
    """数据集配置：名称、数据目录名及 npz 文件名（不可变）。"""

    name: str
    folder: str
    filename: str


DATASETS = {
    "mosi": DatasetConfig(
        name="mosi",
        folder="mosi",
        filename="mosi_aligned.npz",
    ),
    "mosei": DatasetConfig(
        name="mosei",
        folder="mosei",
        filename="mosei_aligned.npz",
    ),
}


@dataclass
class ExperimentConfig:
    """实验配置数据中心，涵盖数据、模型、训练、损失及分析的所有超参数。

    Attributes:
        dataset: 数据集名称（"mosi" 或 "mosei"）。
        data_root: 数据根目录。
        output_root: 输出根目录。
        seed: 随机种子。
        batch_size: 批次大小。
        epochs: 最大训练轮数。
        lr: 学习率。
        weight_decay: AdamW 权重衰减。
        hidden_dim: 共享隐层维度。
        dropout: Dropout 比率。
        eta_unimodal: 单模态分类损失权重。
        label_mode: 标签模式（"three_class" 或 "binary"）。
        lambda_align_values: 对齐损失权重的搜索列表。
        direct_add_alpha_values: Direct Add 注入强度的搜索列表。
        pair_mode: 通用模态对模式。
        align_pair_mode: 对齐损失的模态对模式。
        direct_add_pair_mode: Direct Add 的模态对模式。
        run_infonce: 是否运行 InfoNCE 实验。
        lambda_nce_values: InfoNCE 损失权重的搜索列表。
        nce_temperature: InfoNCE 温度参数。
        nce_pair_mode: InfoNCE 的模态对模式。
        use_nce_projection: 是否使用 InfoNCE 投影头。
        nce_proj_dim: InfoNCE 投影维度。
        disagreement_metric: 分歧度量方式（"prob_jsd" 或 "kernel"）。
        disagreement_pair_mode: 分歧计算的模态对模式。
        kernel_bandwidth: 核分歧的带宽（"median" 或数值）。
        kernel_pair_mode: 核分歧的模态对模式。
        kernel_class_weight: 核分歧中类别 MMD 的权重。
        kernel_max_class_samples: 类别 MMD 中每类最大采样数。
        relation_split: 关系状态划分方式（"balanced_within_d" 等）。
        residual_modes: 残差探针模式列表。
        tau_agreement: 一致性转换的温度参数。
        patience: 早停耐心值。
        num_workers: DataLoader 工作进程数。
        deterministic: 是否启用确定性训练。
        quiet: 是否静默模式（不显示 tqdm 进度条）。
    """

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
    eta_unimodal: float = 0.1
    label_mode: str = "three_class"
    lambda_align_values: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1]
    )
    direct_add_alpha_values: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 1.0]
    )
    pair_mode: str = "text_anchor"
    align_pair_mode: str = "text_anchor"
    direct_add_pair_mode: str = "text_anchor"
    run_infonce: bool = False
    lambda_nce_values: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5])
    nce_temperature: float = 0.1
    nce_pair_mode: str = "text_anchor"
    use_nce_projection: bool = True
    nce_proj_dim: int = 128
    disagreement_metric: str = "prob_jsd"
    disagreement_pair_mode: str = "text_anchor"
    kernel_bandwidth: str = "median"
    kernel_pair_mode: str = "text_anchor"
    kernel_class_weight: float = 0.5
    kernel_max_class_samples: int = 1024
    relation_split: str = "balanced_within_d"
    residual_modes: list[str] = field(default_factory=lambda: ["abs", "signed", "prod", "all"])
    tau_agreement: float = 0.1
    patience: int = 8
    num_workers: int = 0
    deterministic: bool = False
    quiet: bool = False

    @property
    def dataset_config(self) -> DatasetConfig:
        """获取数据集配置对象。

        Raises:
            ValueError: 数据集名称不合法时抛出。
        """
        if self.dataset not in DATASETS:
            valid = ", ".join(sorted(DATASETS))
            raise ValueError(f"Unknown dataset '{self.dataset}'. Valid choices: {valid}")
        return DATASETS[self.dataset]

    @property
    def data_path(self) -> Path:
        """npz 数据文件的完整路径。"""
        ds = self.dataset_config
        return self.data_root / ds.folder / ds.filename

    @property
    def num_classes(self) -> int:
        """根据 label_mode 返回类别数（3 或 2）。

        Raises:
            ValueError: label_mode 不合法时抛出。
        """
        if self.label_mode == "three_class":
            return 3
        if self.label_mode == "binary":
            return 2
        raise ValueError("label_mode must be 'three_class' or 'binary'.")
