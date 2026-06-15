# CoPA Experiment 1: MOSEI 20-Seed Report

生成时间：2026-06-15

## 1. 运行配置

本次运行命令等价于：

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py `
  --dataset mosei `
  --data_root E:\Xu\data\MultiBench `
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 `
  --batch_size 1024 `
  --num_workers 0 `
  --epochs 50 `
  --patience 6 `
  --quiet `
  --run_copa `
  --deterministic
```

关键设置：

- Dataset: MOSEI
- Seeds: 1-20
- Epochs: max 50
- Early stopping patience: 6
- Deterministic: True
- Batch size: 1024
- CoPA enabled: True
- Error control: `min_seeds=5`, `same-sign rate >= 0.8`, `95% CI` 不跨 0

输出目录：

```text
code/disagreement_phenomenon/outputs/mosei/multi_seed_20260615_115853/summary
```

## 2. Overall 结果

| Method | Overall Macro-F1 mean | 95% CI | Acc mean |
|---|---:|---:|---:|
| Concat | 0.6126 | [0.6105, 0.6147] | 0.6123 |
| UncondAlign | 0.6139 | [0.6119, 0.6159] | 0.6136 |
| DirectAdd | 0.6134 | [0.6116, 0.6153] | 0.6137 |
| CoPA | 0.6164 | [0.6147, 0.6180] | 0.6159 |

相对 Concat 的 Overall Delta Macro-F1：

| Method | Delta Macro-F1 mean | 95% CI | 同方向比例 | Error Control |
|---|---:|---:|---:|---|
| UncondAlign | +0.0013 | [-0.0013, +0.0039] | 60% positive | False |
| DirectAdd | +0.0008 | [-0.0017, +0.0034] | 60% positive | False |
| CoPA | +0.0038 | [+0.0011, +0.0065] | 85% positive | True |

结论：CoPA 的 Overall 提升较小，但在 20 seeds 下通过误差控制。UncondAlign 和 DirectAdd 的 Overall 提升没有通过误差控制。

## 3. Disagreement Group 结果

### Unconditional Alignment vs Concat

| Group | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---|---:|---:|---:|---|
| Low-D | +0.0004 | [-0.0060, +0.0067] | 50% | False |
| Mid-D | +0.0027 | [-0.0015, +0.0069] | 60% | False |
| High-D | -0.0025 | [-0.0072, +0.0021] | 50% | False |
| Overall | +0.0013 | [-0.0013, +0.0039] | 60% | False |

按验证集自动选择 lambda 后，UncondAlign 的整体分组效果不稳定。

### DirectAdd vs Concat

| Group | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---|---:|---:|---:|---|
| Low-D | +0.0024 | [-0.0042, +0.0089] | 50% | False |
| Mid-D | +0.0017 | [-0.0030, +0.0064] | 65% | False |
| High-D | -0.0025 | [-0.0064, +0.0014] | 30% | False |
| Overall | +0.0008 | [-0.0017, +0.0034] | 60% | False |

DirectAdd 没有稳定替代关系条件建模。

### CoPA vs Concat

| Group | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---|---:|---:|---:|---|
| Low-D | +0.0088 | [+0.0019, +0.0156] | 65% | False |
| Mid-D | +0.0035 | [-0.0007, +0.0077] | 70% | False |
| High-D | -0.0005 | [-0.0042, +0.0031] | 40% | False |
| Overall | +0.0038 | [+0.0011, +0.0065] | 85% | True |

CoPA 的主要稳定收益来自 Overall；Low-D 均值和 CI 为正，但同方向比例未达到 0.8，因此按当前误差控制规则不算通过。

## 4. Relation-State 结果

### CoPA Relation-State Delta

| State | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---|---:|---:|---:|---|
| RA | -0.0035 | [-0.0107, +0.0037] | 50% | False |
| UA | +0.0127 | [+0.0051, +0.0203] | 70% | False |
| Mid-D | +0.0035 | [-0.0007, +0.0077] | 70% | False |
| RD | +0.0001 | [-0.0029, +0.0031] | 40% | False |
| ND | -0.0021 | [-0.0169, +0.0127] | 45% | False |

Interpretation:

- UA 上有明显正均值和正 CI，但同方向比例只有 70%，还不够稳。
- RD 基本贴近 0，没有稳定正收益。
- ND 方差很大，不能下强结论。

这说明当前 CoPA prototype 的 relation-state 效果还没有完全打出来；论文中可以保留 relation-state 动机，但不能声称 CoPA 已经稳定提升 RD。

## 5. 固定强度分析

自动选超参会引入选择噪声，因此还需要看固定强度曲线。

### UncondAlign 固定 lambda

最关键发现：

| lambda_align | Group | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---:|---|---:|---:|---:|---|
| 0.01 | High-D | -0.0049 | [-0.0080, -0.0018] | 20% positive | True |

结论：固定 `lambda_align=0.01` 时，无条件对齐会稳定伤害 High-D。这是实验一动机中最强的证据之一。

### DirectAdd 固定 alpha

| alpha | Group | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---:|---|---:|---:|---:|---|
| 0.1 | High-D | -0.0044 | [-0.0073, -0.0015] | 15% positive | True |
| 0.3 | High-D | -0.0039 | [-0.0077, -0.0002] | 20% positive | True |

结论：DirectAdd 在较小 alpha 下也会稳定伤害 High-D，说明“原表示 + 直接相加对齐摘要”不足以替代关系条件拆分。

### CoPA 固定 lambda

CoPA 固定 lambda 下没有分组通过 error control。`lambda_copa=0.1` 的 Overall 有正 CI：

| lambda_copa | Group | Delta Macro-F1 mean | 95% CI | Positive rate | Error Control |
|---:|---|---:|---:|---:|---|
| 0.1 | Overall | +0.0027 | [+0.0001, +0.0053] | 65% positive | False |

因为同方向比例不足 0.8，所以不算通过当前误差控制。

## 6. 超参选择稳定性

最佳 `lambda_copa` 分布：

| lambda_copa | Count |
|---:|---:|
| 0.01 | 5 |
| 0.05 | 9 |
| 0.1 | 6 |

最佳 `lambda_align` 分布：

| lambda_align | Count |
|---:|---:|
| 0.001 | 5 |
| 0.005 | 9 |
| 0.01 | 3 |
| 0.05 | 2 |
| 0.1 | 1 |

最佳 DirectAdd `alpha` 分布：

| alpha | Count |
|---:|---:|
| 0.1 | 5 |
| 0.3 | 7 |
| 0.5 | 5 |
| 1.0 | 3 |

结论：DirectAdd 和 CoPA 的强度选择仍有波动；报告最终实验时应同时给出 fixed-strength 与 validation-selected 两类结果。

## 7. 图表文件

建议优先看：

```text
multi_seed_delta_macro_f1_detailed.png
copa_delta_macro_f1_detailed.png
relation_state_method_comparison_heatmap.png
lambda_delta_macro_f1_curve.png
copa_lambda_delta_macro_f1_curve.png
direct_add_relation_state_delta_detailed.png
```

所有图表都在：

```text
code/disagreement_phenomenon/outputs/mosei/multi_seed_20260615_115853/summary
```

## 8. 当前论文结论建议

可以稳妥写：

1. Concat 能学习标签边界，但 relation-state 仍揭示不同样本关系下的对齐风险。
2. Unconditional alignment 在固定中等强度时会稳定伤害 High-D。
3. DirectAdd 在 High-D 上也可能稳定受损，说明直接加回对齐摘要不是充分的“无损对齐”。
4. 当前 CoPA prototype 在 Overall 上取得小幅但通过误差控制的提升。

不建议强写：

1. CoPA 已经稳定提升 RD。
2. CoPA 在所有 relation states 上都优于 Concat。
3. DirectAdd 完全无效。

更准确的表述是：

> Experiment 1 shows that unconditional or direct alignment is not uniformly beneficial and can reliably harm high-disagreement samples under fixed alignment strength. The current CoPA prototype gives a small but error-controlled overall gain, while stronger relation-state-specific gains, especially on RD, require further method refinement.

## 9. 运行警告

运行中 residual probe 的 sklearn Logistic Regression 仍出现 `lbfgs max_iter=500` 收敛警告。它影响 residual probe 诊断数值，不影响 Concat / UncondAlign / DirectAdd / CoPA 主模型训练结果。
