# CoPA-v6 实验计划：Relation-Conditioned Balanced Utilization

## 0. 实验总目标

v6 实验不再证明“分歧有害，所以要对齐”，而是证明：

```text
分歧有结构；
高分歧不等于困难；
统一对齐/普通对比/动态权重都不充分；
可靠高分歧更适合被平衡利用。
```

最终目标是验证一个新的方法方向：

```text
Relation-Conditioned Balanced Utilization
```

即根据 RA/UA/RD/ND 等关系状态决定模态交互方式。

## 1. 当前结果基线

必须区分三个结果来源：

| Run | Seeds | 用途 |
|---|---:|---|
| `multi_seed_20260616_111835` | 1-10 | 主诊断，10 个独立 seed |
| `multi_seed_20260616_151132` | 1-5 | InfoNCE 补充 |
| `multi_seed_20260616_160036` | 1-5 | 复跑 + DynamicFusion |

注意：

```text
151132 和 160036 都是 seeds 1-5；
160036 不是新的 11-15 seed；
它证明可复现，并新增 DynamicFusion 结果，但不增加独立 seed 数。
```

如果要真正形成 15 independent seeds，需要补跑：

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py `
  --dataset mosei `
  --data_root E:\Xu\data\MultiBench `
  --seeds 11 12 13 14 15 `
  --batch_size 1024 `
  --num_workers 0 `
  --epochs 25 `
  --patience 6 `
  --lr 1e-3 `
  --weight_decay 1e-4 `
  --dropout 0.2 `
  --deterministic `
  --run_infonce `
  --run_kernel_dist_diagnostic `
  --kernel_dist_min_group_size 10 `
  --quiet
```

## 2. v6 实验问题

v6 要回答 5 个问题：

1. 高分歧是否稳定不是困难样本？
2. 高分歧是否需要 reliability 来进一步解释？
3. 无条件对齐、普通 InfoNCE、DynamicFusion 是否都不充分？
4. BalancedDirectAdd 为什么在 RD 上更有效？
5. relation-conditioned balanced 方法能否超过固定 BalancedDirectAdd？

## 3. 实验 1：Disagreement Is Not Difficulty

### 目的

证明：

```text
High-D 不是困难样本。
```

### 表格

主表使用 Concat baseline：

| Dataset | Seeds | Low-D | Mid-D | High-D | Seed Consistency |
|---|---:|---:|---:|---:|---|
| MOSEI | 10 | 0.5029 | 0.5985 | 0.6830 | 10/10 |

### 结论模板

中文：

> 在 MOSEI 的 10 个独立随机种子中，基础融合模型在 High-D 样本上的表现始终高于 Mid-D 和 Low-D。这说明跨模态分歧高并不意味着样本更难或质量更低。

英文：

> Across 10 independent seeds on MOSEI, the baseline fusion model consistently performs best on High-D samples, indicating that high cross-modal disagreement does not necessarily imply difficulty or low quality.

## 4. 实验 2：Reliability Disambiguates Disagreement

### 目的

证明：

```text
High-D 内部不是同一种状态。
```

### 使用表

使用：

```text
relation_state_distribution_calibration_summary.csv
kernel_distribution_relation_summary.csv
```

### 报告指标

| Group | N | avg_D_sample | avg_R | Text Acc | Audio Acc | Vision Acc | Fusion Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| RA |  |  |  |  |  |  |  |
| UA |  |  |  |  |  |  |  |
| RD |  |  |  |  |  |  |  |
| ND |  |  |  |  |  |  |  |

### 目前观察

- RD 的 `avg_D_sample` 和 `avg_R` 都最高；
- RD 的 fusion accuracy 最高；
- UA 的 `avg_R` 和 fusion accuracy 最低；
- ND 虽然低可靠，但 fusion accuracy 也不低，说明 reliability 不是简单质量分数。

### v6 解释

```text
Reliability 不是质量标签，而是预测确定性结构。
```

## 5. 实验 3：Uniform Alignment Is Insufficient

### 目的

证明：

```text
统一对齐不是主解法。
```

### 对比方法

| Method | 解释 |
|---|---|
| Concat | 基础拼接融合 |
| UncondAlign | 无条件拉近同样本跨模态 hidden state |
| UncondInfoNCE | 同样本跨模态正样本，batch 内负样本 |

### 需要报告

| Method | Overall | Low-D | Mid-D | High-D | RA | UA | RD | ND |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UncondAlign - Concat |  |  |  |  |  |  |  |  |
| InfoNCE - Concat |  |  |  |  |  |  |  |  |

### 当前结论

- UncondAlign 很弱；
- InfoNCE 很弱；
- 强 alignment / 强 nce 有负面信号；
- 因此“同一样本跨模态一定是好正样本”的假设过强。

## 6. 实验 4：Dynamic Weighting Is Not Enough

### 目的

证明：

```text
只做动态权重分配，也不能解决可靠高分歧。
```

### 当前结果

最新 5-seed DynamicFusion：

| Group | Delta Macro-F1 |
|---|---:|
| Low-D | +0.0052 |
| Mid-D | +0.0086 |
| High-D | -0.0029 |
| RD | -0.0053 |
| Overall | +0.0035 |

### 解释

DynamicFusion 有整体收益，但没有处理好 RD。它可能更像“模态选择”，而不是“互补利用”。

## 7. 实验 5：Balanced Utilization Helps RD

### 目的

证明：

```text
可靠高分歧更适合被平衡利用。
```

### 当前最强证据

MOSEI 10-seed：

```text
BalancedDirectAdd on RD:
Delta Macro-F1 = +0.0095
95% CI = [0.0027, 0.0163]
```

### 表格

| Method | RD Delta Macro-F1 | Pass | 说明 |
|---|---:|---:|---|
| UncondAlign | +0.0035 | false | 弱 |
| InfoNCE | +0.0015 | false | 弱，仅 5-seed |
| DynamicFusion | -0.0053 | false | 不适合 RD |
| TextInject |  |  | 文本锚点 |
| BalancedDirectAdd | +0.0095 | true | 当前最强线索 |

### 结论

```text
RD 不应被简单拉近，也不只是动态降权；
更合适的是保留三模态信息并进行平衡利用。
```

## 8. 实验 6：v6 方法验证

### 候选方法 1：RC-BalancedAdd-Hard

使用硬关系状态设置不同注入强度：

| State | alpha |
|---|---:|
| RD | 1.0 |
| RA | 0.3 |
| Mid-D | 0.3 |
| ND | 0.1 |
| UA | 0.1 |

### 候选方法 2：RC-BalancedAdd-Soft

使用软门控：

```text
alpha_i = alpha_base + alpha_rd * g_RD(i) - alpha_unrel * g_lowR(i)
```

### 候选方法 3：RD-Only BalancedAdd

只对 RD 使用 BalancedAdd，其余样本保持 Concat：

```text
if state == RD:
    use BalancedAdd
else:
    use Concat
```

这是最小验证版本，能最直接回答：

```text
RD 是否真的需要平衡利用？
```

## 9. 实验 7：Kernel Distribution Diagnostic

### 定位

Appendix / supplementary。

### 目的

说明：

```text
RD 不仅 sample-level D 高，
在预测类条件批的 hidden distribution MMD 上也偏高。
```

### 不能做的事

不要把 MMD 当主分组标准，也不要说它证明语义标签分布。

## 10. 实验 8：Residual Probe Negative Result

### 目的

把 residual probe 从主线移到边界分析。

当前结果：

```text
residual_gain_macro_f1 多为负。
```

结论：

```text
残差可能有结构，但简单残差利用不充分；
v6 不应把 residual branch 当主贡献。
```

## 11. 主表建议

### Table 1：Motivation

| Dataset | Seeds | Low-D | Mid-D | High-D | Consistency |
|---|---:|---:|---:|---:|---|
| MOSEI | 10 |  |  |  |  |
| MOSI | 10 |  |  |  |  |

### Table 2：Method Insufficiency

| Method | Overall | High-D | RD | Comment |
|---|---:|---:|---:|---|
| UncondAlign |  |  |  | weak |
| InfoNCE |  |  |  | weak |
| DynamicFusion |  |  |  | weak on RD |

### Table 3：Balanced Utilization

| Method | Overall | RD | ND | RA | UA |
|---|---:|---:|---:|---:|---:|
| Concat |  |  |  |  |  |
| BalancedDirectAdd |  |  |  |  |  |
| RC-BalancedAdd-v6 |  |  |  |  |  |

### Appendix Table：Kernel Diagnostic

| Group | D_sample | R | MMD |
|---|---:|---:|---:|
| RA |  |  |  |
| UA |  |  |  |
| RD |  |  |  |
| ND |  |  |  |

## 12. 成功标准

v6 方法成功不要求巨大 overall 提升，而要求：

1. Overall 不低于 Concat；
2. RD 稳定高于 Concat / UncondAlign / InfoNCE / DynamicFusion；
3. ND/UA 不明显受伤；
4. 比固定 BalancedDirectAdd 更符合 relation-state 差异；
5. 多 seed 稳定。

## 13. 推荐执行顺序

### Step 1：补真正独立 15 seeds

先补：

```text
seeds 11 12 13 14 15
```

确认：

```text
High-D > Mid-D > Low-D 是否 15/15 成立；
BalancedDirectAdd on RD 是否仍稳定；
DynamicFusion on RD 是否仍弱。
```

### Step 2：实现 RD-Only BalancedAdd

最小代码实验，不要先做复杂方法。

### Step 3：实现 RC-BalancedAdd-Hard

用固定状态 alpha。

### Step 4：实现 RC-BalancedAdd-Soft

再引入 soft gate。

### Step 5：再考虑是否需要 relation-aware InfoNCE

只有在 balanced utilization 不够时，再做：

```text
RA as positive alignment
RD as complementary utilization
ND/UA as cautious samples
```

## 14. v6 论文叙事模板

中文：

> 我们首先观察到，高跨模态分歧样本并不一定更难。在 MOSEI 上，基础融合模型在 High-D 样本上的表现反而稳定高于 Mid-D 和 Low-D。这说明跨模态分歧不是简单噪声，而可能包含强信息或互补证据。进一步实验表明，无条件对齐、普通对比学习和动态权重融合都不能稳定处理可靠高分歧样本；相比之下，平衡式跨模态注入在 RD 上表现出更稳定的正向趋势。因此，我们将问题从“如何减少分歧”重新定义为“如何根据关系状态调度和利用分歧”。

英文：

> We first observe that high cross-modal disagreement does not necessarily indicate difficult or low-quality samples. On MOSEI, the baseline fusion model consistently performs best on High-D samples. This suggests that disagreement can encode strong or complementary evidence rather than pure noise. Further experiments show that unconditional alignment, ordinary contrastive learning, and dynamic weighting are insufficient for reliable high-disagreement samples, whereas balanced cross-modal utilization yields the most stable improvement on RD. We therefore reformulate the problem from disagreement reduction to relation-conditioned disagreement utilization.

