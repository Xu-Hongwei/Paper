# CoPA version6：从“消除分歧”转向“调度分歧”

> 暂定题目：**CoPA-v6: Relation-Conditioned Balanced Utilization of Multimodal Disagreement**
>
> 中文定位：**面向多模态融合的关系条件分歧利用方法**

## 0. v6 为什么要重建

最近 MOSEI 实验给出了一个比原先假设更强、也更有意思的现象：

```text
Concat baseline 下：
High-D > Mid-D > Low-D
```

也就是说，高分歧样本不是更难、更差或更噪声的样本。相反，在 MOSEI 上，高分歧样本反而更容易被普通融合模型预测。

因此，v6 不再把核心动机写成：

```text
跨模态分歧有害，所以要减少分歧。
```

而改成：

```text
跨模态分歧是一种信息结构；真正的问题不是消除分歧，而是识别和调度分歧。
```

## 1. v5 到 v6 的关键变化

| 维度 | v5 旧口径 | v6 新口径 |
|---|---|---|
| 对 High-D 的理解 | 可能是对齐风险或可靠不一致 | 高分歧不等于困难，可能是强信息/互补 |
| 对 alignment 的理解 | 需要关系门控对齐 | 无条件对齐和普通对比学习都偏弱 |
| 对 RD 的理解 | 可靠不一致，适合残差对比 | 可靠高分歧，适合平衡利用 |
| 对 residual 的理解 | 可能是主贡献 | residual probe 当前是负结果，只能辅助 |
| 对方法方向 | relation-gated contrastive/residual | relation-conditioned balanced utilization |
| 对核分布诊断 | 新增补强 | appendix 诊断，不替代单样本分组 |

## 2. 当前结果直接支持的事实

### 2.1 高分歧不是困难样本

MOSEI 10 个独立 seed 中，Concat 的分组表现稳定为：

| Group | Macro-F1 |
|---|---:|
| Low-D | 约 0.503 |
| Mid-D | 约 0.599 |
| High-D | 约 0.683 |

并且逐 seed 都满足：

```text
High-D > Mid-D > Low-D
```

这是 v6 最重要的动机来源。

### 2.2 无条件对齐很弱

`UncondAlign - Concat` 在 MOSEI 10 seeds 上：

| Group | Delta Macro-F1 |
|---|---:|
| Low-D | +0.0055 |
| Mid-D | +0.0013 |
| High-D | +0.0018 |
| Overall | +0.0024 |

所有置信区间都跨 0，说明简单把模态拉近不是有效主解法。

### 2.3 普通 InfoNCE 也不够

5-seed InfoNCE 实验显示：

| Group | Delta Macro-F1 |
|---|---:|
| Low-D | -0.0045 |
| Mid-D | +0.0003 |
| High-D | +0.0036 |
| Overall | +0.0017 |

普通同样本跨模态正样本假设并没有稳定解决问题。

### 2.4 DynamicFusion 也不是 RD 的答案

最新 5-seed 复跑中新增了 DynamicFusion：

| Group | Delta Macro-F1 |
|---|---:|
| Low-D | +0.0052 |
| Mid-D | +0.0086 |
| High-D | -0.0029 |
| RD | -0.0053 |
| Overall | +0.0035 |

它有一点整体收益，但对 High-D/RD 并不友好。这说明“动态给模态加权”不能直接解释可靠高分歧。

### 2.5 BalancedDirectAdd 是最有价值的线索

MOSEI 10-seed 主结果中：

| Method | Overall Macro-F1 |
|---|---:|
| Concat | 0.6121 |
| UncondAlign | 0.6145 |
| TextInject | 0.6151 |
| BalancedDirectAdd | 0.6161 |

在 RD 上：

```text
BalancedDirectAdd - Concat:
Delta Macro-F1 = +0.0095
95% CI = [0.0027, 0.0163]
```

这是目前最稳定的正向方法信号。

## 3. v6 的核心观点

v6 的核心观点可以写成：

> Multimodal disagreement is structured rather than inherently harmful. High-disagreement samples can be highly predictive, while uniform alignment and ordinary contrastive alignment provide only marginal gains. Therefore, multimodal fusion should condition its interaction strategy on relation states: reliable disagreement should be exploited through balanced cross-modal utilization, while unreliable relations should be treated cautiously.

中文版本：

> 多模态分歧并非天然有害，而是具有结构性。高分歧样本可能更容易预测，说明分歧中包含强信息或互补信息；无条件对齐和普通对比学习收益都很弱。因此，多模态融合不应简单消除分歧，而应根据关系状态调度分歧：可靠高分歧应被平衡利用，不可靠关系应被谨慎处理。

一句话：

```text
不是减少分歧，而是调度分歧。
```

## 4. v6 方法定位

v6 方法不再以“对齐损失”为唯一中心，而以 **relation-conditioned balanced utilization** 为中心。

### 4.1 输入

每个样本有三模态特征：

```text
text, audio, vision
```

经编码器得到：

```text
h_t, h_a, h_v
```

诊断模型给出：

```text
D_sample: 单样本跨模态预测分歧
R_sample: 样本平均预测可靠性
```

再得到关系状态：

```text
RA: 低分歧 + 高可靠
UA: 低分歧 + 低可靠
RD: 高分歧 + 高可靠
ND: 高分歧 + 低可靠
Mid-D: 中分歧
```

### 4.2 v6 主要处理策略

| 状态 | 结果启发 | v6 处理 |
|---|---|---|
| RA | 一致且相对可靠 | 可轻量融合，不需要强对齐 |
| UA | 一致但不可靠 | 谨慎使用，降低注入强度 |
| RD | 高分歧且可靠，Fusion 表现最高 | 重点平衡利用，保留互补 |
| ND | 高分歧但低可靠，结构不稳定 | 避免强对齐，可降权或保守融合 |
| Mid-D | 中间状态 | 默认融合或弱调节 |

### 4.3 候选方法：Relation-Conditioned Balanced Add

在当前 BalancedDirectAdd 的基础上，v6 最小方法可以设计为：

```text
aligned = mean(norm(h_t), norm(h_a), norm(h_v))

fuse_t = h_t + alpha_state * aligned
fuse_a = h_a + alpha_state * aligned
fuse_v = h_v + alpha_state * aligned
```

其中 `alpha_state` 根据关系状态变化：

| State | 建议 alpha |
|---|---:|
| RD | 高 |
| RA | 中低 |
| Mid-D | 中低 |
| ND | 低 |
| UA | 低 |

这比固定 `BalancedDirectAdd` 更符合 v6 动机：

```text
不是所有样本都同样注入，而是可靠高分歧更强地平衡利用。
```

### 4.4 候选方法：Soft Relation Gate

硬分组可能不稳定，因此 v6 更推荐 soft gate：

```text
alpha_i = alpha_base * g_RD(i) + alpha_mid * g_Mid(i) + alpha_low * g_RA/UA/ND(i)
```

其中 gate 不使用 test label，只来自诊断模型预测分布。

## 5. v6 不应再强调的内容

以下说法在当前结果下不稳：

```text
High-D 是噪声。
High-D 是困难样本。
无条件对齐可以解决高分歧。
普通 InfoNCE 足够。
动态权重融合能处理 RD。
残差分支本身有稳定判别增益。
```

v6 应明确收缩：

```text
residual probe 是负结果或边界诊断；
kernel MMD 是补充证据；
InfoNCE 是普通对比学习 baseline；
BalancedDirectAdd 是启发 v6 的最小正向线索。
```

## 6. v6 实验叙事

v6 实验部分应按这个顺序展开：

1. 高分歧不是困难样本；
2. 高分歧需要可靠性解释；
3. 无条件对齐、普通 InfoNCE、动态权重都不充分；
4. 平衡利用在 RD 上最有希望；
5. 提出 relation-conditioned balanced utilization；
6. 再验证 v6 方法是否优于固定 BalancedDirectAdd。

## 7. v6 当前最小主张

最稳主张：

```text
Disagreement-aware fusion should not ask whether modalities should be aligned,
but how different relation states should be utilized.
```

中文：

```text
分歧感知融合不应只问“模态是否应该对齐”，
而应问“不同关系状态下的分歧应该如何被利用”。
```

## 8. v6 成功标准

v6 方法不需要一开始追求巨大 overall 提升。更合理的成功标准是：

1. 保持或提升 overall；
2. 在 RD 上稳定优于 Concat、UncondAlign、InfoNCE、DynamicFusion；
3. 不明显伤害 ND/UA；
4. 比固定 BalancedDirectAdd 更能解释 relation-state 差异；
5. 多 seed 结论稳定，而不是单 seed 偶然。

## 9. 当前证据等级

| 结论 | 证据等级 | 说明 |
|---|---|---|
| High-D 不是困难样本 | 强 | 10 个独立 seed 都成立 |
| UncondAlign 很弱 | 强 | 10-seed 与 5-seed 同方向 |
| InfoNCE 不充分 | 中 | 当前只有 5-seed，趋势清楚但需补 11-15 |
| DynamicFusion 不解决 RD | 中 | 当前 5-seed，新增但需补独立 seed |
| BalancedDirectAdd 在 RD 上最有希望 | 中强 | 10-seed 通过，5-seed 同方向 |
| Kernel MMD 补强 RD | 中 | 两次一致，但只作为 appendix |
| Residual 分支是主贡献 | 弱/不支持 | 当前 probe 多为负结果 |

## 10. 下一步

v6 下一步不是继续堆 alignment，而是实现一个最小 relation-conditioned balanced add：

```text
Concat
UncondAlign
UncondInfoNCE
DynamicFusion
BalancedDirectAdd
RC-BalancedAdd-v6
```

然后用 MOSEI 10 seeds 或 15 independent seeds 评估：

```text
Overall
Low-D / Mid-D / High-D
RA / UA / RD / ND
Kernel distribution appendix
```

