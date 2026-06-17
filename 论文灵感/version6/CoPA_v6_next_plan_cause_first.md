# CoPA-v6 下一步计划：动机证据冻结与最小方法验证

> 核心原则：v6 现在要先证明“论文有动机”，不是立刻证明“最终方法已经成功”。  
> 主线应写成：**分歧有结构，关键是按关系状态调度分歧**。

---

## 0. 当前证据状态

当前主线证据来自 MOSEI 15 个独立 seed：

```text
dataset: MOSEI
seeds: 1-15
label_mode: three_class
pair_mode: text_anchor
relation_split: balanced_within_d
disagreement_metric: prob_jsd

runs:
code/disagreement_phenomenon/outputs/mosei/multi_seed_20260616_204548  # seeds 1-5
code/disagreement_phenomenon/outputs/mosei/multi_seed_20260616_214127  # seeds 6-15
```

已生成的冻结表：

```text
code/disagreement_phenomenon/outputs/mosei/v6_cause_analysis_1_15
code/disagreement_phenomenon/outputs/mosei/v6_motivation_tables_1_15
code/disagreement_phenomenon/outputs/mosei/v6_mechanism_analysis_1_15
```

当前可以稳定写进正文的结论：

```text
1. D 是监督模型诱导的 task-aware prediction disagreement，不是 label leakage，也不是纯数据本征分歧。
2. 在这种任务语义下，High-D 不是困难样本。
3. High-D 的高性能很大程度上伴随更强情感强度与文本分支优势。
4. RD/ND 的差异不是简单的 D 大小差异，而是预测确定性结构 R 区分出的状态差异。
5. UncondAlign / InfoNCE / DynamicFusion 都不足以稳定利用 RD。
6. BalancedDirectAdd 是最小正向线索，尤其在 RD 上稳定为正；但它还不是最终方法成功。
```

不能写成：

```text
High-D 是完全无监督、无先验的数据本征高分歧。
High-D/RD 已被证明天然存在跨模态互补。
BalancedDirectAdd 已经是完整 CoPA 方法。
relation-conditioned 方法已经被正式验证成功。
```

---

## 1. 主表与主叙事

### Experiment 1：High-D 不是困难样本

只用 Concat baseline 做主表，避免把方法增益和现象证据混在一起。这里的 D 应写成 `D_pred`：它来自监督训练后的单模态预测分布，是任务感知关系信号。

| Group | N | avg D | avg R | Concat Macro-F1 | 95% CI | Order Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Low-D | 1547.3 | 0.0137 | 0.0048 | 0.505 | [0.500, 0.510] | 15/15 |
| Mid-D | 1645.6 | 0.0392 | 0.0103 | 0.596 | [0.588, 0.603] | 15/15 |
| High-D | 1450.1 | 0.1015 | 0.0156 | 0.687 | [0.680, 0.695] | 15/15 |

写法：

```text
Contrary to the common assumption that larger inter-modal disagreement indicates harder samples,
Concat performance monotonically increases from Low-D to High-D across all 15 seeds.
```

中文理解：

```text
在监督任务语义下，模型感知到的高预测分歧不是天然坏事。至少在 MOSEI 上，High-D 不是困难样本。
```

---

### Experiment 2：High-D 为什么高

主表不再只写性能，要写原因剖面。

| Group | N | avg \|label\| | Label Entropy | Class 0/1/2 | Text Acc | Audio Acc | Vision Acc | Concat Macro-F1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Low-D | 1547.3 | 0.732 | 1.519 | 0.216/0.460/0.324 | 0.531 | 0.472 | 0.501 | 0.505 |
| Mid-D | 1645.6 | 0.759 | 1.483 | 0.170/0.450/0.380 | 0.587 | 0.468 | 0.498 | 0.596 |
| High-D | 1450.1 | 1.040 | 1.568 | 0.280/0.323/0.397 | 0.696 | 0.368 | 0.377 | 0.687 |

写法要克制：

```text
High-D is not simply easier due to lower label entropy. Instead, it contains stronger sentiment
and substantially stronger text-branch predictability.
```

这里不能说“High-D 已证明跨模态互补”。更稳的解释是：

```text
High-D 高性能首先暴露出一个结构性事实：
高分歧样本内部混合了强文本、弱非文本、以及可能可利用的跨模态信息。
因此问题不是降低分歧，而是识别分歧处于什么关系状态。
```

新增 class-prior control 后，解释要再精确一层：

```text
High-D 的 majority acc = 0.397，低于 Low-D 的 0.460 和 Mid-D 的 0.450，
所以 High-D 高性能不是简单由最大类占比更高导致。

但 class-wise accuracy 显示：
Class 0: Low-D 0.340, Mid-D 0.552, High-D 0.863, High-Low +0.522, High-Mid +0.311
Class 1: Low-D 0.666, Mid-D 0.618, High-D 0.491, High-Low -0.175, High-Mid -0.127
Class 2: Low-D 0.493, Mid-D 0.614, High-D 0.741, High-Low +0.248, High-Mid +0.127

因此 High-D 的优势主要来自极性情感类，而不是 neutral。
写作上应说：High-D 是 task-aware prediction disagreement 下的强情感/类别相关结构，
不能泛化成所有类别都更容易。
```

新增 D-vs-polarity 诊断后，主结论应进一步收紧：

```text
D_pred 与真实 |label_reg| 的相关性较弱/中等：
    Pearson = 0.2423
    Spearman = 0.1630

D_pred 与模型预测极性/置信度的相关性更强：
    pred_polarity_conf Spearman = 0.4081
    pred_confidence Spearman = 0.4407
    pred_margin Spearman = 0.3727
    R_sample Spearman = 0.5557
```

polarity-bin controlled 结果：

```text
Low-P:
    Low-D Acc = 0.665
    Mid-D Acc = 0.617
    High-D Acc = 0.492
    结论：低极性/接近 neutral 时，High-D 反而更差。

Mid-P:
    Low-D Acc = 0.334
    Mid-D Acc = 0.501
    High-D Acc = 0.666

High-P:
    Low-D Acc = 0.547
    Mid-D Acc = 0.701
    High-D Acc = 0.874
```

最终写法：

```text
High-D 不是无条件容易，而是 polarity-conditioned。
D_pred 混合了模态预测不一致、预测置信度和情感极性结构。
这正好支持 v6 的更强命题：disagreement 不是单一难度/噪声变量，而是需要解耦和调度的结构变量。
```

新增 cross-seed decoupled diagnostic 后，可以回应“同一模型自己分组、自己证明”的循环性质疑：

```text
source_seed 负责定义 D group，eval_seed 负责评估 Concat performance，
并且排除 source_seed == eval_seed。

Cross-seed 结果仍保持：
    Low-D  Macro-F1 = 0.504, Acc = 0.539
    Mid-D  Macro-F1 = 0.596, Acc = 0.606
    High-D Macro-F1 = 0.687, Acc = 0.694

High-D - Low-D:
    Macro-F1 = +0.183
    Acc = +0.155

High-D - Mid-D:
    Macro-F1 = +0.091
    Acc = +0.088
```

这不消除 polarity/class confound，但它证明 High-D 排序不是同一模型的自证伪影。
更稳的结论是：`D_pred` 捕捉到可复现的任务诱导结构，但这个结构的含义必须按类别、极性和关系状态解释。

---

### Experiment 3：RD / ND 的结构差异

`R` 必须明确写成预测确定性结构，不是真实质量标签。

| Group | N | avg D | avg R | Text Macro-F1 | Audio Macro-F1 | Vision Macro-F1 | Fusion Macro-F1 | Concat Macro-F1 | Oracle Macro-F1 | Oracle - Fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RD | 754.1 | 0.1175 | 0.0247 | 0.688 | 0.295 | 0.370 | 0.692 | 0.693 | 0.922 | 0.230 |
| ND | 696.0 | 0.0842 | 0.0057 | 0.676 | 0.225 | 0.276 | 0.672 | 0.673 | 0.909 | 0.237 |

推荐写法：

```text
RD and ND are both high-disagreement regimes, but they differ in prediction certainty R.
RD has higher R and higher unimodal/fusion performance, while both RD and ND retain a large
oracle gap, suggesting that usable information exists but is not fully exploited by the current fusion.
```

这段最关键：它把 v6 从“High-D 好”推进到“High-D 内部有结构”。

---

## 2. 旧方法为什么不充分

把 UncondAlign、InfoNCE、DynamicFusion、BalancedDirectAdd 放到同一张表里。

| Method | Overall Delta Macro-F1 | Overall 95% CI | Overall EC | High-D Delta Macro-F1 | High-D 95% CI | High-D EC | RD Delta Macro-F1 | RD 95% CI | RD EC |
|---|---:|---|---|---:|---|---|---:|---|---|
| UncondAlign | 0.0023 | [-0.0003, 0.0048] | False | 0.0014 | [-0.0031, 0.0059] | False | 0.0023 | [-0.0026, 0.0073] | False |
| UncondInfoNCE | 0.0019 | [-0.0004, 0.0042] | False | 0.0035 | [-0.0000, 0.0070] | False | 0.0041 | [-0.0002, 0.0084] | False |
| DynamicFusion | 0.0028 | [-0.0010, 0.0067] | False | -0.0004 | [-0.0039, 0.0032] | False | -0.0008 | [-0.0060, 0.0044] | False |
| BalancedDirectAdd | 0.0034 | [0.0004, 0.0065] | False | 0.0056 | [0.0006, 0.0106] | False | 0.0072 | [0.0023, 0.0121] | True |

解释顺序：

```text
UncondAlign / InfoNCE / DynamicFusion are not stable enough on RD.
BalancedDirectAdd is different: it gives the clearest positive clue on RD.
But because it is still globally applied and not relation-conditioned,
it should be framed as a bridge, not as the final CoPA method.
```

注意事项：

```text
InfoNCE 和 DynamicFusion 当前 15-seed 表来自已有 v6_motivation 输出。
如果论文里要强调它们是补充方法，需在实验设置中交代它们的 seed/run 来源。
不要把 5-seed 开发阶段结果和 15-seed 独立结果混写。
```

---

## 3. 机制诊断：为什么这些图和方法表不矛盾

### UncondAlign

机制诊断显示：

```text
UncondAlign 确实把隐藏表征距离大幅拉近。
例如 RD 上 hidden distance delta 约为 -0.353。
```

但对应的 RD Macro-F1 delta 只有：

```text
+0.0023, 95% CI = [-0.0026, 0.0073], EC = False
```

因此应写成：

```text
Hidden states can be made closer, but indiscriminate closeness does not reliably improve RD prediction.
```

这正好支撑 v6：

```text
问题不是“有没有对齐”，而是“什么时候该对齐、什么时候该保留或重分配分歧”。
```

### DynamicFusion

机制诊断显示 DynamicFusion 的权重高度偏向 text：

```text
RD: avg w_text ≈ 0.706, text-dominant rate ≈ 0.844
RD Delta Macro-F1 ≈ -0.0008, EC = False
```

因此 DynamicFusion 更像是在做模态选择，而不是 relation-state scheduling。

推荐写法：

```text
Dynamic weighting alone tends to collapse toward the dominant text modality and does not reliably
recover RD gains, indicating that relation-state-aware utilization requires more than sample-wise modality weighting.
```

---

## 4. BalancedDirectAdd 的定位

BalancedDirectAdd 应从 appendix baseline 升级为正文里的最小正向线索。

但定位必须是：

```text
minimal positive clue / bridge evidence
```

不要写成：

```text
final method success
```

原因：

```text
1. 它在 RD 上稳定为正，并通过当前 error control。
2. 它仍是全局固定注入，不知道 RA/UA/RD/ND。
3. 它在 ND/其他状态上未必都是最优。
4. 它证明“balanced utilization 方向有希望”，不是证明“完整 relation-conditioned 方法已完成”。
```

论文里的功能：

```text
BalancedDirectAdd closes the motivation loop:
old unconditional objectives are insufficient,
but a simple balanced utilization operation already helps RD,
suggesting that relation-conditioned balanced utilization is a plausible next method direction.
```

---

## 5. 下一步执行计划

### Step A：先冻结动机实验

优先完成正文动机段落和主表，不继续扩 appendix。

正文建议只保留四组表：

```text
1. Concat Low/Mid/High-D 表
2. High-D cause profile 表
3. RD/ND reliability + oracle 表
4. Method insufficiency + BalancedDirectAdd bridge 表
```

appendix 放：

```text
1. lambda 曲线
2. relation-state detailed seed dots
3. kernel/MMD diagnostics
4. residual probe negative/boundary analysis
5. MOSI robustness
```

不是每个 appendix 都必须现在跑。当前最重要的是 MOSEI 主线闭环。

### Step B：再跑最小 v6 方法验证

只在动机文字和表冻结后跑：

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py `
  --preset v6_motivation `
  --dataset mosei `
  --data_root E:\Xu\data\MultiBench `
  --seeds 1 2 3 4 5 `
  --run_rc_balanced_add `
  --rc_balanced_modes rd_only hard `
  --num_workers 0
```

Hard 默认 alpha：

```text
RD=1.0
RA=0.3
Mid-D=0.3
ND=0.1
UA=0.1
```

接受标准：

```text
overall 不低于 Concat；
RD 稳定优于 Concat / UncondAlign / InfoNCE / DynamicFusion；
ND/UA 不出现明确伤害；
RC-BalancedAdd-Hard 至少在 RD 上优于固定 BalancedDirectAdd，
否则写成清楚的负结果。
```

### Step C：成功后再扩 15 seed

如果 5-seed pilot 清楚为正，再跑 1-15。

如果 5-seed pilot 不清楚，不要硬扩；先分析：

```text
是不是 alpha 太强；
是不是 relation-state 阈值不稳；
是不是 RD 本身主要是 text-dominant，balanced injection 只能有限提升；
是不是需要 soft gate，而不是 hard rule。
```

当前 5-seed pilot 更新：

```text
RC-BalancedAdd-RDOnly:
    RD mean delta = -0.0046
    RD mean (RC - fixed BalancedDirectAdd) = -0.0137
    结论：不应扩 15 seed。

RC-BalancedAdd-Hard:
    RD mean delta = +0.0074
    RD mean (RC - fixed BalancedDirectAdd) = -0.0017
    Overall mean delta = +0.0032
    UA mean (RC - fixed BalancedDirectAdd) = -0.0270
    结论：有 RD 正趋势，但没有超过固定 BalancedDirectAdd，且 UA 有相对伤害。
```

因此当前决策是：

```text
不要扩 RC-BalancedAdd-Hard 到 15 seeds。
不要把 hard gate 写成方法成功。
可以把它作为 negative/boundary result：
hard relation-state alpha schedule 不足以超过 fixed balanced utilization，
后续若继续做方法，应转向 learned/soft gate。
```

---

## 6. 当前论文写法边界

现在可以写：

```text
Our motivation experiments reveal that disagreement is structured rather than uniformly harmful.
High disagreement contains reliably predictable regions, but unconditional alignment and dynamic weighting
do not consistently exploit them. A simple balanced utilization operation gives the strongest positive clue on RD,
motivating relation-conditioned utilization.
```

现在不该写：

```text
Our proposed method solves relation-conditioned multimodal disagreement.
```

更稳的 v6 贡献顺序：

```text
1. Phenomenon: High-D is not hard.
2. Diagnosis: High-D/RD structure is partly explained by sentiment strength and prediction certainty.
3. Boundary: unconditional alignment / contrastive / dynamic weighting are insufficient.
4. Bridge: BalancedDirectAdd suggests balanced utilization can help RD.
5. Method next: relation-conditioned balanced utilization is the natural minimal validation.
```
