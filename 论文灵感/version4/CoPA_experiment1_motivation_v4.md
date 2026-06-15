# CoPA 实验1 version4：Selective Agreement and Discriminative Disagreement Motivation Analysis

> 文档定位：本文档只说明 **论文出发点实验 / motivation analysis** 的 version4 设计。  
> v4 核心目标：在 v3 “预测分布发现关系状态 + 特征矩阵验证结构差异”的基础上，进一步加入两条证据：
>
> 1. **一致样本不应全部采纳**：Low-D 样本内部也要区分 reliable agreement 与 uncertain / wrong agreement；
> 2. **不一致样本不应全部丢弃**：High-D+High-R 的 residual 是否具有可验证的判别价值。
>
> 因此，v4 的实验1从双层证据链升级为三层证据链：
>
> \[
> \text{prediction-level relation signal}
> \Rightarrow
> \text{feature-level residual structure}
> \Rightarrow
> \text{residual discriminative evidence}
> \]

---

## 0. v4 相比 v3 的核心变化

| 项目 | v3 | v4 |
|---|---|---|
| 核心命题 | paired modalities 不应无条件对齐 | 一致样本要筛选，不一致样本要验证是否有判别价值 |
| 证据链 | prediction distribution + feature residual | prediction distribution + feature residual + residual probe |
| Concat 边界 | 主要作为无显式对齐 baseline | 明确承认 Concat 能学标签边界，但不显式建模同标签样本内关系状态 |
| 一致样本分析 | 主要关注 Low-D / High-D 差异 | 新增 Low-D 内部 RA / UA 分析 |
| 不一致样本分析 | High-D+High-R 可能有结构化 residual | 新增 residual-only / common+residual 轻量探针 |
| 无损对齐质疑 | 未单独检验直接相加 | 新增 DirectAdd baseline，检验“原路径保留 + 直接相加”是否足够 |
| 结论边界 | 不能仅凭 residual sep 声称有判别价值 | 用 residual probe 支撑“discriminative disagreement” |
| 负对照 | random grouping / shuffled residual | 新增 shuffled relation weight / shuffled residual-label probe |

---

## 0.1 关键边界修正：为什么不是说 Concat 不会学标签

实验1不能写成：

> Concat 学不到同标签 / 不同标签边界，所以 CoPA 才有意义。

这是不准确的。Concat baseline 的监督学习形式是：

\[
z_{concat}=[h_t;h_v;h_a]
\]

\[
\hat y=F(z_{concat})
\]

\[
\mathcal{L}_{task}=CE(\hat y,y)
\]

因此 Concat 确实能通过任务监督学习标签级分类边界，包括：

\[
y^n=y^k
\]

和：

\[
y^n\neq y^k
\]

之间的决策差异。

实验1真正要证明的不是“Concat 不会分类”，而是：

> **Concat can learn label-level decision boundaries, but it does not explicitly model relation states within paired multimodal samples.**

中文表述：

> **Concat 能学习标签层面的分类边界，但不能显式建模同一样本多模态内部的可靠一致、可靠不一致和不可靠关系。**

因此，Experiment 1 的边界应写成：

1. Concat already learns label-level boundaries；
2. Unconditional Alignment imposes a fixed relation assumption on paired modalities；
3. Relation states may still reveal meaningful subgroup differences beyond ordinary concatenation and fixed-strength alignment；
4. Reliable disagreement may contain residual information that ordinary implicit fusion or direct feature addition does not explicitly isolate。

---

## 0.2 “无损对齐”的重新定义

本文中“无损”不应解释为：

\[
\text{lossless alignment} = \text{original feature} + \text{aligned feature}
\]

这种直接相加只是一个朴素 baseline，仍然会把 reliable agreement、reliable disagreement 和 noisy relation 混在同一条融合路径中。

更准确的定义是：

> **Lossless does not mean directly adding aligned representations back to original features. It means preserving the original supervised fusion path for all samples while applying relation-conditioned auxiliary constraints only to reliable agreement or reliable disagreement cases.**

中文：

> **无损不是把对齐表示直接加回原特征，而是所有样本仍保留原始监督融合路径，同时只对可靠一致和可靠不一致样本施加关系条件辅助约束。**

对应训练角色是：

\[
\mathcal{L}_{task}: \text{all samples}
\]

\[
\mathcal{L}_{agr}: \text{reliable agreement only}
\]

\[
\mathcal{L}_{dis}: \text{reliable disagreement only}
\]

\[
\mathcal{L}_{relation}: \text{down-weight unreliable relations}
\]

---

## 1. 实验1最终要回答的问题

Experiment 1 不直接证明 CoPA 方法最强，而是验证关系状态是否能提供超越普通 Concat、固定强度对齐和直接相加的诊断信号。v4 需要回答八个问题：

1. Concat 在 overall 上是否具有合理性能，同时在不同 relation group 上表现存在差异？
2. 无条件对齐收益是否依赖 prediction-level disagreement？
3. 无条件对齐收益是否对 alignment strength 敏感？
4. DirectAdd 是否能稳定解决“保留原始信息 + 对齐信息”的问题？
5. Low-D 样本是否都适合作为可靠对齐正样本？
6. High-D 样本是否能分为 reliable disagreement 与 noisy disagreement？
7. High-D+High-R 是否在特征残差空间中有结构化差异？
8. High-D+High-R 的 residual 是否具有任务判别价值？

对应最终结论：

> Concat can learn label-level boundaries, but paired modalities should not be uniformly aligned or directly mixed without relation awareness. Reliable agreement should be selectively used for common semantic alignment, while reliable disagreement should be examined and utilized as discriminative residual information rather than simply treated as noise.

---

## 2. 数据与基础模型

### 2.1 数据集优先级

建议：

1. MOSI；
2. MOSEI；
3. CH-SIMS / CH-SIMS-v2；
4. IEMOCAP；
5. MUStARD / UR-FUNNY。

其中：

- MOSI/MOSEI 用于主流 benchmark；
- CH-SIMS 更适合分析模态级情感差异；
- MUStARD / UR-FUNNY 更适合证明 reliable disagreement 的判别价值。

### 2.2 基础模型

实验1至少需要三个 baseline：

| 模型 | 作用 |
|---|---|
| Concat baseline | 不显式对齐，作为基础融合 |
| Unconditional Alignment baseline | 将同一样本模态无条件拉近 |
| DirectAdd baseline | 保留原始路径，并把样本级对齐摘要直接加回模态表示 |

Unconditional Alignment 可定义为：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_{align}\mathcal{L}_{align}
\]

\[
\mathcal{L}_{align}
=
\sum_n
\sum_{i<j}
\|z_i^n-z_j^n\|_2^2
\]

也可使用 cosine distance：

\[
\mathcal{L}_{align}
=
\sum_n
\sum_{i<j}
(1-\cos(z_i^n,z_j^n))
\]

第一版建议使用 cosine distance，更适合不同模态投影后的 normalized representation。

DirectAdd 用来检验一个朴素想法：

> 既然担心对齐损失原始信息，那么直接保留原表示并加回对齐表示是否足够？

当前代码中的 DirectAdd 定义为：

\[
h_{align}^n=\frac{1}{3}(h_t^n+h_v^n+h_a^n)
\]

\[
h_m^{add,n}=h_m^n+\alpha h_{align}^n
\]

然后使用：

\[
[h_t^{add};h_v^{add};h_a^{add}]
\]

进行融合预测。

\[
\alpha\in\{0.1,0.3,0.5,1.0\}
\]

DirectAdd 不是 CoPA 主方法，而是一个动机对照：

> Direct addition preserves the original path but still mixes reliable agreement, reliable disagreement, and noisy relations in the same representation path.

中文：

> 直接相加虽然保留原始路径，但仍会把可靠一致、可靠不一致和噪声关系混在同一条表示路径中。

### 2.3 特征来源

Experiment 1 的特征来源应固定为：

- Concat baseline；
- 或 warm-up encoder。

不要使用训练后的 CoPA 特征做 motivation analysis，避免循环论证。

---

## 3. 关系信号定义

### 3.1 单模态预测分布

对每个样本 \(n\)、模态 \(m\)：

\[
p_m^n=\text{Softmax}(C_m(h_m^n))
\]

必须使用完整分布，而不是 argmax：

\[
p_m^n\neq \hat y_m^n
\]

原因：

\[
[0.51,0.49]
\quad \text{and} \quad
[0.99,0.01]
\]

可能 argmax 相同，但可靠性完全不同。

### 3.2 Prediction-level disagreement

\[
D_{pred}^n
=
\frac{1}{3}
\left[
JSD(p_t^n,p_v^n)
+
JSD(p_t^n,p_a^n)
+
JSD(p_v^n,p_a^n)
\right]
\]

这里 \(D_{pred}\) 只能解释为：

\[
D_{pred}=\text{task-level noisy relation indicator}
\]

不能直接解释为真实语义冲突。

### 3.3 Diagnostic reliability

motivation 分组使用：

\[
R_m^{diag,n}
=
1-\frac{H(p_m^n)}{\log K}
\]

样本级可靠性：

\[
R_n^{diag}
=
\frac{1}{3}(R_t^{diag,n}+R_v^{diag,n}+R_a^{diag,n})
\]

可选 margin reliability：

\[
R_m^{margin,n}=p_m^{top1,n}-p_m^{top2,n}
\]

组合：

\[
R_m^{diag,n}
=
\alpha R_m^{entropy,n}
+
(1-\alpha)R_m^{margin,n}
\]

第一版建议：

\[
\alpha=1
\]

即只用 entropy reliability，减少变量。

### 3.4 Label-support reliability

只用于训练集或验证集分析，不能用于 test 分组：

\[
R_m^{label,n}
=
\left(1-\frac{H(p_m^n)}{\log K}\right)\cdot p_m^n(y^n)
\]

它用于分析：

> 一致样本中是否存在低 label-support 或高置信错误样本。

### 3.5 训练时的 Q / S / A 关系权重

在 motivation 分组中，test set 不使用标签；但在训练集上的 CoPA / soft split 诊断中，可以使用训练标签构造 label-aware gates。

定义证据可靠性：

\[
Q_m^n=1-\frac{H(p_m^n)}{\log K}
\]

定义标签支持度：

\[
S_m^n=p_m^n(y^n)
\]

定义模态间一致性：

\[
A_{ij}^n=\exp(-JSD(p_i^n,p_j^n)/\tau_A)
\]

可靠一致权重：

\[
g_{ij}^{agr,n}=Q_i^nQ_j^nS_i^nS_j^nA_{ij}^n
\]

可靠不一致不应使用 \(S_iS_j\) 同时压低两个模态，因为 reliable disagreement 中可能存在“高置信但方向相反”的判别性反差模态。因此使用标签锚点：

\[
B_{ij}^{label,n}=\max(S_i^n,S_j^n)
\]

\[
g_{ij}^{dis,n}=Q_i^nQ_j^nB_{ij}^{label,n}(1-A_{ij}^n)
\]

不可靠关系权重：

\[
g_{ij}^{noise,n}=1-Q_i^nQ_j^n
\]

当前代码中，历史列名 \(g_{ij}^{comp}\) 保留为 \(g_{ij}^{dis}\) 的兼容别名。

---

## 4. 分组方式

### 4.1 D 分组

使用 validation set 确定阈值：

\[
\tau_D^{low},\quad \tau_D^{high}
\]

通常取 1/3 和 2/3 分位数。

测试集只应用阈值，不重新确定阈值：

\[
Low-D,\quad Mid-D,\quad High-D
\]

### 4.2 R 分组

使用 validation set 确定：

\[
\tau_R
\]

通常取中位数。

\[
High-R: R_n^{diag}\geq \tau_R
\]

\[
Low-R: R_n^{diag}< \tau_R
\]

### 4.3 四类关系状态

| Group | 定义 | 解释 |
|---|---|---|
| RA: Low-D+High-R | 预测分歧低、可靠性高 | 可靠一致 |
| UA: Low-D+Low-R | 预测分歧低、可靠性低 | 不确定一致 |
| RD: High-D+High-R | 预测分歧高、可靠性高 | 可靠不一致 |
| ND: High-D+Low-R | 预测分歧高、可靠性低 | 噪声/弱模态不一致 |

Mid-D 可作为过渡组，主文可简化，附录展开。

---

# Part 0：Concat-aware Motivation Analysis

## 4.4 目的

先确认实验1的叙事边界：

> Concat 可以学习标签级分类边界，但 relation state 仍然能揭示普通隐式融合未显式处理的同一样本内部差异。

这一步避免审稿人质疑：

> CE / Concat 不也已经学了同标签和不同标签边界吗？

## 4.5 输出表格

| Dataset | Group | Concat F1 | UncondAlign F1 | DirectAdd F1 | SoftSplit Probe F1 | Residual Gain |
|---|---|---:|---:|---:|---:|---:|
| MOSI | RA |  |  |  |  |  |
| MOSI | UA |  |  |  |  |  |
| MOSI | RD |  |  |  |  |  |
| MOSI | ND |  |  |  |  |  |
| MOSEI | RA |  |  |  |  |  |
| MOSEI | UA |  |  |  |  |  |
| MOSEI | RD |  |  |  |  |  |
| MOSEI | ND |  |  |  |  |  |

当前代码对应输出：

```text
concat_aware_motivation.csv
concat_aware_motivation_summary.csv
```

其中：

- Concat F1 来自普通监督融合；
- UncondAlign F1 来自固定强度无条件对齐；
- DirectAdd F1 来自直接相加 baseline；
- SoftSplit Probe F1 来自 common+residual probe；
- Residual Gain 为 common+residual 相比 common-only 的增益。

## 4.6 支持结论

| 现象 | 支持的结论 |
|---|---|
| Concat overall 合理 | 标签监督本身有效，实验不是证明 Concat 不会分类 |
| Concat 在 RA / UA / RD / ND 上差异明显 | relation state 具有诊断意义 |
| DirectAdd 不稳定优于 Concat | 直接相加不足以替代关系条件拆分 |
| RD 上 SoftSplit Probe 优于 DirectAdd 或有 residual gain | reliable disagreement 可能有判别残差信息 |

---

# Part 1：Unconditional Alignment Gain by Disagreement

## 5.1 目的

证明：

> 无条件对齐收益依赖 prediction-level relation state。

## 5.2 指标

\[
\Delta_g
=
F1_{\text{UncondAlign}}^{g}
-
F1_{\text{Concat}}^{g}
\]

其中：

\[
g\in\{Low-D,Mid-D,High-D\}
\]

## 5.3 输出表格

| Dataset | Group | Sample Num | Concat F1 | Uncond Align F1 | Delta | Std |
|---|---|---:|---:|---:|---:|---:|
| MOSI | Low-D |  |  |  |  |  |
| MOSI | Mid-D |  |  |  |  |  |
| MOSI | High-D |  |  |  |  |  |
| MOSEI | Low-D |  |  |  |  |  |
| MOSEI | Mid-D |  |  |  |  |  |
| MOSEI | High-D |  |  |  |  |  |

## 5.4 可支持结论

只要满足以下之一即可：

- Low/Mid/High-D 的 delta 有系统差异；
- High-D 的方差明显更大；
- High-D 在 MOSEI 等数据集上收益接近 0 或负；
- 不同数据集呈现“对齐收益不稳定”。

支持结论：

> Alignment gain is relation-dependent.

不要写成：

> High-D 一定被无条件对齐伤害。

---

# Part 2：Alignment Strength Sensitivity

## 6.1 目的

证明：

> fixed-strength unconditional alignment is unreliable and strength-sensitive.

## 6.2 设置

训练多个 Unconditional Alignment baseline：

\[
\lambda_{align}\in\{0.001,0.005,0.01,0.05,0.1\}
\]

每个 \(\lambda\) 计算：

\[
\Delta_g(\lambda)
=
F1_{\text{UncondAlign},\lambda}^{g}
-
F1_{\text{Concat}}^{g}
\]

## 6.3 输出表格

| Dataset | \(\lambda_{align}\) | Low-D Delta | Mid-D Delta | High-D Delta | Overall Delta |
|---|---:|---:|---:|---:|---:|
| MOSI | 0.001 |  |  |  |  |
| MOSI | 0.005 |  |  |  |  |
| MOSI | 0.01 |  |  |  |  |
| MOSI | 0.05 |  |  |  |  |
| MOSI | 0.1 |  |  |  |  |

## 6.4 图

建议画折线图：

- x-axis：\(\lambda_{align}\)；
- y-axis：\(\Delta_g(\lambda)\)；
- 三条曲线：Low-D / Mid-D / High-D。

理想现象：

- 小 \(\lambda\) 某些组可能受益；
- 大 \(\lambda\) 多数组退化；
- Low-D 也未必所有强度下都受益；
- High-D 方差更大或收益更不稳定。

---

# Part 2.5：Direct Addition Insufficiency

## 6.5 目的

检验：

> Preserving original features by direct addition is sufficient.

是否成立。

如果 DirectAdd 不能稳定优于 Concat，或对 \(\alpha\) 敏感，说明“原表示 + 对齐摘要”并不能替代 relation-conditioned soft split。

## 6.6 设置

训练多个 DirectAdd baseline：

\[
\alpha\in\{0.1,0.3,0.5,1.0\}
\]

每个 \(\alpha\) 计算：

\[
\Delta_g^{add}(\alpha)
=
F1_{\text{DirectAdd},\alpha}^{g}
-
F1_{\text{Concat}}^{g}
\]

其中：

\[
g\in\{Low-D,Mid-D,High-D,RA,UA,RD,ND\}
\]

## 6.7 输出表格

| Dataset | \(\alpha\) | Low-D Delta | Mid-D Delta | High-D Delta | RA Delta | RD Delta | Overall Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOSI | 0.1 |  |  |  |  |  |  |
| MOSI | 0.3 |  |  |  |  |  |  |
| MOSI | 0.5 |  |  |  |  |  |  |
| MOSI | 1.0 |  |  |  |  |  |  |

当前代码对应输出：

```text
direct_add_alpha_sweep_valid.csv
direct_add_alpha_test_delta_metrics.csv
direct_add_delta_metrics.csv
direct_add_relation_state_delta.csv
```

## 6.8 支持结论

满足任一即可说明 DirectAdd 不足：

- DirectAdd 不稳定优于 Concat；
- DirectAdd 在 RD / High-D+High-R 上不如 SoftSplit Probe；
- DirectAdd 对 \(\alpha\) 敏感；
- DirectAdd 提升 Overall 但不能解释 RA / RD / ND 的差异。

支持结论：

> Direct addition preserves original features but does not assign different learning roles to reliable agreement, reliable disagreement, and unreliable relations.

---

# Part 3：Selective Agreement Diagnostic

## 7.1 新增原因

version4 需要证明：

> 一致样本不应全部采纳为 alignment-positive。

仅有 Low-D 不足以说明可靠一致。Low-D 内部可能包含：

1. 三个模态都高置信且支持真实标签；
2. 三个模态都低置信；
3. 三个模态预测一致但一起预测错；
4. 一个强模态主导，其他模态无效跟随。

因此，需要在 Low-D 内部进一步划分。

## 7.2 分组

在 Low-D 样本内部，根据 \(R^{diag}\) 或 \(R^{label}\) 分组：

| Group | 定义 | 含义 |
|---|---|---|
| Low-D + High-R | 可靠一致 | 可作为 alignment-positive |
| Low-D + Low-R | 低置信一致 | 不应强采纳 |
| Low-D + High label-support | 高置信且支持真实标签 | 训练原型优先使用 |
| Low-D + Low label-support | 一致但不支持真实标签 | 可能污染原型 |

注意：

- test motivation 主分组不用 label-support；
- train/val 分析可使用 label-support；
- 如果展示 test 上 label-support，只能作为事后分析，不能用于方法决策。

## 7.3 诊断指标

### 7.3.1 Alignment gain

| Dataset | Group | Sample Num | Concat F1 | Uncond Align F1 | Delta |
|---|---|---:|---:|---:|---:|
| MOSI | Low-D+High-R |  |  |  |  |
| MOSI | Low-D+Low-R |  |  |  |  |
| MOSEI | Low-D+High-R |  |  |  |  |
| MOSEI | Low-D+Low-R |  |  |  |  |

若 Low-D+Low-R 的收益弱或方差大，说明：

> prediction agreement alone is insufficient for reliable alignment.

### 7.3.2 Prototype contamination check

构造两种普通原型：

1. All-agreement prototype：所有 Low-D 样本更新；
2. Selective-agreement prototype：只用 Low-D+High-R 或 high label-support 样本更新。

比较：

| Prototype | Prototype purity ↑ | Intra-class compactness ↑ | Test F1 |
|---|---:|---:|---:|
| All Low-D |  |  |  |
| Selective RA |  |  |  |

其中 prototype purity 可定义为：

\[
Purity(P_c)=
\frac{
1}{|\mathcal{S}_c|}
\sum_{n\in\mathcal{S}_c}
p_m^n(y^n)
\]

或用 nearest-prototype classification accuracy 近似。

## 7.4 支持结论

如果 Selective RA 原型优于 All Low-D 原型，则支持：

> Consistent samples should be selectively adopted rather than fully accepted.

---

# Part 4：D × R Reliability Stratification

## 8.1 目的

证明 High-D 不是同质群体：

\[
High-D = RD + ND
\]

即：

- High-D+High-R：可能是可靠不一致；
- High-D+Low-R：更可能是噪声、弱模态或不稳定预测。

## 8.2 输出表格

| Dataset | Group | Sample Num | Ratio | Avg-Dpred | Avg-R | Concat F1 | Uncond Align F1 | Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MOSI | RA: Low-D+High-R |  |  |  |  |  |  |  |
| MOSI | UA: Low-D+Low-R |  |  |  |  |  |  |  |
| MOSI | RD: High-D+High-R |  |  |  |  |  |  |  |
| MOSI | ND: High-D+Low-R |  |  |  |  |  |  |  |

## 8.3 支持结论

如果 RD 和 ND 的性能、delta、方差或残差结构不同，说明：

> High-D contains both reliable disagreement and noisy disagreement.

---

# Part 5：Feature-level Consistency Check

## 9.1 目的

回应质疑：

> 预测分布压缩了高维特征，是否会损失太多信息？

因此，需要验证：

> prediction-level disagreement 是否在 feature space 中也有差异迹象。

## 9.2 特征投影

从 Concat baseline 或 warm-up encoder 提取：

\[
h_t^n,h_v^n,h_a^n
\]

投影到同一维度：

\[
\tilde h_m^n=Proj_m(h_m^n)
\]

L2 normalization：

\[
\bar h_m^n=
\frac{\tilde h_m^n}{\|\tilde h_m^n\|_2}
\]

## 9.3 Feature-level disagreement

\[
D_{feat}^n
=
\frac{1}{3}
\sum_{i<j}
(1-\cos(\bar h_i^n,\bar h_j^n))
\]

计算：

\[
\rho=\text{corr}(D_{pred},D_{feat})
\]

可使用 Pearson 或 Spearman。

## 9.4 输出表格

| Dataset | Group | Avg-Dpred | Avg-Dfeat | Corr(Dpred,Dfeat) |
|---|---|---:|---:|---:|
| MOSI | RA |  |  |  |
| MOSI | RD |  |  |  |
| MOSI | ND |  |  |  |
| MOSEI | RA |  |  |  |

## 9.5 解释边界

如果相关性较高：

> Prediction disagreement is not purely a classifier artifact.

如果相关性不高，也不能直接否定：

> feature spaces are modality-heterogeneous; prediction disagreement reflects task-level evidence inconsistency, which may not be linearly captured by raw feature distance.

此时需要依赖 Part 6 和 Part 7。

---

# Part 6：Residual Distribution Diagnostic

## 10.1 目的

判断 RD 样本是否存在结构化 residual，而不是随机噪声。

## 10.2 残差构造

无显式 residual head 时，使用类内残差近似：

\[
z_m^{r,n}
=
h_m^n-\mu_{m,y^n}^{train}
\]

其中：

\[
\mu_{m,c}^{train}
=
\frac{1}{|\mathcal{S}_{train,c}|}
\sum_{n:y^n=c}h_m^n
\]

## 10.3 残差分布

在每个模态 \(m\)、类别 \(c\)、关系组 \(g\) 下：

\[
P_{m,c}^{r,g}
=
\mathcal{N}(\mu_{m,c}^{r,g},\sigma_{m,c}^{r,g})
\]

使用 diagonal Gaussian 估计。

## 10.4 Residual Distance

\[
D(P,Q)
=
\|\mu_P-\mu_Q\|_2^2
+
\|\sigma_P-\sigma_Q\|_2^2
\]

\[
D_r^g
=
\frac{1}{C}
\sum_c
\frac{1}{3}
\sum_{i<j}
D(P_{i,c}^{r,g},P_{j,c}^{r,g})
\]

## 10.5 Residual Separation

二分类时：

\[
B_m^g
=
\|\mu_{m,Positive}^{r,g}
-
\mu_{m,Negative}^{r,g}\|_2^2
\]

\[
W_m^g
=
\frac{1}{C}
\sum_c
Mean_{n\in g,y^n=c}
\|z_m^{r,n}-\mu_{m,c}^{r,g}\|_2^2
\]

\[
Sep_{r,m}^g
=
\frac{B_m^g}{W_m^g+\epsilon}
\]

三模态平均：

\[
Sep_r^g
=
\frac{1}{3}(Sep_{r,t}^g+Sep_{r,v}^g+Sep_{r,a}^g)
\]

## 10.6 输出表格

| Dataset | Group | Dpred | Dfeat | Residual Dist | Residual Sep ↑ |
|---|---|---:|---:|---:|---:|
| MOSI | RA |  |  |  |  |
| MOSI | RD |  |  |  |  |
| MOSI | ND |  |  |  |  |
| MOSEI | RA |  |  |  |  |
| MOSEI | RD |  |  |  |  |
| MOSEI | ND |  |  |  |  |

## 10.7 理想现象

| Group | 理想现象 | 解释 |
|---|---|---|
| RA | residual dist 低/中，sep 中 | 可靠一致，适合公共对齐 |
| RD | residual dist 高，sep 高 | 可靠不一致中有结构化残差 |
| ND | residual dist 高，sep 低 | 差异大但更像噪声 |

---

# Part 7：Residual Discriminative Probe

## 11.1 新增原因

v3 中 residual diagnostic 只能说明：

> RD 可能有结构化 residual。

但不能说明：

> residual 对任务真的有判别价值。

version4 新增轻量 probe 来回答：

> reliable disagreement 的 residual 是否有助于判断？

## 11.2 Probe 设置

冻结 Concat baseline 或 warm-up encoder 的特征，训练三个轻量分类器：

| Probe | 输入 | 目的 |
|---|---|---|
| Common-only probe | \(z_t^c,z_v^c,z_a^c\) 或 projected common features | 公共语义判别力 |
| Residual-only probe | \(z_t^r,z_v^r,z_a^r\) 或 class residual features | 残差判别力 |
| Common+Residual probe | common + residual | 残差是否补充公共语义 |

如果暂时没有显式 \(z^c,z^r\)，可用：

\[
common_m^n=\mu_{m,y^n}^{train}
\]

\[
residual_m^n=h_m^n-\mu_{m,y^n}^{train}
\]

作为诊断近似。

## 11.3 输出表格

| Dataset | Group | Common-only F1 | Residual-only F1 | Common+Residual F1 | Residual Gain |
|---|---|---:|---:|---:|---:|
| MOSI | RA |  |  |  |  |
| MOSI | RD |  |  |  |  |
| MOSI | ND |  |  |  |  |
| MOSEI | RD |  |  |  |  |

其中：

\[
Residual\ Gain
=
F1_{common+residual}
-
F1_{common-only}
\]

## 11.4 理想结果

| Group | 理想现象 |
|---|---|
| RA | common-only 已经较强，residual gain 不一定大 |
| RD | residual-only 有一定判别力，common+residual 明显优于 common-only |
| ND | residual-only 弱，common+residual gain 小或不稳定 |

如果 RD 上 residual gain 明显高于 ND，就可以支撑：

> Reliable disagreement contains discriminative residual information.

## 11.5 负对照

### Shuffled residual-label probe

打乱 residual 与标签关系，再训练 residual-only probe：

| Dataset | Group | Real Residual-only F1 | Shuffled Residual-only F1 |
|---|---|---:|---:|
| MOSI | RD |  |  |
| MOSEI | RD |  |  |

如果真实 residual 明显优于 shuffled residual，说明 residual 判别力不是随机噪声造成。

---

# Part 8：Negative Controls

## 12.1 Random-D Grouping

保持每组样本数量不变，随机打乱 D 分组：

| Dataset | Grouping | Low-D Delta | Mid-D Delta | High-D Delta |
|---|---|---:|---:|---:|
| MOSI | Real D grouping |  |  |  |
| MOSI | Random grouping |  |  |  |

如果 random grouping 不能复现真实趋势，说明 relation-state grouping 有意义。

## 12.2 Shuffled Residual

打乱 residual 与标签的对应关系，重新计算 residual sep：

| Dataset | Group | Real Residual Sep | Shuffled Residual Sep |
|---|---|---:|---:|
| MOSI | RD |  |  |
| MOSEI | RD |  |  |

## 12.3 Shuffled Relation Weight

在 batch 内打乱 \(g^{agr}\)、\(g^{dis}\)：

| Method | Overall F1 | RA F1 | RD F1 | ND F1 |
|---|---:|---:|---:|---:|
| Real relation weights |  |  |  |  |
| Shuffled relation weights |  |  |  |  |

如果真实 relation weights 更好，说明不是简单加权正则带来的收益。

---

# Part 9：Case Study

## 13.1 目的

补上模型指标与真实样本现象之间的桥梁。

每个样本展示：

- 原始文本；
- 音频描述或 prosody 特征；
- 视觉表情摘要；
- 单模态预测分布；
- \(D_{pred}\)、\(R^{diag}\)；
- relation state；
- Concat / UncondAlign / CoPA 预测；
- residual contribution。

## 13.2 样本类型

至少选：

| 类型 | 数量 | 目的 |
|---|---:|---|
| RA | 2 | 展示可靠一致，适合对齐 |
| UA | 1 | 展示一致但低置信，不应强采纳 |
| RD | 3 | 展示可靠不一致，有判别 residual |
| ND | 2 | 展示高分歧但更像噪声 |
| UncondAlign failure | 2 | 展示无条件对齐失败 |
| CoPA improvement | 2 | 展示 CoPA 改善 |

---

# Part 10：分级成功标准 v4

## Level 0：Concat boundary confirmation

满足：

1. Concat 在 overall 上具备合理性能；
2. 不同 relation group 上 Concat 表现存在差异；
3. 尤其 RD 与 ND，或 RA 与 UA 的表现、方差、残差结构不同。

支持：

> Concat can learn label boundaries, but relation states still reveal meaningful subgroup differences.

## Level 1：relation-dependent alignment

满足任一：

1. Low/Mid/High-D 的 alignment gain 有系统差异；
2. High-D 的方差更大；
3. High-D 在 MOSEI 等数据集上接近 0 或负收益。

支持：

> Alignment gain depends on relation state.

## Level 2：strength-sensitive alignment

满足任一：

1. \(\lambda_{align}\) 增大后多个 D 组退化；
2. 不同 D 组对 \(\lambda\) 敏感性不同；
3. Low-D 并非所有强度下稳定受益。

支持：

> Fixed-strength unconditional alignment is unreliable.

## Level 3：selective agreement

满足任一：

1. Low-D+High-R 优于 Low-D+Low-R；
2. selective agreement prototype 优于 all-agreement prototype；
3. low label-support agreement 会降低 prototype purity 或性能。

支持：

> Consistent samples should not be fully adopted.

## Level 4：reliable/noisy disagreement distinction

满足任一：

1. RD 与 ND 的 delta 或方差不同；
2. RD 与 ND 的 \(D_{feat}\)、residual sep 不同；
3. ND 的 residual probe 表现弱或不稳定。

支持：

> High-D contains both reliable and noisy disagreement.

## Level 5：structured residual

需要：

1. RD 的 residual distance 较高；
2. RD 的 residual sep 高于 ND；
3. shuffled residual 不能复现真实 residual sep。

支持：

> Reliable disagreement may contain structured residual information.

## Level 6：discriminative residual

需要：

1. RD 上 residual-only probe 有明显判别力；
2. RD 上 common+residual 优于 common-only；
3. shuffled residual-label probe 明显下降。

支持：

> Reliable disagreement contains discriminative residual information.

## Level 7：DirectAdd insufficiency

满足任一：

1. DirectAdd 不稳定优于 Concat；
2. DirectAdd 在 RD 或 High-D+High-R 上不如 SoftSplit Probe；
3. DirectAdd 对 \(\alpha\) 敏感；
4. DirectAdd 提升 Overall 但不能解释 RA / UA / RD / ND 的差异。

支持：

> Preserving original features by direct addition is insufficient; relation-conditioned soft decomposition is needed.

## Level 8：支持 CoPA-v4 主方法

需要后续方法实验满足：

1. CoPA-v4 Overall 不弱于 Concat / UncondAlign / Standard Prototype；
2. CoPA-v4 在 RD 或 High-D+High-R 上优于 UncondAlign；
3. w/o selective agreement 下降；
4. w/o disagreement prototype 或 residual loss 下降；
5. case study 能解释可靠不一致样本。

---

# Part 11：失败情况与叙事调整

| 失败情况 | 解释 | 调整 |
|---|---|---|
| D 分组无趋势 | 预测分歧不是有效关系信号 | 换数据集或改用 feature-assisted grouping |
| Low-D+High-R 与 Low-D+Low-R 无差异 | 一致样本筛选证据弱 | 弱化 selective agreement 主张 |
| Concat 各 relation group 无明显差异 | relation state 诊断价值不足 | 不强调 Concat-aware motivation，只保留无条件对齐敏感性 |
| DirectAdd 稳定优于其他诊断 | 直接相加可能已足够 | 将 CoPA 收缩为更轻量的 direct-add regularization 或补更强数据集 |
| RD residual sep 不高 | 不一致未形成结构化残差 | 把 disagreement 仅作为避免错误对齐机制 |
| RD residual probe 无贡献 | 不能说 residual 有判别价值 | 删除 discriminative disagreement 强主张 |
| random grouping 也能复现趋势 | relation grouping 证据不足 | 重新检查分组和阈值 |
| case study 无法解释 | 真实语义证据不足 | 谨慎收缩为 relation-aware regularization |

---

# Part 12：论文中推荐写法

英文：

> We first conduct a relation-state motivation analysis to examine whether paired modalities should be uniformly treated as alignment positives. This analysis does not assume that concatenation fails to learn label-level boundaries; rather, it tests whether relation states reveal meaningful subgroup differences beyond ordinary supervised fusion. Prediction-distribution disagreement is used as a task-level relation indicator, while feature-level residual diagnostics are adopted to examine whether such disagreement corresponds to structured modality-specific variations. We further compare unconditional alignment and a direct-addition baseline, showing that fixed-strength alignment or simply adding aligned summaries back to original features is insufficient to model reliable agreement, reliable disagreement, and unreliable relations. Beyond structural evidence, lightweight residual probes are trained to test whether reliable disagreement contains discriminative information. These findings motivate CoPA to preserve the original supervised fusion path while selectively aligning reliable agreement and learning discriminative residuals from reliable disagreement.

中文：

> 我们首先进行关系状态动机分析，以检验同一样本内的多模态配对是否都应被统一视为对齐正样本。该分析并不假设 Concat 无法学习标签级边界，而是检验关系状态是否能揭示普通监督融合之外的有意义子群差异。我们使用预测分布分歧作为任务层面的关系指示信号，并通过特征层残差诊断检验这种分歧是否对应结构化模态差异。进一步地，我们比较无条件对齐和直接相加 baseline，说明固定强度对齐或简单将对齐摘要加回原特征，并不足以区分可靠一致、可靠不一致和不可靠关系。最后，我们训练轻量残差探针，验证可靠不一致样本中的残差信息是否具有判别价值。这些发现推动 CoPA 保留原始监督融合路径，同时选择性对齐可靠一致样本，并从可靠不一致样本中学习判别性残差。

---

# Part 13：最小执行清单

建议按以下顺序执行：

1. 训练 Concat baseline，保存单模态预测和特征；
2. 训练不同 \(\lambda_{align}\) 的 Unconditional Alignment；
3. 训练不同 \(\alpha\) 的 DirectAdd baseline；
4. 使用 validation threshold 划分 Low/Mid/High-D；
5. 计算 D-group delta；
6. 计算 \(\lambda\)-sensitivity 曲线；
7. 划分 RA / UA / RD / ND；
8. 输出 concat-aware motivation table；
9. 做 selective agreement diagnostic；
10. 计算 \(D_{feat}\) 和 residual sep；
11. 做 residual-only / common-only / common+residual probe；
12. 做 shuffled residual-label probe、random grouping、shuffled residual、shuffled relation weight；
13. 挑选 case study；
14. 再决定 CoPA-v4 主方法是否强讲 discriminative disagreement。

---

# 14. 一句话总结

Experiment 1-v4 的核心是：

> **不是证明 Concat 不会学标签，也不是只证明 High-D 会被对齐伤害，而是证明“同一样本关系有不同学习价值”：Concat 能学习标签边界，但不能显式区分可靠一致、可靠不一致和不可靠关系；直接相加也不足以替代关系条件拆分。可靠一致值得对齐，低可靠一致不应全采纳，可靠不一致可能含有判别残差，低可靠不一致更应被抑制。**
