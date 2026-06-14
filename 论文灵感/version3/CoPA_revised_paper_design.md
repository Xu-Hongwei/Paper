# CoPA version3：收紧后的论文设计

> 方法暂定名：**CoPA-min: Conditional Positive Alignment via Relation-Aware Prototypes**
>
> 论文定位：**监督式多模态融合中的关系条件对齐**
>
> version3 目标：不继续堆模块，而是把论文收紧为一个更稳、更容易被实验证明的问题：**同一样本跨模态配对不应被无条件视为 alignment-positive；跨模态对齐收益依赖 relation state。**

---

## 1. 论文定位

### 1.1 做什么

本文只定位在有监督多模态融合任务：

\[
\{x_t^n, x_v^n, x_a^n, y^n\}_{n=1}^{N}
\]

其中 \(x_t,x_v,x_a\) 分别表示文本、视觉、音频输入，\(y\) 是监督标签。适合的任务包括：

- 多模态情感分析；
- 多模态情绪识别；
- 多模态讽刺检测；
- 多模态幽默检测；
- 其他有明确类别标签或可离散标签的监督式多模态分类/回归任务。

### 1.2 不做什么

本文暂不定位为通用多模态对齐方法，不直接覆盖：

- 图文检索；
- 大规模 VLM 预训练；
- CLIP 式零样本分类；
- 视频-文本检索；
- 无标签或弱标签跨模态对齐。

原因是 CoPA-min 依赖单模态预测分布、监督标签和类别原型，这些条件在上述任务中并不总是自然成立。

---

## 2. 核心问题

### 2.1 传统隐含假设

很多跨模态对齐方法默认：

\[
\text{paired multimodal samples} \Rightarrow \text{alignment-positive samples}
\]

也就是说，只要文本、视觉、音频来自同一个样本，它们就应该被拉近，或被映射到同一个共享语义空间。

### 2.2 version3 的稳健命题

version2 中较强的说法是：

> High-disagreement 样本上无条件对齐会伤害性能。

这个说法有直觉，但现有实验还不足以在所有数据集上强断言。因此 version3 改成更稳的命题：

> **The benefit of multimodal alignment depends on cross-modal relation states. Paired multimodal samples are not always unconditional alignment-positive pairs.**

中文表述：

> **多模态对齐收益依赖跨模态关系状态。同一样本内的多模态配对不应被无条件视为可对齐正样本。**

该命题允许三种结果同时成立：

- Low-disagreement 样本中，无条件对齐可能有收益；
- High-disagreement 样本中，无条件对齐收益可能变弱、不稳定；
- 在部分数据集或设置下，High-disagreement 样本可能出现负收益。

这比“High-D 一定被伤害”更稳，也更符合当前 MOSI/MOSEI 初步结果。

---

## 3. 最稳论文主线

### 3.1 主问题归位

version3 的主线不是“做一个噪声鲁棒模型”，而是：

> 现有方法常把所有 paired modalities 当作正样本对齐，但监督式多模态任务中，同一样本内的跨模态关系并不总是 alignment-positive。

这里有两种不应被强制对齐的情况：

| 情况 | 本文定义 | 处理方式 | 在论文中的地位 |
|---|---|---|---|
| 可靠但不一致 | complementary-positive | 不做强制拉近，保留结构化差异 | 核心主角 |
| 不可靠或受污染 | noisy-positive | 降低其对齐、原型更新和融合影响 | 边界条件 |

因此，本文不是简单地做 noise-aware fusion，而是做 conditional positive alignment：

> 先判断 paired modalities 属于哪种 relation state，再决定对齐、保留差异或降低影响。

### 3.2 噪声不是主角

Reliability 模块容易让论文看起来像噪声问题，因为它会计算：

\[
R_m,\quad g^{noise}=1-R_iR_j
\]

但在 CoPA-min 中，reliability 的主要作用是过滤和分界：

- 区分 reliable disagreement 和 noisy disagreement；
- 防止不可靠模态污染 agreement prototype；
- 防止把噪声误当作 complementary information；
- 让真正的 reliable disagreement 可以被单独分析。

也就是说，噪声处理是 conditional alignment 的边界条件，而不是论文主贡献。

### 3.3 真正要证明的核心

论文最需要证明的不是：

> 模态有噪声，所以要降权。

而是：

> 在监督式多模态任务中，存在可靠但不一致的 paired samples；这些样本不应被无条件对齐，其中的残差信息可能具有判别价值。

对应到实验上，最关键的是：

- High-D + High-R 样本是否存在；
- High-D + High-R 是否不是简单噪声；
- residual branch 是否在 High-D + High-R 上有贡献；
- complementary prototype / separation 是否比普通原型更有效；
- case study 是否能展示真实跨模态反差。

如果这些证据成立，CoPA 的主张就是“条件正样本对齐”。如果这些证据不成立，论文应主动收缩为“relation-aware alignment regularization”，而不是继续堆噪声鲁棒模块。

### 3.4 最稳叙事句

推荐全文围绕这一句展开：

> Existing multimodal alignment methods often treat paired modalities as unconditional positive pairs. However, supervised multimodal samples may contain both reliable complementary disagreement and unreliable noisy signals. CoPA first estimates the cross-modal relation state, then aligns only agreement-positive pairs, preserves reliable complementary differences, and suppresses unreliable pairs from contaminating prototype learning.

中文版本：

> 现有多模态对齐方法常把同一样本内的模态配对视为无条件正样本，但监督式多模态任务中既可能存在可靠互补的不一致，也可能存在不可靠噪声。CoPA 先估计跨模态关系状态，只对可靠一致样本进行对齐，对可靠不一致样本保留差异，并降低不可靠样本对原型学习的污染。

---

## 4. 当前证据边界

### 4.1 已有初步现象

当前已有 MOSI/MOSEI 5 seeds 诊断结果可以作为 motivation evidence，但不能当作最终结论。

| Dataset | Low-D delta Macro-F1 | Mid-D delta Macro-F1 | High-D delta Macro-F1 | 可支持的结论 |
|---|---:|---:|---:|---|
| MOSI | +0.0468 | +0.0145 | +0.0213 | High-D 收益弱于 Low-D，但未变负 |
| MOSEI | +0.0129 | +0.0033 | -0.0006 | High-D 出现轻微负收益 |

其中 delta 表示：

\[
\Delta = \text{Unconditional Alignment} - \text{Concat}
\]

### 4.2 现阶段不能过度声明

不能直接写：

> 无条件对齐一定伤害 high-disagreement 样本。

更合理写法是：

> 无条件对齐的收益并不是跨样本关系状态无关的；在高分歧样本中，对齐收益明显减弱，并在更大规模 MOSEI 上出现退化。

### 4.3 还缺的关键证据

本文真正需要补强的是：

> High-D + High-R 是否代表有判别力的 reliable complementary information？

仅凭单模态预测分布的 JSD 和熵，最多能证明“模型预测状态不同”，还不能完全证明“原始多模态语义存在可靠互补冲突”。因此必须补：

- High-D + High-R 专门消融；
- residual branch 是否贡献性能；
- complementary prototype 是否比普通 prototype 更有效；
- 原始样本 case study。

### 4.4 样本对齐判断的边界

当前的 disagreement / agreement 判断基于单模态预测分布。它首先衡量的是：

> 不同模态在任务标签空间中的预测是否一致。

这并不等价于严格的原始语义一致。两个模态可能都高置信预测同一错误类别，此时预测分布一致，但它们不应被当作可靠 agreement-positive 样本更新原型。

因此 version3 需要区分两种用途：

| 用途 | 是否使用真实标签 | 推荐 reliability | 原因 |
|---|---|---|---|
| Motivation 分组 | 不使用 test label | entropy-based reliability | 避免测试集标签泄漏 |
| 训练 CoPA 原型 | 使用 train label | label-aware reliability | 防止高置信错误污染原型 |

这一点是 version3 中判断“样本是否适合对齐”的关键修正。

---

## 5. 为什么这个动机不是硬凑

### 5.1 合理之处

多模态监督任务中，跨模态关系天然不总是一致：

- 情感任务中，文本表面积极但语音低沉；
- 讽刺任务中，文字含义和语气反差本身是线索；
- 幽默任务中，多模态错位可能构成判别信息；
- 噪声场景中，视觉遮挡或音频污染不应参与强对齐。

因此，“paired 不等于 unconditional alignment-positive”有现实基础。

### 5.2 风险之处

风险不在问题本身，而在证明路径：

- 如果只用模型预测分布定义 disagreement，容易被质疑是模型内部现象；
- 如果模块太多，容易被认为是把 DecAlign、UDML、ProtoMM 等方法拼起来；
- 如果没有 High-D + High-R 证据，complementary-positive 会显得像概念包装。

### 5.3 version3 的控制策略

因此 version3 做三点收缩：

1. 主张从“强制对齐必然伤害”改为“对齐收益依赖 relation state”；
2. 方法从 full CoPA 收缩到 CoPA-min；
3. 把原始样本 case study 和 High-D + High-R 消融作为必须证据。

---

## 6. 方法主线：CoPA-min

### 6.1 输入与编码

对每个模态 \(m \in \{t,v,a\}\)，先得到单模态表示：

\[
h_m = E_m(x_m)
\]

再通过两个投影头得到：

\[
z_m^c = P_m^c(h_m), \quad z_m^r = P_m^r(h_m)
\]

其中：

- \(z_m^c\)：候选公共语义，用于可靠一致样本；
- \(z_m^r\)：候选残差信息，用于可靠不一致样本。

第一版实现可以使用标准预提取特征，不做端到端原始数据训练。

### 6.2 关系状态估计

每个模态接单模态分类头：

\[
p_m = \text{Softmax}(C_m(z_m^c))
\]

#### 6.2.1 诊断阶段：entropy-based reliability

在 motivation diagnostic 和 test-set 分组中，为避免使用测试标签，使用预测熵估计可靠性：

\[
R_m = 1 - \frac{H(p_m)}{\log K}
\]

它表示该模态预测是否确定，但不保证该预测一定正确。因此它适合作为无标签诊断指标，不适合单独作为训练原型的可靠性依据。

#### 6.2.2 训练阶段：label-aware reliability

在监督训练阶段，训练标签 \(y\) 已知。为了避免高置信错误模态污染原型，CoPA-min 使用 label-aware reliability：

\[
C_m = 1 - \frac{H(p_m)}{\log K}
\]

\[
S_m = p_m(y)
\]

\[
R_m^{label} = C_m \cdot S_m
\]

其中：

- \(C_m\)：confidence，表示预测是否确定；
- \(S_m\)：label support，表示该模态是否支持真实标签；
- \(R_m^{label}\)：训练原型时使用的可靠性。

这样可以避免两类误判：

1. 高置信但预测错误的模态被当作 reliable；
2. 多个模态一起预测错误时被当作 agreement-positive。

#### 6.2.3 跨模态一致性

用 Jensen-Shannon Divergence 估计跨模态分歧，并得到一致性：

\[
A_{ij} = \exp(-\text{JSD}(p_i,p_j)/\tau_A)
\]

#### 6.2.4 关系权重

三类关系权重：

\[
g_{ij}^{agr}=R_iR_jA_{ij}
\]

\[
g_{ij}^{comp}=R_iR_j(1-A_{ij})
\]

\[
g_{ij}^{noise}=1-R_iR_j
\]

其中 \(R_i,R_j\) 的选择取决于用途：

- motivation diagnostic：使用 entropy-based \(R_m\)；
- supervised prototype training：使用 label-aware \(R_m^{label}\)。

训练中 \(g^{agr}\) 和 \(g^{comp}\) 使用 stop-gradient，避免模型通过操纵关系权重逃避损失。

### 6.3 关系感知双原型

对每个模态 \(m\)、类别 \(c\)，维护两个 EMA 原型：

\[
P_{m,c}^{agr}, \quad P_{m,c}^{comp}
\]

含义：

| 原型 | 使用特征 | 样本来源 | 作用 |
|---|---|---|---|
| \(P_{m,c}^{agr}\) | \(z_m^c\) | reliable agreement | 稳定公共语义 |
| \(P_{m,c}^{comp}\) | \(z_m^r\) | reliable disagreement | 保留有类别结构的模态差异 |

### 6.4 CoPA-min 损失

CoPA-min 只保留必要项：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}^{proto}
+
\lambda_2\mathcal{L}_{comp}^{sep}
\]

其中：

- \(\mathcal{L}_{task}\)：主任务分类或回归损失；
- \(\mathcal{L}_{agr}^{proto}\)：可靠一致样本靠近对应 agreement prototype；
- \(\mathcal{L}_{comp}^{sep}\)：可靠不一致样本的不同模态 complementary prototype 保持最小间隔。

### 6.5 暂缓项

以下内容不作为 version3 核心方法，只作为 optional extension：

- relation-conditioned fusion；
- \(\mathcal{L}_{orth}\)；
- \(\mathcal{L}_{uni}\)；
- 完整 \(\mathcal{L}_{agr}^{cross}\)；
- 完整 \(\mathcal{L}_{comp}^{proto}\)。

只有当 CoPA-min 结果站稳后，才逐步加入这些增强项。

---

## 7. Full CoPA 的位置

Full CoPA 不是第一篇实验的起点，而是后续增强版本。建议论文中按如下方式处理：

| 版本 | 定位 | 是否作为主方法 |
|---|---|---|
| Concat | 基础融合 baseline | 否 |
| Unconditional Alignment | 反例 baseline | 否 |
| Standard Prototype | 普通原型 baseline | 否 |
| CoPA-min | 核心方法 | 是 |
| Full CoPA | 增强版本或 appendix | 可选 |

这样可以降低“模块堆砌”的观感。

---

## 8. 原始数据使用边界

### 8.1 主 benchmark

MOSI/MOSEI 主实验继续使用标准预提取特征：

- Text：GloVe / BERT 类文本特征；
- Visual：OpenFace 类视觉特征；
- Audio：COVAREP 类音频特征。

这样更利于和已有方法对比，也降低实现成本。

### 8.2 原始数据或可解释样本

原始数据更适合用于：

- motivation case study；
- 文本内容展示；
- 语音/视觉冲突解释；
- 噪声与缺失实验；
- 证明 High-D + High-R 不只是模型预测分歧。

### 8.3 暂不做端到端训练

第一阶段不建议从原始文本、音频、视频端到端训练 CoPA。这样会把论文难度从“关系条件对齐”变成“复杂多模态表征学习系统”，容易掩盖核心问题。

---

## 9. 投稿叙事建议

### 9.1 Introduction

推荐叙事顺序：

1. 多模态融合常用跨模态对齐提升共享语义；
2. 现有方法常默认 paired samples 是 alignment-positive；
3. 但监督式多模态任务中，同一样本跨模态可能一致、互补或不可靠；
4. 诊断实验显示，无条件对齐收益依赖 disagreement 状态，高分歧样本上收益变弱或退化；
5. 因此提出 CoPA-min，根据 relation state 做条件正样本对齐。

### 9.2 Method

Method 不要从复杂模块开始，而要围绕一个问题展开：

> Given a paired multimodal sample, should its modalities be aligned, preserved as complementary, or suppressed as unreliable?

对应三步：

1. estimate relation state；
2. maintain relation-aware prototypes；
3. optimize agreement alignment and complementary preservation conditionally。

### 9.3 Experiments

实验必须按“证明问题 -> 验证方法 -> 排除硬凑”展开：

1. Disagreement diagnostic；
2. CoPA-min main comparison；
3. High-D + High-R complementary evidence；
4. Ablation；
5. Case study；
6. Robustness。

---

## 10. 失败时的叙事调整

### 10.1 如果 CoPA-min 提升 Overall，但不提升 High-D

论文主张收缩为：

> Relation-aware prototypes improve alignment robustness as a regularizer.

不要强讲 complementary-positive。

### 10.2 如果 High-D + High-R 中 residual 没有贡献

说明 reliable disagreement 不一定是有用互补信息，应该：

- 删除或弱化 complementary-positive 贡献；
- 把 \(g^{comp}\) 作为避免错误对齐的权重，而不是显式互补建模；
- 保留 agreement selective alignment 作为核心。

### 10.3 如果无条件对齐在所有分组都提升

说明当前任务或设置中 conditional alignment 必要性不足，应该：

- 增强 disagreement 场景；
- 加入讽刺/幽默数据集；
- 改用更强 alignment baseline；
- 调整论文为 robustness 或 noisy modality setting。

### 10.4 如果无条件对齐在所有分组都下降

说明 sample-level alignment baseline 可能太弱或太粗糙，应该：

- 加入 standard class prototype baseline；
- 调低 alignment loss；
- 使用 prototype-level alignment 作为更强反例；
- 避免把问题归因到 high-disagreement。

---

## 11. version3 一句话总结

> CoPA-min 不再试图证明“高分歧样本一定被对齐伤害”，而是证明“跨模态对齐不应无条件施加；其收益依赖样本级关系状态”。通过关系状态估计和关系感知原型，模型对可靠一致样本进行选择性对齐，对可靠不一致样本保留结构化差异，并降低不可靠模态对训练和融合的影响。
