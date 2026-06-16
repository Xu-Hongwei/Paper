# CoPA version5：Relation-Gated Contrastive Learning for Selective Agreement and Discriminative Disagreement

> 暂定题目：**CoPA: Relation-Gated Contrastive Learning for Selective Agreement and Discriminative Disagreement in Supervised Multimodal Fusion**  
> 中文定位：**监督式多模态融合中的关系门控对比学习：选择性一致对齐与判别性不一致学习**  
> 方法简称：**CoPA-v5 / CoPA-NCE**

---

## 0. version5 相比 version4 的核心变化

version4 的核心思想是：

\[
\text{same-sample / same-label} \neq \text{unconditional alignment-positive}
\]

并进一步提出：

\[
\text{disagreement} \neq \text{noise}
\]

version5 在这个基础上做一次关键修改：

> **预测分布不再被直接当作判别信息，而只作为 relation gate；真正的表示学习交给 InfoNCE / SupCon。**

也就是说，v5 从：

\[
\text{prediction disagreement} + \text{prototype separation}
\]

改为：

\[
\text{prediction-based relation gate} + \text{relation-gated contrastive representation learning}
\]

### 0.1 改动对照表

| 项目 | v4 设计 | v5 设计 |
|---|---|---|
| 预测分布作用 | 判断 RA / RD / UA / ND，并参与 prototype 权重 | 只做 soft relation gate，不直接作为判别特征 |
| RA 处理 | 可靠一致样本更新 common prototype，做公共对齐 | 可靠一致样本做 **gated common InfoNCE** |
| RD 处理 | 可靠不一致样本做 complementary prototype / residual separation | 可靠不一致样本做 **residual InfoNCE / residual SupCon** |
| Prototype 地位 | 主体约束之一 | 辅助 memory / prototype NCE，用于稳定 batch 内正样本不足 |
| 无条件 InfoNCE | 未作为核心对照 | 新增 baseline，用来证明普通 same-sample positive 不够稳 |
| 论文主张 | 选择性一致 + 判别性不一致 | 关系门控对比学习：RA 学公共语义，RD 学残差结构 |

### 0.2 为什么要改成 InfoNCE

早期 v4 / prototype 多 seed 实验说明：

1. CoPA 目前在 overall 上有小幅稳定提升；
2. 但 relation-state 中 RD 的效果没有稳定打出来；
3. 无条件对齐和 DirectAdd/TextInject 类方法在 High-D 上会出现稳定伤害；
4. 因此问题不是“要不要对齐”，而是“哪些样本适合对齐，哪些样本应该学习差异”。

所以 v5 的核心设计是：

\[
\text{Reliable Agreement} \Rightarrow \text{alignment-positive}
\]

\[
\text{Reliable Disagreement} \Rightarrow \text{residual-positive, not alignment-positive}
\]

\[
\text{Uncertain / Noisy Relations} \Rightarrow \text{relation learning down-weighted}
\]

### 0.3 当前代码实现状态

当前 `code/disagreement_phenomenon` 还没有实现完整 CoPA-v5 主方法。它实现的是 **CoPA-v5 motivation evidence loop**，也就是先验证：

1. 普通 same-sample alignment / InfoNCE 的收益是否依赖 relation state；
2. text-anchor residual 是否至少具有补充诊断价值；
3. text 作为语义锚点时，是否比三模态全互相对齐更符合任务假设。

当前代码主线为：

```text
three_class + text_anchor + balanced_within_d
```

其中：

- `Concat` 是监督融合基线；
- `UncondAlign` 是普通同样本距离/余弦拉近；
- `UncondInfoNCE` 更准确地说是 **UncondPairInfoNCE**：普通 pairwise same-sample InfoNCE baseline；
- `TextInject` 是原 text-anchor DirectAdd 的重新命名，只作为 appendix/diagnostic；
- `BalancedDirectAdd` 是更公平的 appendix baseline，不进入主动机表；
- residual probe 是补充诊断，不再作为 v5 主命题的唯一成功条件。

因此本文档后面的 CoPA-v5 主方法仍是下一阶段设计；当前已落地代码主要服务于“为什么需要 relation-gated alignment”的动机闭环。

---

## 1. 论文最终定位

### 1.1 任务范围

CoPA-v5 定位于**监督式多模态融合任务**，输入形式为：

\[
\{x_t^n,x_v^n,x_a^n,y^n\}_{n=1}^{N}
\]

其中：

- \(x_t\)：文本模态；
- \(x_v\)：视觉模态；
- \(x_a\)：音频模态；
- \(y\)：样本级监督标签。

适用任务包括：

- 多模态情感分析；
- 多模态情绪识别；
- 多模态讽刺检测；
- 多模态幽默检测；
- 其他有明确监督标签的多模态分类或回归任务。

### 1.2 不直接定位为通用 VLM 对齐方法

CoPA-v5 不直接定位为：

- CLIP 式大规模图文预训练；
- 图文检索；
- 零样本分类；
- 无监督跨模态匹配；
- 大规模生成模型条件对齐。

原因是 CoPA-v5 依赖：

1. 单模态预测分布；
2. 样本级监督标签；
3. 任务标签空间中的跨模态证据关系；
4. relation-gated supervised / contrastive objectives。

因此本文的核心不是通用跨模态检索对齐，而是：

> **监督式多模态融合中，同一样本内部的模态关系不应被无条件视为对齐正样本。**

---

## 2. 核心问题与论文主张

### 2.1 被质疑的隐含假设

很多跨模态对齐方法隐含使用：

\[
\text{same sample} \Rightarrow \text{positive pair}
\]

也就是：

\[
(x_i^n,x_j^n) \Rightarrow z_i^n \text{ and } z_j^n \text{ should be aligned}
\]

在普通 InfoNCE 中，这通常表现为：

\[
z_i^n \leftrightarrow z_j^n \quad \text{positive}
\]

\[
z_i^n \leftrightarrow z_j^k, k\neq n \quad \text{negative}
\]

但在情感、讽刺、幽默等监督式多模态任务中，同一样本内部可能存在：

- 文本表面积极，但语音低沉；
- 语言表达中性，但视觉表情明显负面；
- 音频和视觉支持真实情绪，但文本存在反讽；
- 某个模态受噪声干扰导致预测不可靠。

因此，same-sample positive 不是绝对成立的。

### 2.2 version5 核心观点

CoPA-v5 的核心观点是：

\[
\text{same sample} \neq \text{unconditional alignment-positive}
\]

\[
\text{prediction agreement} \neq \text{reliable agreement}
\]

\[
\text{prediction disagreement} \neq \text{noise}
\]

更完整地说：

> **在监督式多模态融合中，同一样本内部的跨模态关系具有条件性。可靠一致样本适合学习公共语义对齐；可靠不一致样本不应被强制拉近，而应学习其跨模态残差结构；不可靠样本不应污染对齐或残差原型。**

### 2.3 推荐摘要句

英文：

> Existing multimodal alignment methods often treat paired modalities as unconditional positive pairs. However, in supervised multimodal fusion, task-level agreement is not always reliable, and task-level disagreement is not always noise. CoPA estimates instance-level relation gates from unimodal prediction distributions, selectively aligns reliable agreement samples with gated InfoNCE, and learns discriminative residual structures from reliable disagreement samples through residual contrastive learning.

中文：

> 现有多模态对齐方法常将同一样本内的模态配对视为无条件正样本。然而，在监督式多模态融合中，任务层面的一致不一定可靠，任务层面的不一致也不一定是噪声。CoPA 基于单模态预测分布估计样本级关系门控，对可靠一致样本进行门控式 InfoNCE 公共语义对齐，并通过残差对比学习从可靠不一致样本中学习判别性跨模态差异。

---

## 3. 术语定义

### 3.1 不再说“真实标签不一致的同一样本”

在 MOSI/MOSEI 等数据集中，每个样本只有一个样本级标签：

\[
(x_t^n,x_v^n,x_a^n,y^n)
\]

严格来说，同一样本内部不存在独立的真实模态标签：

\[
y_t^n,y_v^n,y_a^n
\]

因此不要写：

> 同一样本中真实标签不一致。

更准确的写法是：

> 同一样本内部存在跨模态任务证据不一致。

### 3.2 推荐术语

| 日常说法 | 论文表述 | 含义 |
|---|---|---|
| 标签一致 | task-evidence agreement / prediction agreement | 不同模态对任务标签的预测分布相近 |
| 标签不一致 | task-evidence disagreement / prediction disagreement | 不同模态对任务标签的预测分布差异大 |
| 可靠一致 | reliable agreement, RA | 预测分布一致且预测可靠 |
| 不可靠一致 | uncertain agreement, UA | 预测分布一致但预测不可靠 |
| 可靠不一致 | reliable disagreement, RD | 预测分布差异大且预测可靠 |
| 噪声不一致 | noisy disagreement, ND | 预测分布差异大但至少部分模态不可靠 |

---

## 4. 样本关系状态

对任意两个模态 \(i,j\)，CoPA-v5 将同一样本内部关系划分为四类：

| 状态 | 预测分歧 | 可靠性 | 是否拉近 | 是否学习残差 | 处理方式 |
|---|---:|---:|---|---|---|
| Reliable Agreement, RA | 低 | 高 | 是 | 否 | common InfoNCE |
| Uncertain Agreement, UA | 低 | 低 | 否 | 否 | 只保留 task loss |
| Reliable Disagreement, RD | 高 | 高 | 否 | 是 | residual InfoNCE / SupCon |
| Noisy Disagreement, ND | 高 | 低 | 否 | 否 | 降低 relation learning 权重 |

核心规则：

\[
RA \Rightarrow \text{alignment-positive}
\]

\[
RD \Rightarrow \text{residual-positive}
\]

\[
UA,ND \Rightarrow \text{relation learning down-weighted}
\]

---

## 5. 方法总览

CoPA-v5 包含四个核心模块：

| 模块 | 名称 | 作用 |
|---|---|---|
| Module 1 | Common-Residual Representation Decomposition | 将模态表示拆成公共语义和残差信息 |
| Module 2 | Instance-level Relation Gate Estimation | 基于预测分布估计 RA / RD / UA / ND soft gate |
| Module 3 | Relation-Gated Common InfoNCE | 只对可靠一致样本进行公共语义对齐 |
| Module 4 | Residual Contrastive Disagreement Learning | 只对可靠不一致样本学习判别性残差结构 |

整体流程：

\[
\text{Unimodal Features}
\Rightarrow
\text{Common / Residual Split}
\Rightarrow
\text{Prediction Distribution}
\Rightarrow
\text{Relation Gates}
\Rightarrow
\begin{cases}
RA: \mathcal{L}_{agr}^{NCE}\\
RD: \mathcal{L}_{dis}^{NCE}\\
UA/ND: \mathcal{L}_{task}\ \text{only}
\end{cases}
\]

---

# Module 1：Common-Residual Representation Decomposition

## 1.1 单模态编码

给定三模态输入：

\[
x_t^n,x_v^n,x_a^n
\]

使用模态编码器得到：

\[
h_m^n=E_m(x_m^n),\quad m\in\{t,v,a\}
\]

其中 \(E_m\) 可以是：

- LSTM / GRU；
- Transformer encoder；
- MLP over pretrained features；
- MMSA / MultiBench 中常用的 feature encoder。

## 1.2 公共表示与残差表示

每个模态表示经过两个投影头：

\[
z_m^{c,n}=P_m^c(h_m^n)
\]

\[
z_m^{r,n}=P_m^r(h_m^n)
\]

其中：

| 表示 | 作用 |
|---|---|
| \(z_m^c\) | 公共语义表示，用于 reliable agreement alignment |
| \(z_m^r\) | 残差表示，用于 reliable disagreement residual learning |

## 1.3 融合预测

主任务仍然保留原始监督融合路径：

\[
z_{fusion}^n=F([h_t^n;h_v^n;h_a^n])
\]

\[
\hat y^n=C_{fusion}(z_{fusion}^n)
\]

\[
\mathcal{L}_{task}=CE(\hat y^n,y^n)
\]

对于回归任务，可替换为：

\[
\mathcal{L}_{task}=MAE/MSE
\]

但为了关系分组和 InfoNCE，建议把情感分数离散化为二分类或多分类辅助标签。

## 1.4 单模态预测头

为了估计关系状态，每个模态接一个单模态预测头：

\[
p_m^n=\text{Softmax}(C_m(z_m^{c,n}))
\]

单模态辅助损失：

\[
\mathcal{L}_{uni}=\frac{1}{3}\sum_{m\in\{t,v,a\}}CE(p_m^n,y^n)
\]

单模态预测头的作用不是最终预测，而是提供 relation gate。

---

# Module 2：Instance-level Relation Gate Estimation

## 2.1 Prediction-level disagreement

对模态对 \((i,j)\)：

\[
D_{ij}^n=JSD(p_i^n,p_j^n)
\]

其中 Jensen-Shannon Divergence 为：

\[
JSD(p_i,p_j)=\frac{1}{2}KL(p_i||m)+\frac{1}{2}KL(p_j||m)
\]

\[
m=\frac{1}{2}(p_i+p_j)
\]

三模态 full-pair 样本级平均分歧：

\[
D_{pred}^n=\frac{1}{3}\left[D_{ta}^n+D_{tv}^n+D_{av}^n\right]
\]

当前 motivation 主线使用 text-anchor 版本：

\[
D_{pred}^{ta/tv,n}=\frac{1}{2}\left[D_{ta}^n+D_{tv}^n\right]
\]

原因是 text 在 MOSI/MOSEI 中通常承担主要语义锚点，audio 与 vision 未必应该被强制互相对齐。`full_pair` 可以作为 appendix，对应额外加入 \(D_{av}\) 和 A-V 对齐/InfoNCE。

注意：

\[
D_{pred}
\]

只能解释为：

> task-level cross-modal evidence disagreement

不能直接解释为真实语义冲突或互信息。

## 2.2 Agreement score

将分歧转成一致性分数：

\[
A_{ij}^n=\exp(-D_{ij}^n/\tau_A)
\]

其中：

- \(A_{ij}^n\) 越大，表示两个模态预测分布越一致；
- \(A_{ij}^n\) 越小，表示两个模态预测分布越不一致。

## 2.3 Diagnostic reliability

用于 motivation analysis 和 test 分组，不能使用 test label：

\[
Q_m^n=1-\frac{H(p_m^n)}{\log K}
\]

其中：

\[
H(p_m^n)=-\sum_{c=1}^{K}p_m^n(c)\log p_m^n(c)
\]

\(Q_m^n\) 只表示预测分布是否尖锐，不保证预测正确。

## 2.4 Label-aware reliability

训练集上可以使用标签支持度：

\[
S_m^n=p_m^n(y^n)
\]

\[
R_m^{label,n}=Q_m^nS_m^n
\]

它同时要求：

1. 预测分布低熵；
2. 预测分布支持真实标签。

作用是防止高置信错误样本污染 RA 对齐。

## 2.5 Reliable agreement gate

第一版无标签门控：

\[
g_{ij}^{agr,n}=Q_i^nQ_j^nA_{ij}^n
\]

训练时更推荐 label-aware gate：

\[
g_{ij}^{agr,n}=Q_i^nQ_j^nS_i^nS_j^nA_{ij}^n
\]

含义：

> 两个模态都可靠、都支持标签、预测分布一致时，才强烈参与 common InfoNCE。

## 2.6 Reliable disagreement gate

不推荐直接使用 \(S_iS_j\)，因为 reliable disagreement 中可能存在“一个模态支持标签，另一个模态提供反差信息”的情况。

推荐定义：

\[
B_{ij}^{label,n}=\max(S_i^n,S_j^n)
\]

\[
g_{ij}^{dis,n}=Q_i^nQ_j^nB_{ij}^{label,n}(1-A_{ij}^n)
\]

无标签诊断版本：

\[
g_{ij}^{dis,n}=Q_i^nQ_j^n(1-A_{ij}^n)
\]

含义：

> 两个模态都比较自信，但任务证据明显不同，并且至少一个模态支持标签，则认为存在可学习的 reliable disagreement。

## 2.7 Stop-gradient rule

所有 gate 在损失中使用时 detach：

\[
\tilde g_{ij}^{agr,n}=\text{stopgrad}(g_{ij}^{agr,n})
\]

\[
\tilde g_{ij}^{dis,n}=\text{stopgrad}(g_{ij}^{dis,n})
\]

防止模型通过操纵预测分布来逃避对比损失。

---

# Module 3：Relation-Gated Common InfoNCE

## 3.1 普通 InfoNCE 的问题

普通 cross-modal InfoNCE 对模态 \(i\rightarrow j\) 写作：

\[
\ell_{i\rightarrow j}^{n}
=
-\log
\frac{
\exp(\text{sim}(z_i^n,z_j^n)/\tau)
}{
\sum_{k=1}^{B}\exp(\text{sim}(z_i^n,z_j^k)/\tau)
}
\]

它默认：

\[
\text{same sample} \Rightarrow \text{positive pair}
\]

但 CoPA 认为这个假设过强。

## 3.2 Reliable agreement common InfoNCE

CoPA-v5 只在 reliable agreement 上把同样本模态当作正样本。

对模态 \(i\rightarrow j\)：

\[
\ell_{i\rightarrow j}^{agr,n}
=
-\tilde g_{ij}^{agr,n}
\log
\frac{
\exp(\text{sim}(z_i^{c,n},z_j^{c,n})/\tau_c)
}{
\sum_{k=1}^{B}\omega_{nk}^{neg}
\exp(\text{sim}(z_i^{c,n},z_j^{c,k})/\tau_c)
}
\]

其中 \(\omega_{nk}^{neg}\) 可有两种版本。

### 版本 A：batch 内所有样本作为分母

\[
\omega_{nk}^{neg}=1
\]

这是最简单版本。

### 版本 B：排除同类样本，减少 false negative

\[
\omega_{nk}^{neg}=\mathbb{1}(y_k\neq y_n)+\mathbb{1}(k=n)
\]

也就是：

- 同一样本正对保留；
- 不同类别样本作为负样本；
- 同类别不同样本不作为强负样本。

第一版建议采用版本 B，更适合监督式情感/情绪任务。

## 3.3 双向 InfoNCE

\[
\mathcal{L}_{ij}^{agr}
=
\frac{1}{2}
\left(
\mathcal{L}_{i\rightarrow j}^{agr}
+
\mathcal{L}_{j\rightarrow i}^{agr}
\right)
\]

text-anchor 主线总损失：

\[
\mathcal{L}_{agr}^{NCE}
=
\frac{1}{2}
\left(
\mathcal{L}_{ta}^{agr}
+
\mathcal{L}_{tv}^{agr}
\right)
\]

full-pair appendix 才使用三对：

\[
\mathcal{L}_{agr,full}^{NCE}
=
\frac{1}{3}
\left(
\mathcal{L}_{ta}^{agr}
+
\mathcal{L}_{tv}^{agr}
+
\mathcal{L}_{av}^{agr}
\right)
\]

## 3.4 直觉

普通 InfoNCE：

\[
\text{same sample} \Rightarrow \text{pull close}
\]

CoPA-v5：

\[
\text{same sample + reliable agreement} \Rightarrow \text{pull common semantics close}
\]

RD、UA、ND 不被强制拉近。

---

# Module 4：Residual Contrastive Disagreement Learning

## 4.1 RD 不应被强制对齐

对于 reliable disagreement，CoPA-v5 不做：

\[
z_i^n \rightarrow z_j^n
\]

因为这会破坏跨模态反差信息。

而是构造残差表示：

\[
r_{ij}^n=R_{ij}(z_i^{r,n},z_j^{r,n})
\]

最简单形式：

\[
r_{ij}^n=|z_i^{r,n}-z_j^{r,n}|
\]

更推荐形式：

\[
r_{ij}^n=MLP([z_i^{r,n};z_j^{r,n};|z_i^{r,n}-z_j^{r,n}|])
\]

在 text-anchor 主线中，残差只使用：

\[
r_{ta}^n,\quad r_{tv}^n
\]

不把 \(r_{av}\) 放入主 residual objective。A-V residual 可以保留为 diagnostic/appendix 对照，避免把 audio-vision 的潜在方向冲突强行纳入主路径。

## 4.2 Residual SupCon

对可靠不一致 residual 做监督式对比学习：

\[
\ell_{ij}^{dis,n}
=
-\tilde g_{ij}^{dis,n}
\log
\frac{
\sum_{k\neq n,y_k=y_n}\tilde g_{ij}^{dis,k}
\exp(\text{sim}(r_{ij}^n,r_{ij}^k)/\tau_r)
}{
\sum_{k\neq n}
\exp(\text{sim}(r_{ij}^n,r_{ij}^k)/\tau_r)
}
\]

含义：

> 同类别的 reliable disagreement residual 应该具有相似的反差结构；不同类别的 residual 应该分开。

## 4.3 Prototype Residual NCE

如果 batch 内同类 RD 样本太少，Residual SupCon 会不稳定。此时使用类别 residual prototype：

\[
P_c^{dis}
\]

用 EMA 更新：

\[
P_{y_n}^{dis}\leftarrow \mu P_{y_n}^{dis}+(1-\mu)r_{ij}^n
\]

更新时按 \(\tilde g_{ij}^{dis,n}\) 加权。

Prototype NCE：

\[
\ell_{ij}^{dis,n}
=
-\tilde g_{ij}^{dis,n}
\log
\frac{
\exp(\text{sim}(r_{ij}^n,P_{y_n}^{dis})/\tau_r)
}{
\sum_{c=1}^{K}
\exp(\text{sim}(r_{ij}^n,P_{c}^{dis})/\tau_r)
}
\]

第一版建议优先实现 Prototype Residual NCE，因为它更稳定、实现更简单。

## 4.4 text-anchor residual loss

\[
\mathcal{L}_{dis}^{NCE}
=
\frac{1}{2}
\left(
\mathcal{L}_{ta}^{dis}
+
\mathcal{L}_{tv}^{dis}
\right)
\]

full-pair appendix 可再加入 \(\mathcal{L}_{av}^{dis}\)，但不作为默认论文主线。

## 4.5 直觉

RA 学的是：

\[
\text{what is common across modalities}
\]

RD 学的是：

\[
\text{what is different but discriminative across modalities}
\]

---

## 6. 总体优化目标

最终损失：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_u\mathcal{L}_{uni}
+
\lambda_a\mathcal{L}_{agr}^{NCE}
+
\lambda_d\mathcal{L}_{dis}^{NCE}
\]

其中：

| 损失 | 作用样本 | 作用 |
|---|---|---|
| \(\mathcal{L}_{task}\) | 所有样本 | 保留原始监督融合能力 |
| \(\mathcal{L}_{uni}\) | 所有样本 | 训练单模态预测头，提供 relation gate |
| \(\mathcal{L}_{agr}^{NCE}\) | RA 权重大 | 学公共语义对齐 |
| \(\mathcal{L}_{dis}^{NCE}\) | RD 权重大 | 学判别性残差结构 |

UA / ND：

\[
UA,ND \Rightarrow \mathcal{L}_{task}\ \text{only or very low relation weight}
\]

---

## 7. 训练流程

### Stage 1：Warm-up

训练：

\[
\mathcal{L}_{task}+\lambda_u\mathcal{L}_{uni}
\]

不启用：

\[
\mathcal{L}_{agr}^{NCE},\quad \mathcal{L}_{dis}^{NCE}
\]

目的：

- 让单模态预测分布 \(p_m\) 不再随机；
- 让 \(D_{ij}\)、\(A_{ij}\)、\(Q_m\) 有基本意义；
- 避免早期错误 gate 污染对比学习。

推荐 warm-up：

\[
5\sim 10\ \text{epochs}
\]

### Stage 2：Prototype / memory initialization

用 warm-up 后的训练集特征初始化：

- common prototype / memory；
- residual prototype / memory。

优先初始化：

\[
P_c^{dis}
\]

因为 RD residual NCE 对 batch 正样本较敏感。

### Stage 3：Joint training

训练完整损失：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_u\mathcal{L}_{uni}
+
\lambda_a\mathcal{L}_{agr}^{NCE}
+
\lambda_d\mathcal{L}_{dis}^{NCE}
\]

建议使用 ramp-up：

\[
\lambda_a(t)=\lambda_a^{max}\cdot \min(1,t/T_r)
\]

\[
\lambda_d(t)=\lambda_d^{max}\cdot \min(1,t/T_r)
\]

防止早期 gate 不稳定。

---

## 8. 与已有方法的区别

### 8.1 与普通 InfoNCE / CLIP 式对齐

普通 InfoNCE：

\[
\text{same sample} \Rightarrow \text{positive}
\]

CoPA-v5：

\[
\text{same sample + reliable agreement} \Rightarrow \text{alignment positive}
\]

\[
\text{same sample + reliable disagreement} \Rightarrow \text{residual positive}
\]

所以 CoPA-v5 不是无条件跨模态对齐。

当前代码中的 `UncondInfoNCE` 是普通 pairwise same-sample baseline：

```text
text_anchor: T-A, T-V 双向 InfoNCE
full_pair:   T-A, T-V, A-V 双向 InfoNCE
```

它不同于 MMIM。MMIM 使用 fusion-to-modality CPC 和 BA/MI 估计，目标更接近“最大化融合表示与各模态之间的信息保持”；当前 baseline 只检验“同一样本跨模态 pair 是否应被无条件视为正样本”。

### 8.2 与 MMIM / Self-MM

MMIM 的训练目标可以概括为：

\[
\mathcal{L}_{task}+\alpha\mathcal{L}_{CPC}-\beta\mathcal{L}_{BA/MI}
\]

它说明“主任务 + 信息/对比辅助目标”在 MSA 中是合理范式。但 CoPA-v5 不直接复现 MMIM，因为本文关注的是 relation-state-dependent same-sample positive assumption，而不是泛化的 fusion-to-modality MI maximization。

Self-MM 采用多任务结构，同时训练 fusion 与单模态预测头，并动态生成单模态监督。它支持本文使用 reference model：

\[
CE_{fusion}+\eta_{uni}\cdot mean(CE_t,CE_a,CE_v)
\]

来获得单模态预测分布和 relation diagnostics。但 CoPA-v5 不使用 Self-MM 的动态伪标签作为主方法，避免把 label generation 与 relation-gated alignment 混在一起。

### 8.3 与 CS-Aligner / UniAlign

CS-Aligner 和 UniAlign 主要关注：

- InfoNCE 的 alignment-uniformity conflict；
- 分布级对齐；
- 跨模态表示空间 gap；
- 图文或多模态预训练场景。

CoPA-v5 关注：

- 监督式融合任务中的 same-sample relation state；
- 任务证据一致与不一致；
- RA 与 RD 的不同学习目标；
- 不把所有 paired modalities 当作 positive。

### 8.4 与 DecAlign

DecAlign 强调：

- modality-unique / modality-common decoupling；
- 异质特征原型 OT；
- 同质特征分布匹配。

CoPA-v5 也有 common / residual，但核心不同：

- DecAlign 是层次跨模态对齐；
- CoPA-v5 是 relation-gated selective alignment；
- DecAlign 没有显式区分 RA / RD / UA / ND；
- CoPA-v5 不把 reliable disagreement 强行对齐，而是学习 residual contrastive structure。

### 8.5 与 CaReFlow

CaReFlow 将模态 gap 视为 distribution mapping 问题，用 rectified flow 做分布映射。

CoPA-v5 不做全局模态分布映射，而是：

- 在样本级判断关系状态；
- 对 RA 做对齐；
- 对 RD 做残差学习；
- 不对所有样本做统一映射。

### 8.6 与 ARL / UDML

ARL / UDML 主要解决：

- 模态不平衡；
- 模态依赖偏置；
- 动态权重与不确定性；
- 弱模态被抑制。

CoPA-v5 主要解决：

- same-sample positive assumption 过强；
- 可靠一致与可靠不一致应该采用不同学习目标；
- 预测不一致不应直接等同于噪声。

---

## 9. 预期贡献

建议写成三个贡献：

### Contribution 1：Relation-state perspective

提出监督式多模态融合中的 relation-state 观点：

\[
\text{paired modalities are not unconditional alignment positives}
\]

并将同一样本内部关系分为：

- reliable agreement；
- uncertain agreement；
- reliable disagreement；
- noisy disagreement。

### Contribution 2：Relation-gated contrastive learning

提出 relation-gated contrastive learning：

- RA 使用 gated common InfoNCE；
- RD 使用 residual InfoNCE / residual prototype NCE；
- UA / ND 降低关系学习权重。

### Contribution 3：Empirical evidence for selective agreement and discriminative disagreement

通过实验说明：

- 无条件对齐和普通 pairwise same-sample InfoNCE 的收益依赖 relation state；
- TextInject / BalancedDirectAdd 只能作为 appendix，对主动机证明有限；
- High-D+High-R residual 作为补充诊断，用来判断是否能进一步主张 discriminative disagreement；
- CoPA-v5 在 overall 和 relation-state subgroup 上优于 baseline。

---

## 10. 风险与收缩方案

### 10.1 如果 RD residual probe 成功

论文主张可以保持：

> Selective Agreement and Discriminative Disagreement Learning

重点强调：

- reliable disagreement 不是噪声；
- residual contrastive learning 有贡献；
- RD-NCE 在 High-D+High-R 上有效。

### 10.2 如果 RD residual probe 不成功

论文应收缩为：

> Relation-aware selective alignment regularization

弱化：

- discriminative disagreement；
- complementary residual；
- RD 的强贡献。

保留：

- RA 选择性对齐；
- UA / ND 防污染；
- relation-gated InfoNCE 优于无条件 InfoNCE。

### 10.3 如果普通 InfoNCE 也稳定提升

不能说 InfoNCE 不好，只能说：

> Unconditional InfoNCE improves overall performance but is less stable in high-disagreement or reliable-disagreement groups compared with relation-gated InfoNCE.

### 10.4 如果 CoPA-v5 only improves overall

论文可以降级为：

> relation-gated contrastive regularization for supervised multimodal fusion

不要强行声明 RD 被稳定利用。

---

## 11. 推荐最终方法名

可选名称：

1. **CoPA-NCE**：Conditional Positive Alignment with Relation-Gated InfoNCE；
2. **CoPA-SAD**：Selective Agreement and Discriminative Disagreement；
3. **RG-CoPA**：Relation-Gated Conditional Positive Alignment。

推荐论文中使用：

> **CoPA: Relation-Gated Contrastive Learning for Selective Agreement and Discriminative Disagreement**

简称仍写 CoPA。

---

## 12. 最终一句话总结

> CoPA-v5 不再把预测不一致本身当作判别信息，而是把预测分布作为 relation gate；可靠一致样本通过 gated InfoNCE 学习公共语义对齐，可靠不一致样本通过 residual contrastive learning 学习判别性跨模态差异，不可靠样本只保留主任务监督，避免污染对齐和残差学习。
