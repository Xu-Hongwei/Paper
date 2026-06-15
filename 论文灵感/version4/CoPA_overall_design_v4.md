# CoPA version4：Selective Agreement and Discriminative Disagreement Learning

> 暂定题目：**CoPA: Selective Agreement and Discriminative Disagreement Learning for Supervised Multimodal Fusion**  
> 中文定位：**监督式多模态融合中的选择性一致学习与判别性不一致学习**  
> version4 目标：在 version3 的“关系条件正样本对齐”基础上进一步收紧论文主张：  
> **标签/预测一致样本不应全部采纳为对齐正样本；标签/预测不一致样本也不应全部视为噪声。CoPA 应选择可靠一致样本学习公共语义，并从可靠不一致样本中学习有判别价值的残差信息。**

---

## 0. version4 相比 version3 的核心修改

version3 已经明确了：

\[
\text{paired modalities} \neq \text{unconditional alignment-positive}
\]

并提出了基于 relation state 的 agreement / complementary / noisy 三类处理方式。version4 在此基础上做四个关键收紧：

| 修改点 | version3 | version4 |
|---|---|---|
| 论文中心 | 条件正样本对齐 | 选择性一致学习 + 判别性不一致学习 |
| 一致样本处理 | reliable agreement 用于对齐 | 一致样本也要筛选，低可靠/高置信错误不能更新公共原型 |
| 不一致样本处理 | reliable disagreement 保留残差 | 明确加入 residual discriminative learning，证明不一致样本能提供判别信息 |
| 跨类别边界 | 作为完整主线之一 | 降为 optional regularizer，避免主方法显得堆模块 |
| 原型设计 | agreement prototype + complementary prototype + distribution prototype 可并行 | 主方法优先采用 vector prototype，distribution prototype 作为扩展或诊断 |

version4 的核心不是“更复杂”，而是**更聚焦**：

\[
\text{same-sample relation}
\Rightarrow
\begin{cases}
\text{selectively align common semantics},\\
\text{discriminatively learn residual disagreement},\\
\text{suppress unreliable relation learning}.
\end{cases}
\]

---

## 1. 论文最终主张

### 1.1 要质疑的隐含假设

很多监督式多模态融合或对齐方法隐含使用：

\[
(x_t^n,x_v^n,x_a^n,y^n)
\Rightarrow
(z_t^n,z_v^n,z_a^n)\text{ should be aligned}
\]

也就是：

\[
\text{same sample} \Rightarrow \text{alignment-positive}
\]

更进一步，在普通监督对比学习中常见的设定是：

\[
\text{same label} \Rightarrow \text{positive}
\]

但在情感、讽刺、幽默等监督式多模态任务中，这两个假设都过强。

### 1.2 version4 的核心观点

CoPA-v4 的主张是：

\[
\text{same-sample / same-label} \neq \text{reliable alignment-positive}
\]

并进一步提出：

\[
\text{disagreement} \neq \text{noise}
\]

即：

> **一致样本需要选择性采纳；不一致样本需要区分可靠互补和不可靠噪声。可靠一致样本用于学习公共语义，可靠不一致样本用于学习判别性残差信息，不可靠样本只保留任务监督而降低关系学习权重。**

### 1.3 推荐摘要句

英文：

> Existing multimodal alignment methods often treat paired modalities as unconditional alignment positives. However, in supervised multimodal fusion, task-level agreement is not always reliable, and task-level disagreement is not always noise. CoPA estimates instance-level relation states, selectively aligns reliable agreement samples, and learns discriminative residual features from reliable disagreement samples while suppressing unreliable relations from contaminating prototype learning.

中文：

> 现有多模态对齐方法常将同一样本内的模态配对视为无条件对齐正样本。然而，在监督式多模态融合中，任务层面的一致不一定可靠，任务层面的不一致也不一定是噪声。CoPA 通过样本级关系估计，选择性对齐可靠一致样本，并从可靠不一致样本中学习判别性残差信息，同时抑制不可靠关系对原型学习的污染。

---

## 2. 术语修正：不要混淆“真实标签一致”和“预测证据一致”

在 MOSI/MOSEI 这类数据中，每个样本只有一个样本级真实标签：

\[
(x_t^n,x_v^n,x_a^n,y^n)
\]

因此严格来说：

\[
y_t^n=y_v^n=y_a^n=y^n
\]

这里的 \(y^n\) 是**多模态样本级标签**，不是每个模态独立标注得到的真实标签。

所以 version4 中的“标签一致 / 标签不一致”建议在论文里改成更准确的说法：

| 日常说法 | 论文中建议表述 | 含义 |
|---|---|---|
| 标签一致样本 | task-evidence agreement / prediction agreement | 不同模态对任务标签的预测分布相近 |
| 标签不一致样本 | task-evidence disagreement / prediction disagreement | 不同模态对任务标签的预测分布差异大 |
| 真实标签一致 | same sample-level label | 数据集中样本共享同一个监督标签 |
| 真实标签不一致样本 | different-label samples | 不同样本之间的标签不同，可作为类别边界学习 |

因此，全文最好避免写：

> 标签不一致的同一样本。

更稳写法是：

> **同一样本内部存在跨模态任务证据不一致。**

---

## 3. 样本关系状态重新定义

version4 将同一样本内的关系状态划分为四类：

| 状态 | 预测分歧 | 可靠性 | 是否适合对齐 | 是否有判别价值 | CoPA 处理 |
|---|---:|---:|---|---|---|
| Reliable Agreement, RA | 低 | 高 | 是 | 公共语义 | 选择性公共对齐 |
| Uncertain Agreement, UA | 低 | 低 | 否 | 不稳定 | 降低关系学习权重 |
| Reliable Disagreement, RD | 高 | 高 | 否 | 可能有 | 判别性残差学习 |
| Noisy Disagreement, ND | 高 | 低 | 否 | 不稳定 | 抑制原型更新 |

核心变化：

\[
RA \Rightarrow \text{alignment-positive}
\]

\[
RD \Rightarrow \text{discriminative-residual-positive}
\]

\[
UA,ND \Rightarrow \text{relation down-weighted}
\]

不同真实标签样本：

\[
y^n\neq y^k
\]

不属于同一样本关系状态，它们只作为**类别边界信息**，可以用于 optional inter-class regularization，而不作为 version4 主贡献。

---

## 4. 方法总体命名

建议主方法命名为：

\[
\textbf{CoPA-SAD}
\]

其中：

- **S**elective **A**greement；
- **D**iscriminative **D**isagreement。

完整名称：

> **CoPA-SAD: Conditional Positive Alignment via Selective Agreement and Discriminative Disagreement Learning**

如果觉得名字过长，论文中可仍称 CoPA，方法标题中写：

> **Selective Agreement and Discriminative Disagreement Learning**

---

## 5. 方法框架

CoPA-v4 由四个核心模块组成：

| 模块 | 名称 | 核心作用 |
|---|---|---|
| Module 1 | Common-Residual Representation Decomposition | 将每个模态拆成公共语义和残差信息 |
| Module 2 | Instance-level Relation Estimation | 估计可靠一致、可靠不一致和不可靠关系 |
| Module 3 | Relation-Aware Dual Prototype Learning | 用可靠一致更新公共原型，用可靠不一致更新残差原型 |
| Module 4 | Conditional Objective Optimization | 对齐公共语义、学习判别残差、抑制不可靠关系 |

---

# Module 1：公共-残差表示分解

给定三模态输入：

\[
x_t^n,\quad x_v^n,\quad x_a^n
\]

每个模态通过编码器得到：

\[
h_m^n=E_m(x_m^n),\quad m\in\{t,v,a\}
\]

再通过两个投影头得到：

\[
z_m^{c,n}=P_m^c(h_m^n)
\]

\[
z_m^{r,n}=P_m^r(h_m^n)
\]

其中：

| 表示 | 作用 |
|---|---|
| \(z_m^c\) | 用于可靠一致样本的公共语义对齐 |
| \(z_m^r\) | 用于可靠不一致样本的残差信息学习 |

为了稳定关系估计，每个模态接单模态预测头：

\[
p_m^n=\text{Softmax}(C_m(z_m^{c,n}))
\]

---

# Module 2：样本级关系估计

## 2.1 预测分歧

对任意两个模态 \(i,j\)：

\[
D_{ij}^n=JSD(p_i^n,p_j^n)
\]

转化为 agreement score：

\[
A_{ij}^n=\exp(-D_{ij}^n/\tau_A)
\]

其中：

- \(A_{ij}^n\) 越大，表示两个模态任务证据越一致；
- \(A_{ij}^n\) 越小，表示两个模态任务证据越不一致。

## 2.2 可靠性

version4 明确区分两个可靠性：

### 诊断可靠性

用于 motivation analysis，不能使用 test label：

\[
R_m^{diag,n}
=
1-\frac{H(p_m^n)}{\log K}
\]

它只表示预测分布是否尖锐。

### 训练可靠性

用于训练集 prototype 更新，可使用训练标签：

\[
Q_m^n
=
1-\frac{H(p_m^n)}{\log K}
\]

\[
S_m^n=p_m^n(y^n)
\]

\[
R_m^{label,n}=Q_m^nS_m^n
\]

它同时要求：

1. 预测置信度高；
2. 预测分布支持真实标签。

这样可以避免：

> 三个模态都高置信预测错误，却被误当作 reliable agreement 更新公共原型。

## 2.3 关系权重

训练时默认显式区分证据可靠性 \(Q_m\)、标签支持度 \(S_m\) 和模态间一致性 \(A_{ij}\)。可靠一致要求两个模态都证据明确、都支持真实标签，并且预测分布接近：

\[
g_{ij}^{agr,n}=Q_i^nQ_j^nS_i^nS_j^nA_{ij}^n
\]

可靠不一致允许某个模态给出与真实标签方向相反但高置信的反差证据，因此不使用 \(S_iS_j\) 同时压低两个模态，而使用标签锚点：

\[
B_{ij}^{label,n}=\max(S_i^n,S_j^n)
\]

可靠不一致权重：

\[
g_{ij}^{dis,n}=Q_i^nQ_j^nB_{ij}^{label,n}(1-A_{ij}^n)
\]

不可靠关系权重：

\[
g_{ij}^{unr,n}=1-Q_i^nQ_j^n
\]

这样做的动机是：如果文本高置信表达正向、音频和视觉高置信支持负向，文本虽然不支持真实负向标签，但它的反差本身可能是讽刺、反讽或情绪冲突中的判别性残差信息。

训练时所有关系权重都建议使用 stop-gradient：

\[
\text{sg}(g_{ij}^{agr}),\quad \text{sg}(g_{ij}^{dis}),\quad \text{sg}(g_{ij}^{unr})
\]

避免模型通过操纵关系权重逃避损失。

---

# Module 3：关系感知双原型学习

version4 建议主方法优先使用**向量原型**，而不是一开始就使用 Gaussian distribution prototype。原因是：

1. 向量原型更稳定；
2. 更容易消融；
3. 不容易被审稿人认为模块过多；
4. distribution prototype 可以作为扩展实验。

对每个模态 \(m\)、类别 \(c\)，维护两个 EMA 原型：

\[
P_{m,c}^{agr}
\]

\[
P_{m,c}^{dis}
\]

其中：

| 原型 | 使用特征 | 样本来源 | 作用 |
|---|---|---|---|
| \(P_{m,c}^{agr}\) | \(z_m^c\) | Reliable Agreement | 公共语义中心 |
| \(P_{m,c}^{dis}\) | \(z_m^r\) | Reliable Disagreement | 判别残差中心 |

## 3.1 Agreement prototype 更新

对模态 \(m\)，定义：

\[
w_{m,n}^{agr}
=
\frac{1}{M-1}
\sum_{j\neq m}
g_{mj}^{agr,n}
\]

对类别 \(c\)：

\[
\hat P_{m,c}^{agr}
=
\frac{
\sum_{n:y^n=c}w_{m,n}^{agr}z_m^{c,n}
}{
\sum_{n:y^n=c}w_{m,n}^{agr}+\epsilon
}
\]

EMA 更新：

\[
P_{m,c}^{agr}
\leftarrow
\rho P_{m,c}^{agr}
+
(1-\rho)\hat P_{m,c}^{agr}
\]

## 3.2 Disagreement prototype 更新

对模态 \(m\)：

\[
w_{m,n}^{dis}
=
\frac{1}{M-1}
\sum_{j\neq m}
g_{mj}^{dis,n}
\]

对类别 \(c\)：

\[
\hat P_{m,c}^{dis}
=
\frac{
\sum_{n:y^n=c}w_{m,n}^{dis}z_m^{r,n}
}{
\sum_{n:y^n=c}w_{m,n}^{dis}+\epsilon
}
\]

EMA 更新：

\[
P_{m,c}^{dis}
\leftarrow
\rho P_{m,c}^{dis}
+
(1-\rho)\hat P_{m,c}^{dis}
\]

注意：

\[
P_{t,c}^{dis},P_{v,c}^{dis},P_{a,c}^{dis}
\]

是**模态专属残差原型**，不要跨模态强制拉近。原因是 residual 表示的正是模态特有证据。

---

# Module 4：条件优化目标

## 4.1 主任务损失

融合表示可以先采用简单 concat：

\[
z_{fuse}^n
=
[z_t^{c,n};z_v^{c,n};z_a^{c,n};z_t^{r,n};z_v^{r,n};z_a^{r,n}]
\]

\[
\hat y^n=F(z_{fuse}^n)
\]

分类任务：

\[
\mathcal{L}_{task}=CE(\hat y^n,y^n)
\]

回归任务可以使用 MAE / MSE / CCC loss，按数据集标准设置。

## 4.2 Selective Agreement Loss

可靠一致样本的公共语义靠近 agreement prototype：

\[
\mathcal{L}_{agr}
=
\sum_n
\sum_{i<j}
g_{ij}^{agr,n}
\left[
d(z_i^{c,n},P_{j,y^n}^{agr})
+
d(z_j^{c,n},P_{i,y^n}^{agr})
\right]
\]

其中 \(d(\cdot,\cdot)\) 可用：

\[
d(u,v)=1-\cos(u,v)
\]

或者欧氏距离。第一版建议用 cosine distance。

直观含义：

> 只有两个模态既可靠又预测一致时，才把它们作为公共语义对齐正样本。

## 4.3 Discriminative Disagreement Prototype Loss

可靠不一致样本的残差信息不做跨模态拉近，而是进行**模态内类别判别学习**：

\[
\mathcal{L}_{dis}^{proto}
=
-\sum_n
\sum_m
w_{m,n}^{dis}
\log
\frac{
\exp(\text{sim}(z_m^{r,n},P_{m,y^n}^{dis})/\tau)
}{
\sum_{c'}
\exp(\text{sim}(z_m^{r,n},P_{m,c'}^{dis})/\tau)
}
\]

这部分是 version4 相比 version3 最重要的新增点。

它回答的问题是：

> 可靠不一致样本中的 residual 是否能形成类别相关结构？

如果该损失有效，说明 reliable disagreement 不只是“避免错误对齐”，而是可以提供任务判别信息。

## 4.4 Residual Classification Loss

为了更直接地验证和利用可靠不一致样本，可加入轻量 residual classifier：

\[
p_{res}^n=C_{res}([z_t^{r,n};z_v^{r,n};z_a^{r,n}])
\]

样本级可靠不一致权重：

\[
g_n^{dis}
=
\frac{1}{3}
\sum_{i<j}g_{ij}^{dis,n}
\]

残差分类损失：

\[
\mathcal{L}_{res}
=
\sum_n
g_n^{dis}\cdot CE(p_{res}^n,y^n)
\]

它的作用是：

> 直接鼓励 reliable disagreement 的 residual branch 学到对任务有用的信息。

## 4.5 Bounded Residual Separation Loss

可靠不一致样本不应被强行拉近，但也不能完全散开。因此采用 bounded separation：

\[
d_{ij}^{r,n}=d(z_i^{r,n},z_j^{r,n})
\]

\[
\mathcal{L}_{sep}
=
\sum_n
\sum_{i<j}
g_{ij}^{dis,n}
\left[
\max(0,\delta_{min}-d_{ij}^{r,n})
+
\gamma\max(0,d_{ij}^{r,n}-\delta_{max})
\right]
\]

含义：

- 如果 residual 太近，说明不一致信息被抹掉；
- 如果 residual 太远，说明表示可能变成无结构噪声；
- 因此只要求残差保持在合理区间。

## 4.6 单模态辅助损失

为稳定 \(p_m\)、\(D_{ij}\)、\(R_m\)，保留单模态辅助头：

\[
\mathcal{L}_{uni}
=
\sum_n
\sum_m
CE(p_m^n,y^n)
\]

如果担心低可靠模态主导训练，可使用软权重：

\[
\mathcal{L}_{uni}
=
\sum_n
\sum_m
(0.5+0.5R_m^n)CE(p_m^n,y^n)
\]

## 4.7 version4 主损失

推荐主方法损失为：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{dis}^{proto}
+
\lambda_3\mathcal{L}_{res}
+
\lambda_4\mathcal{L}_{sep}
+
\lambda_5\mathcal{L}_{uni}
\]

第一版实现可以使用更轻量的 CoPA-v4-min：

\[
\mathcal{L}_{min}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{dis}^{proto}
+
\lambda_3\mathcal{L}_{res}
+
\lambda_5\mathcal{L}_{uni}
\]

如果 \(\mathcal{L}_{sep}\) 不稳定，可以先不放入主方法。

---

## 6. Optional：跨类别异质边界学习的位置

version3 中将 different-label relation 放入主线。version4 建议将其降级为 optional regularizer：

\[
\mathcal{L}_{inter}
\]

原因：

1. 主贡献已经足够：selective agreement + discriminative disagreement；
2. 跨类别边界容易让方法显得过复杂；
3. 对 MOSI/MOSEI 的连续标签，负样本定义和边界强度需要额外处理；
4. 如果主结果站稳，再把它作为增强模块更稳。

推荐论文中处理方式：

| 模块 | version4 定位 |
|---|---|
| Selective Agreement | 主贡献 |
| Discriminative Disagreement | 主贡献 |
| Unreliable Relation Suppression | 边界条件 |
| Inter-class Heterogeneous Boundary | 附录或增强实验 |
| Distribution Prototype | 附录或增强实验 |
| Relation-conditioned Fusion | 后续扩展 |

---

## 7. 训练流程

### Stage 1：Warm-up

训练：

\[
\mathcal{L}_{task}+\eta\mathcal{L}_{uni}
\]

目的：

- 让单模态预测头可用；
- 让 \(D_{ij}\) 和 \(R_m\) 不完全随机；
- 得到初始 common/residual 表示。

### Stage 2：Relation Calibration and Prototype Initialization

使用训练集特征和 \(R^{label}\) 初始化：

\[
P^{agr},\quad P^{dis}
\]

初始化时建议：

- 只使用训练集；
- 不使用 test label；
- relation weights 使用 detach；
- 低样本类别可退化为普通类别原型。

### Stage 3：Joint Training

每个 batch：

1. 提取 \(h_m\)；
2. 计算 \(z_m^c,z_m^r\)；
3. 计算单模态预测 \(p_m\)；
4. 计算 \(D_{ij},A_{ij},R_m,g^{agr},g^{dis}\)；
5. detach relation weights；
6. EMA 更新 \(P^{agr},P^{dis}\)；
7. 优化：

\[
\mathcal{L}_{min}
\quad \text{or} \quad
\mathcal{L}
\]

### Stage 4：Evaluation

测试阶段：

- 不使用真实标签计算 \(R^{label}\)；
- 关系分组诊断使用 \(R^{diag}\)；
- 最终预测只使用模型输出；
- case study 可以在预测后展示真实标签和原始样本内容。

---

## 8. 超参数建议

第一轮建议不要搜索太多：

| 参数 | 建议初值 | 搜索范围 |
|---|---:|---|
| \(\lambda_1\) | 0.05 | 0.01, 0.05, 0.1 |
| \(\lambda_2\) | 0.05 | 0.01, 0.05, 0.1 |
| \(\lambda_3\) | 0.05 | 0.01, 0.05, 0.1 |
| \(\lambda_4\) | 0.01 | 0, 0.01, 0.05 |
| \(\lambda_5\) | 0.1 | 0.05, 0.1 |
| \(\tau_A\) | 0.5 | 0.1, 0.5, 1.0 |
| \(\tau\) | 0.1 | 0.05, 0.1, 0.2 |
| \(\rho\) | 0.95 | 0.9, 0.95, 0.99 |
| \(\delta_{min}\) | 0.2 | 0.1, 0.2, 0.5 |
| \(\delta_{max}\) | 1.0 | 0.8, 1.0, 1.2 |

第一轮只搜索：

\[
\lambda_1,\lambda_2,\lambda_3
\]

其他固定，避免调参空间过大。

---

## 9. 实验验证对应关系

| 论文主张 | 需要的实验 |
|---|---|
| Concat 能学习标签边界，但 relation state 仍有诊断价值 | Experiment 1：concat-aware motivation table，比较 Concat / UncondAlign / DirectAdd / SoftSplit Probe |
| 无条件对齐收益依赖关系状态 | Experiment 1：Low/Mid/High-D delta 与 \(\lambda\) 敏感性 |
| 直接相加不是充分的“无损对齐” | DirectAdd baseline：\(\alpha\) sweep、DirectAdd vs SoftSplit Probe、RD 分组表现 |
| 一致样本不应全部采纳 | Selective agreement diagnostic：RA vs UA / high label-support vs low label-support |
| 不一致样本不一定是噪声 | D×R 分层：RD 与 ND 的差异 |
| reliable disagreement 有结构 | residual distribution diagnostic |
| reliable disagreement 有判别价值 | residual-only / common+residual probe，\(\mathcal{L}_{dis}^{proto}\) 消融 |
| CoPA 方法有效 | 主结果 + relation group analysis |
| 不是随机分组造成 | random grouping / shuffled residual / shuffled relation weight |

---

## 10. 与已有方法的区别

### 10.1 与 DecAlign

DecAlign 的重点是：

> 解耦 modality-common 和 modality-unique，再分别设计层次对齐。

CoPA-v4 的重点是：

> 判断同一样本内哪些关系应被对齐，哪些关系应保留为判别性残差。

| DecAlign | CoPA-v4 |
|---|---|
| feature type first | relation state first |
| 主要解决 common/unique entanglement | 主要解决 unconditional positive assumption |
| unique 也会被对齐机制约束 | reliable disagreement residual 不强制拉近 |
| 层次对齐 | 选择性一致 + 判别性不一致 |

### 10.2 与普通 SupCon

普通 SupCon：

\[
\text{same label}\Rightarrow positive
\]

CoPA-v4：

\[
\text{same label + reliable agreement}\Rightarrow alignment-positive
\]

\[
\text{same label + reliable disagreement}\Rightarrow residual-discriminative-positive
\]

\[
\text{same label + unreliable}\Rightarrow relation-down-weighted
\]

### 10.3 与 UDML / uncertainty fusion

UDML 类方法主要处理：

> 哪个模态更可靠，融合时给多少权重。

CoPA-v4 处理的是：

> 同一样本内的跨模态关系应如何被学习：对齐、保留，还是降权。

因此 reliability 在 CoPA 中不是主角，而是用来区分：

- reliable disagreement；
- noisy disagreement。

### 10.4 与 CS-Aligner / distribution alignment

分布对齐方法强调降低模态分布差异。CoPA-v4 则强调：

> 某些差异不应被消除，因为它们可能是任务判别信息。

---

## 11. 论文结构建议

### Introduction

推荐逻辑：

1. 多模态融合依赖跨模态共享语义；
2. 对齐常被用于增强共享语义；
3. 但现有方法常把 paired modalities 当作 unconditional positives；
4. 监督式多模态任务中，一致样本可能不可靠，不一致样本可能有用；
5. 但实验1不声称 Concat 无法学习标签边界，而是进一步说明普通隐式融合和直接相加不能显式区分关系状态；
6. 诊断实验显示无条件对齐收益依赖 relation state，DirectAdd 并不能稳定替代关系条件拆分，且 reliable disagreement 可能具有结构化残差；
7. 提出 CoPA：保留原始监督路径，同时进行 selective agreement + discriminative disagreement。

### Method

按问题展开：

> Given a paired multimodal sample, should its modalities be aligned, learned as discriminative residuals, or down-weighted as unreliable?

对应：

1. relation estimation；
2. agreement prototype；
3. disagreement prototype；
4. conditional losses。

### Experiments

按证据链展开：

1. Motivation analysis；
2. Main comparison；
3. Relation group analysis；
4. Residual discriminative probe；
5. Ablation；
6. Case study；
7. Robustness / appendix。

---

## 12. 失败情况与叙事收缩

| 失败情况 | 说明 | 收缩方式 |
|---|---|---|
| \(D_{pred}\) 分组无趋势 | 预测分歧不是有效关系信号 | 改用 relation-aware regularization 或更换数据集 |
| RA 与 UA 差异不明显 | 一致样本筛选证据不足 | 弱化“consistent samples not all reliable” |
| RD residual-only 无贡献 | 不一致样本未证明有判别信息 | 将 \(g^{dis}\) 改写为避免错误对齐 |
| \(P^{dis}\) 无贡献 | 残差原型不稳定 | 保留 residual classifier，去掉 comp prototype |
| CoPA 只提升 Overall | relation group 证据不足 | 论文主张收缩为 prototype regularization |
| \(\mathcal{L}_{inter}\) 无贡献 | 跨类别边界不是核心 | 放入 appendix 或删除 |

---

## 13. version4 一句话总结

> **CoPA-v4 不再只问“是否对齐”，而是问“这个样本关系应该如何被利用”：可靠一致用于公共语义对齐，可靠不一致用于判别性残差学习，不可靠关系则避免污染原型。**

公式化表达：

\[
RA \Rightarrow \text{align common semantics}
\]

\[
RD \Rightarrow \text{learn discriminative residuals}
\]

\[
UA,ND \Rightarrow \text{down-weight relation learning}
\]

\[
\text{different-label samples} \Rightarrow \text{optional boundary regularization}
\]
