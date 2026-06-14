# CoPA 论文整体设计与实现细节

> 论文暂定方向：**监督式多模态融合中的条件正样本对齐**  
> 方法暂定名：**CoPA: Conditional Positive Alignment via Relation-Aware Prototypes for Multimodal Fusion**  
> 中文暂定题目：**CoPA：面向多模态融合的关系感知原型条件正样本对齐方法**

---

## 1. 论文定位

### 1.1 不做“大一统多模态对齐”

当前论文不建议定位为：

> 面向所有多模态对齐任务的通用方法。

原因是图文检索、视觉语言零样本分类、大规模多模态预训练、视频-文本检索等任务范式差异较大，通常没有明确的类别标签和单模态分类分布，无法直接使用本文的类别关系原型。

更合理的定位是：

> **面向监督式多模态融合任务的条件正样本对齐框架。**

也就是说，本文主要面向如下形式的数据：

\[
\{x_t^n, x_v^n, x_a^n, y^n\}_{n=1}^{N}
\]

其中：

- \(x_t^n\)：文本模态；
- \(x_v^n\)：视觉模态；
- \(x_a^n\)：音频模态；
- \(y^n\)：监督标签，可以是分类标签，也可以是离散化后的情感标签。

典型任务包括：

- 多模态情感分析；
- 多模态情绪识别；
- 多模态讽刺检测；
- 多模态幽默检测；
- 音视频事件分类或情绪分类的补充验证。

---

## 2. 核心问题

### 2.1 传统多模态对齐的隐含假设

许多多模态对齐方法默认：

\[
(x_t^n, x_v^n, x_a^n)
\]

来自同一个样本，因此它们天然是跨模态正样本，应被拉近或映射到一致的共享语义空间。

这个假设可以写成：

\[
\text{paired multimodal samples} \Rightarrow \text{alignment-positive samples}
\]

即：

> 同一样本内的不同模态配对，一定是可靠且可对齐的正样本。

---

### 2.2 本文反思的核心假设

本文认为，在监督式多模态融合任务中，上述假设并不总成立。

同一样本内的跨模态关系可能至少有三类：

| 关系类型 | 含义 | 示例 | 应如何处理 |
|---|---|---|---|
| **Agreement-positive** | 可靠且一致 | 文本、语音、表情都表达负面 | 应对齐 |
| **Complementary-positive** | 可靠但不一致 | 文本表面积极，语音低沉，表情疲惫 | 不应强制对齐，应保留差异 |
| **Noisy-positive** | 至少一个模态不可靠 | 音频噪声、视觉遮挡、文本缺失 | 应降低影响 |

因此，本文的核心命题是：

> **Paired multimodal samples are not always alignment-positive.**

中文表述：

> 同一样本内的多模态配对不一定都是可对齐正样本。

---

## 3. 从八篇参考论文中吸收的设计原则

本文不直接拼接已有论文模块，而是从八篇论文中提炼出四条设计原则。

### 3.1 八篇论文的启发与 CoPA 的转化

| 参考论文 | 主要精华 | CoPA 中的吸收方式 |
|---|---|---|
| **DecAlign** | 对齐前要区分共享语义和模态特有信息；直接混合或直接对齐会导致语义干扰；原型级结构对齐比简单点对点更稳定。 | 设计 \(z^c\) 与 \(z^r\)，但由关系状态决定其训练方式；使用关系感知原型而不是样本点硬对齐。 |
| **UDML** | 模态质量不是静态的；不可靠模态不应被同等使用；弱模态可能被双重抑制。 | 使用可靠性 \(R_m\) 判断样本是否参与 agreement / complementary / noisy 分支，不只在融合阶段使用。 |
| **ARL** | 多模态学习不应追求简单平衡；模态贡献应与其统计性质和任务需求相关。 | CoPA 不要求所有模态同等对齐，而是根据关系状态决定对齐、保留或抑制。 |
| **ProtoMM** | 类别原型应融合多模态证据，并可作为动态分布/语义锚点，而不是固定单点。 | 构建 \(P^{agr}\) 与 \(P^{comp}\) 双原型库，使原型承载不同跨模态关系。 |
| **UniAlign / Uniformity-Alignment** | 多模态正样本之间可能存在非共线冲突；强行多路正样本对齐会引入 intra-alignment conflict。 | 将“paired 是否一定可对齐”作为核心问题，避免 reliable disagreement 被强行拉近。 |
| **CS-Aligner** | 单纯 pairwise MI / InfoNCE 不足以保证分布级对齐；需要引入分布结构。 | 从样本级 pair alignment 转为类别-关系原型级结构对齐。 |
| **MASK** | 原型空间需要保持语义结构；prototype consistency contrastive loss 可降低方差影响并提升判别性。 | 用 prototype contrastive loss 训练 agreement / complementary prototypes，而不只用 L2 拉近。 |
| **CaReFlow** | 一对一映射视野太窄，应利用目标分布；同时对齐不应损失源模态信息。 | 不做复杂 flow，但采用原型级对齐和 residual preservation，避免一对一硬对齐和信息抹除。 |

### 3.2 抽象出的四条设计原则

从上述八篇论文中，本文抽象出以下原则：

1. **对齐对象要选择**  
   不是所有配对模态都应强制对齐。

2. **对齐尺度要提升**  
   样本点对点对齐不稳定，应引入类别/关系原型作为稳定语义锚点。

3. **模态状态要动态判断**  
   模态可靠性会随样本和噪声变化，不能默认所有模态都同等可靠。

4. **模态差异不能简单消除**  
   可靠但不一致的信息可能是判别性互补信息，应结构化保留。

---

## 4. 论文创新点

### 创新点 1：提出 Conditional Positive Alignment 问题

本文指出，监督式多模态融合中，同一样本内的不同模态不一定都是可靠的可对齐正样本。

现有方法通常默认：

\[
\text{paired} \Rightarrow \text{positive}
\]

本文进一步区分：

\[
\text{paired} \Rightarrow 
\begin{cases}
\text{agreement-positive}\\
\text{complementary-positive}\\
\text{noisy-positive}
\end{cases}
\]

这使多模态对齐从“无条件正样本对齐”变为“条件正样本对齐”。

---

### 创新点 2：提出 Cross-modal Relation State Estimation

本文通过单模态预测分布同时估计：

- 模态可靠性 \(R_m\)；
- 跨模态语义一致性 \(A_{ij}\)；
- 三类关系权重 \(g^{agr}, g^{comp}, g^{noise}\)。

该模块用于判断一个配对关系应进入：

- 对齐分支；
- 互补保留分支；
- 噪声抑制分支。

---

### 创新点 3：提出 Relation-Aware Prototype Bank

不同于普通类别原型：

\[
P_{m,c}
\]

本文为每个模态 \(m\)、每个类别 \(c\) 构造两个关系原型：

\[
P_{m,c}^{agr}, \quad P_{m,c}^{comp}
\]

分别表示：

- 可跨模态对齐的公共语义中心；
- 应保留的互补差异中心。

---

### 创新点 4：提出 Conditional Prototype Objective

本文对三类关系采用不同优化方式：

| 关系 | 优化方式 |
|---|---|
| Agreement-positive | 进行原型对齐 |
| Complementary-positive | 进行残差信息结构化保留 |
| Noisy-positive | 降低其原型更新和融合影响 |

这样可以避免错误对齐，同时保留跨模态差异中的判别信息。

---

## 5. 方法总览

### 5.1 输入与符号

给定三模态输入：

\[
X^n=\{x_t^n,x_v^n,x_a^n\}
\]

标签：

\[
y^n
\]

模态集合：

\[
m\in\{t,v,a\}
\]

其中：

- \(t\)：text；
- \(v\)：visual；
- \(a\)：audio。

---

### 5.2 整体流程

\[
x_m
\rightarrow h_m
\rightarrow (z_m^c,z_m^r)
\rightarrow (R_m,A_{ij},g_{ij})
\rightarrow (P_{m,c}^{agr},P_{m,c}^{comp})
\rightarrow \mathcal{L}_{agr},\mathcal{L}_{comp}
\rightarrow z_f
\rightarrow \hat{y}
\]

模块包括：

1. 公共-残差表示分离；
2. 跨模态关系状态估计；
3. 关系感知原型库；
4. 条件原型对齐与互补保留；
5. 关系条件融合与预测。

---

## 6. 模块一：公共-残差表示分离

### 6.1 单模态编码

每个模态通过编码器得到特征：

\[
h_m^n=E_m(x_m^n)
\]

其中：

\[
m\in\{t,v,a\}
\]

可以使用：

| 模态 | 可选输入特征 |
|---|---|
| Text | BERT / RoBERTa / GloVe |
| Visual | OpenFace / ResNet / ViT features |
| Audio | COVAREP / wav2vec / acoustic features |

---

### 6.2 公共语义和残差信息

对每个模态特征使用两个投影头：

\[
z_m^{c,n}=P_m^c(h_m^n)
\]

\[
z_m^{r,n}=P_m^r(h_m^n)
\]

其中：

- \(z_m^{c,n}\)：候选公共语义，用于 agreement-positive；
- \(z_m^{r,n}\)：候选残差信息，用于 complementary-positive。

进行 L2 归一化：

\[
z_m^{c,n}\leftarrow \frac{z_m^{c,n}}{\|z_m^{c,n}\|_2}
\]

\[
z_m^{r,n}\leftarrow \frac{z_m^{r,n}}{\|z_m^{r,n}\|_2}
\]

---

### 6.3 可选正交约束

为了减少公共语义和残差信息混杂，可以加入轻量正交约束：

\[
\mathcal{L}_{orth}
=
\sum_{n,m}
|\cos(z_m^{c,n},z_m^{r,n})|
\]

该项不是核心创新，只是辅助项。第一版实现时可以暂时不加入。

---

## 7. 模块二：跨模态关系状态估计

### 7.1 单模态预测分布

每个模态接一个单模态预测头：

\[
p_m^n=\text{Softmax}(C_m(z_m^{c,n}))
\]

其中：

\[
p_m^n\in \mathbb{R}^{K}
\]

表示第 \(m\) 个模态对第 \(n\) 个样本的类别预测分布。

---

### 7.2 模态可靠性

使用归一化熵估计可靠性：

\[
H(p_m^n)=-\sum_{k=1}^{K}p_{m,k}^n\log p_{m,k}^n
\]

\[
R_m^n=1-\frac{H(p_m^n)}{\log K}
\]

其中：

\[
R_m^n\in[0,1]
\]

解释：

- \(R_m^n\) 高：该模态预测确定，可靠性高；
- \(R_m^n\) 低：该模态预测混乱，可能是噪声或弱信息。

---

### 7.3 跨模态语义一致性

对任意模态 \(i,j\)，计算 Jensen-Shannon Divergence：

\[
q_{ij}^n=\frac{1}{2}(p_i^n+p_j^n)
\]

\[
\text{JSD}(p_i^n,p_j^n)
=
\frac{1}{2}\text{KL}(p_i^n\|q_{ij}^n)
+
\frac{1}{2}\text{KL}(p_j^n\|q_{ij}^n)
\]

定义一致性：

\[
A_{ij}^n=
\exp\left(
-\frac{\text{JSD}(p_i^n,p_j^n)}{\tau_A}
\right)
\]

其中：

- \(A_{ij}^n\in[0,1]\)；
- \(\tau_A\) 是温度系数。

---

### 7.4 三类关系权重

#### Agreement-positive

\[
g_{ij}^{agr,n}=R_i^nR_j^nA_{ij}^n
\]

可靠且一致，应该对齐。

#### Complementary-positive

\[
g_{ij}^{comp,n}=R_i^nR_j^n(1-A_{ij}^n)
\]

可靠但不一致，应该保留差异。

#### Noisy-positive

\[
g_{ij}^{noise,n}=1-R_i^nR_j^n
\]

至少一个模态不可靠，应降低影响。

---

### 7.5 模态级关系权重

对模态 \(m\)，聚合它和其他模态的关系：

\[
g_m^{agr,n}
=
\frac{1}{M-1}
\sum_{j\neq m}g_{mj}^{agr,n}
\]

\[
g_m^{comp,n}
=
\frac{1}{M-1}
\sum_{j\neq m}g_{mj}^{comp,n}
\]

其中：

\[
M=3
\]

实际实现中，建议：

\[
g_m^{agr,n},g_m^{comp,n}
\]

使用 `detach()`，避免模型通过操纵关系权重逃避损失。

---

## 8. 模块三：关系感知原型库

### 8.1 原型定义

对每个模态 \(m\)、每个类别 \(c\)，维护：

\[
P_{m,c}^{agr}
\]

\[
P_{m,c}^{comp}
\]

其中：

| 原型 | 使用特征 | 样本来源 | 语义 |
|---|---|---|---|
| \(P_{m,c}^{agr}\) | \(z_m^c\) | reliable agreement 样本 | 可对齐公共语义中心 |
| \(P_{m,c}^{comp}\) | \(z_m^r\) | reliable disagreement 样本 | 互补差异中心 |

---

### 8.2 Agreement prototype 更新

对 batch 中类别为 \(c\) 的样本：

\[
\bar{P}_{m,c}^{agr}
=
\frac{
\sum_{n:y_n=c}
g_m^{agr,n}z_m^{c,n}
}{
\sum_{n:y_n=c}g_m^{agr,n}+\epsilon
}
\]

EMA 更新：

\[
P_{m,c}^{agr}
\leftarrow
\text{Norm}
\left(
\mu P_{m,c}^{agr}
+
(1-\mu)\bar{P}_{m,c}^{agr}
\right)
\]

---

### 8.3 Complementary prototype 更新

\[
\bar{P}_{m,c}^{comp}
=
\frac{
\sum_{n:y_n=c}
g_m^{comp,n}z_m^{r,n}
}{
\sum_{n:y_n=c}g_m^{comp,n}+\epsilon
}
\]

EMA 更新：

\[
P_{m,c}^{comp}
\leftarrow
\text{Norm}
\left(
\mu P_{m,c}^{comp}
+
(1-\mu)\bar{P}_{m,c}^{comp}
\right)
\]

其中：

- \(\mu\)：EMA 动量，一般取 \(0.9\sim0.99\)；
- \(\epsilon\)：防止除零；
- \(\text{Norm}\)：L2 归一化。

如果某个类别在当前 batch 中关系权重过低，则跳过该原型更新。

---

## 9. 模块四：条件原型对齐与互补保留

### 9.1 Agreement branch：可靠一致关系对齐

#### 9.1.1 样本到 agreement prototype

使用 prototype contrastive loss：

\[
\mathcal{L}_{agr}^{proto}
=
-
\frac{
\sum_{n,m}
g_m^{agr,n}
\log
\frac{
\exp(\text{sim}(z_m^{c,n},P_{m,y_n}^{agr})/\tau)
}{
\sum_{c=1}^{K}
\exp(\text{sim}(z_m^{c,n},P_{m,c}^{agr})/\tau)
}
}{
\sum_{n,m}g_m^{agr,n}+\epsilon
}
\]

含义：

> reliable agreement 样本的公共特征应靠近正确类别的 agreement prototype，并远离其他类别原型。

---

#### 9.1.2 跨模态 agreement prototype 对齐

\[
\mathcal{L}_{agr}^{cross}
=
\sum_{c=1}^{K}
\sum_{i<j}
\bar{g}_{ij,c}^{agr}
D(P_{i,c}^{agr},P_{j,c}^{agr})
\]

其中：

\[
\bar{g}_{ij,c}^{agr}
=
\frac{1}{N_c}
\sum_{n:y_n=c}g_{ij}^{agr,n}
\]

\[
D(a,b)=1-\cos(a,b)
\]

最终：

\[
\mathcal{L}_{agr}
=
\mathcal{L}_{agr}^{proto}
+
\alpha\mathcal{L}_{agr}^{cross}
\]

---

### 9.2 Complementary branch：可靠不一致关系保留

#### 9.2.1 残差到 complementary prototype

\[
\mathcal{L}_{comp}^{proto}
=
-
\frac{
\sum_{n,m}
g_m^{comp,n}
\log
\frac{
\exp(\text{sim}(z_m^{r,n},P_{m,y_n}^{comp})/\tau)
}{
\sum_{c=1}^{K}
\exp(\text{sim}(z_m^{r,n},P_{m,c}^{comp})/\tau)
}
}{
\sum_{n,m}g_m^{comp,n}+\epsilon
}
\]

含义：

> reliable disagreement 样本的残差信息仍需保持类别结构，而不是完全散开。

---

#### 9.2.2 不同模态 complementary prototype 保持间隔

\[
\mathcal{L}_{comp}^{sep}
=
\sum_{c=1}^{K}
\sum_{i<j}
\bar{g}_{ij,c}^{comp}
\max
\left(
0,
\delta
-
D(P_{i,c}^{comp},P_{j,c}^{comp})
\right)^2
\]

其中：

\[
\bar{g}_{ij,c}^{comp}
=
\frac{1}{N_c}
\sum_{n:y_n=c}g_{ij}^{comp,n}
\]

最终：

\[
\mathcal{L}_{comp}
=
\mathcal{L}_{comp}^{proto}
+
\beta\mathcal{L}_{comp}^{sep}
\]

含义：

> reliable disagreement 不是噪声，互补原型不应塌缩到同一点，而应保留一定模态差异。

---

## 10. 模块五：关系条件融合与预测

### 10.1 基础融合

最简单实现：

\[
z_f^n=
[z_t^{c,n};z_v^{c,n};z_a^{c,n};z_t^{r,n};z_v^{r,n};z_a^{r,n}]
\]

\[
\hat{y}^n=C_f(z_f^n)
\]

---

### 10.2 关系条件融合

为了进一步利用关系权重：

\[
w_m^{c,n}=
\frac{\exp(g_m^{agr,n})}{\sum_j\exp(g_j^{agr,n})}
\]

\[
w_m^{r,n}=
\frac{\exp(g_m^{comp,n})}{\sum_j\exp(g_j^{comp,n})}
\]

融合：

\[
z_f^n=
[
w_t^{c,n}z_t^{c,n};
w_v^{c,n}z_v^{c,n};
w_a^{c,n}z_a^{c,n};
w_t^{r,n}z_t^{r,n};
w_v^{r,n}z_v^{r,n};
w_a^{r,n}z_a^{r,n}
]
\]

预测：

\[
\hat{y}^n=C_f(z_f^n)
\]

第一版可先使用基础拼接融合，稳定后再加入关系条件融合。

---

## 11. 总损失函数

基础版本：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{comp}
\]

增强版本：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{comp}
+
\lambda_3\mathcal{L}_{orth}
+
\lambda_4\mathcal{L}_{uni}
\]

其中：

- \(\mathcal{L}_{task}\)：主任务损失；
- \(\mathcal{L}_{agr}\)：可靠一致原型对齐；
- \(\mathcal{L}_{comp}\)：可靠不一致残差保留；
- \(\mathcal{L}_{orth}\)：公共/残差正交辅助；
- \(\mathcal{L}_{uni}\)：模态内均匀性辅助，可选。

第一版实现建议：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}^{proto}
+
\lambda_2\mathcal{L}_{comp}^{sep}
\]

跑通后再逐步加入：

- \(\mathcal{L}_{agr}^{cross}\)；
- \(\mathcal{L}_{comp}^{proto}\)；
- relation-conditioned fusion；
- \(\mathcal{L}_{orth}\)。

---

## 12. 训练策略

### Stage 1：Warm-up

前 \(5\sim10\) 个 epoch 只训练：

\[
\mathcal{L}_{task}
\]

以及单模态辅助头。

目的：

- 让 \(p_m\) 有基本可信度；
- 让 \(R_m\) 与 \(A_{ij}\) 不完全随机；
- 避免错误关系判断污染原型。

---

### Stage 2：原型初始化

使用 warm-up 后的训练集特征初始化：

\[
P_{m,c}^{agr}
\]

\[
P_{m,c}^{comp}
\]

按加权均值计算。

---

### Stage 3：联合训练

每个 batch：

1. 编码三模态；
2. 得到 \(z_m^c,z_m^r\)；
3. 计算单模态预测 \(p_m\)；
4. 计算 \(R_m,A_{ij},g^{agr},g^{comp},g^{noise}\)；
5. 使用 EMA 更新原型；
6. 计算 \(\mathcal{L}_{task},\mathcal{L}_{agr},\mathcal{L}_{comp}\)；
7. 反向传播更新网络参数。

---

## 13. 实现注意事项

### 13.1 关系权重 detach

建议：

```python
g_agr = g_agr.detach()
g_comp = g_comp.detach()
```

避免模型通过降低关系权重逃避损失。

---

### 13.2 原型用 buffer，不建议直接学习

原型建议作为 memory buffer，用 EMA 更新，而不是作为可学习参数。

原因：

- 原型代表类别统计中心；
- EMA 更稳定；
- 不易被梯度异常牵引。

---

### 13.3 类别不均衡处理

如果某些类别样本少：

- 使用 EMA；
- 使用 queue；
- 多 batch 累积更新；
- 对少数类别降低更新阈值。

---

### 13.4 回归任务标签离散化

对于 MOSI/MOSEI：

可以将连续标签 \([-3,3]\) 离散为：

- 二分类：positive / negative；
- 三分类：negative / neutral / positive；
- 七分类：Acc-7 对应类别。

建议原型用三分类或七分类，主任务仍保留回归损失。

---

## 14. 论文结构建议

### 1 Introduction

重点回答：

1. 现有 paired multimodal alignment 的默认假设；
2. 为什么 paired 不一定 alignment-positive；
3. 无条件对齐会导致什么问题；
4. CoPA 如何解决。

---

### 2 Related Work

建议四节：

#### 2.1 Cross-modal Alignment and Modality Gap

包括：

- DecAlign；
- CaReFlow；
- CS-Aligner；
- UniAlign。

#### 2.2 Prototype-based Multimodal Learning

包括：

- ProtoMM；
- MASK；
- DecAlign。

#### 2.3 Dynamic Reliability and Imbalanced Multimodal Learning

包括：

- UDML；
- ARL。

#### 2.4 Conditional Positive Alignment

提出本文与已有方法区别：

> 现有方法研究了对齐、原型、可靠性和模态不平衡，但很少显式讨论 paired modalities 是否一定是可对齐正样本。

---

### 3 Method

建议结构：

1. Problem Formulation；
2. Common-Residual Representation；
3. Cross-modal Relation Estimation；
4. Relation-Aware Prototype Bank；
5. Conditional Prototype Learning；
6. Relation-conditioned Fusion；
7. Training Objective。

---

### 4 Experiments

建议结构：

1. Disagreement diagnostic experiment；
2. Main comparison；
3. Ablation study；
4. Relation group analysis；
5. Robustness experiment；
6. Visualization and case study。

---

## 15. 最终一句话总结

> CoPA 质疑“同一样本跨模态配对一定是可对齐正样本”的传统假设，并通过关系感知类别原型，对跨模态关系进行条件性对齐、差异保留或噪声抑制，从而避免错误对齐，同时保留有判别力的跨模态互补信息。
