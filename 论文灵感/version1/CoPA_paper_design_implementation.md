# 关系感知类别原型的条件正样本对齐多模态学习方法

## 0. 论文暂定信息

### 0.1 中文题目

**关系感知类别原型的条件正样本对齐多模态学习方法**

### 0.2 英文题目

**Relation-Aware Class Prototype based Conditional Positive Alignment for Multimodal Learning**

也可以简称为：

**CoPA: Conditional Positive Alignment for Multimodal Learning**

### 0.3 核心关键词

- Multimodal Learning
- Conditional Positive Alignment
- Relation-Aware Prototype
- Cross-modal Agreement
- Complementary Information
- Modality Reliability
- Multimodal Sentiment Analysis

---

# 1. 论文核心问题

## 1.1 传统多模态对齐的隐含假设

现有很多多模态对齐方法默认：

$$
(x_t, x_v, x_a)
$$

来自同一个样本，因此文本、视觉、音频之间天然构成可靠正样本关系，应该被拉近或对齐。

其中：

- $x_t$：文本模态输入；
- $x_v$：视觉模态输入；
- $x_a$：音频模态输入。

传统方法的潜在逻辑是：

$$
\text{paired multimodal samples} \Rightarrow \text{alignment-positive samples}
$$

也就是说，只要属于同一个样本，不同模态就应该在共享语义空间中靠近。

---

## 1.2 本文反思的核心假设

本文认为：

> **Paired multimodal samples are not always alignment-positive.**

即：

> 同一样本内的多模态配对，并不总是可靠的可对齐正样本。

在真实多模态任务中，跨模态关系可能有三种情况：

| 跨模态关系 | 示例 | 是否应强制对齐 |
|---|---|---|
| 可靠且一致 | 文本、语音、表情都表达负面情绪 | 应该强对齐 |
| 可靠但不一致 | 文本表面积极，但语音低沉、表情疲惫 | 不应强制对齐，应保留差异 |
| 不可靠 | 音频噪声、视觉遮挡、文本缺失 | 应降低其训练和融合影响 |

因此，本文的核心问题不是“如何更强地对齐多模态”，而是：

> **什么时候应该对齐，什么时候不应该对齐，什么时候应该保留跨模态差异？**

---

# 2. 论文核心思想

本文提出一种 **条件正样本对齐** 框架。

核心原则是：

> 先判断跨模态关系，再决定对齐、保留还是抑制。

具体来说：

1. 如果两个模态 **可靠且语义一致**，则将其视为 **agreement-positive**，应该通过类别原型进行对齐；
2. 如果两个模态 **可靠但语义不一致**，则将其视为 **complementary-positive**，不应强制拉近，而应保留差异；
3. 如果至少一个模态 **不可靠**，则将其视为 **noisy-positive**，降低其对齐和融合影响。

---

# 3. 方法整体架构

论文方法由五个部分组成：

| 编号 | 模块 | 作用 | 是否创新 |
|---|---|---|---|
| 0 | 单模态特征编码器 | 提取文本、视觉、音频表示 | 否 |
| 1 | 跨模态关系判别模块 | 判断 agreement / complementary / noisy | 是 |
| 2 | 关系感知类别原型构建模块 | 构造一致性原型和互补性原型 | 是 |
| 3 | 条件原型对齐与差异保留模块 | 一致信息对齐，互补信息保留 | 是 |
| 4 | 融合预测模块 | 输出分类或回归结果 | 否 |

整体流程为：

$$
x_m
\rightarrow h_m
\rightarrow (z_m^c,z_m^r)
\rightarrow \text{relation identification}
\rightarrow \text{relation-aware prototypes}
\rightarrow \text{conditional alignment and fusion}
\rightarrow \hat{y}
$$

其中：

$$
m \in \{t,v,a\}
$$

分别表示文本、视觉、音频。

---

# 4. 模块 0：单模态特征编码器

## 4.1 输入定义

给定一个三模态样本：

$$
X^n = \{x_t^n,x_v^n,x_a^n,y^n\}
$$

其中：

- $x_t^n$：第 $n$ 个样本的文本输入；
- $x_v^n$：第 $n$ 个样本的视觉输入；
- $x_a^n$：第 $n$ 个样本的音频输入；
- $y^n$：样本标签。

对于分类任务：

$$
y^n \in \{1,2,\dots,K\}
$$

对于回归任务，可以先将连续情感分数离散化为类别，用于构造类别原型，同时保留回归损失作为主任务损失。

---

## 4.2 单模态编码

每个模态通过独立编码器得到单模态特征：

$$
h_m^n = E_m(x_m^n)
$$

其中：

$$
m \in \{t,v,a\}
$$

即：

$$
h_t^n = E_t(x_t^n)
$$

$$
h_v^n = E_v(x_v^n)
$$

$$
h_a^n = E_a(x_a^n)
$$

### 可选实现

如果使用预提取特征：

| 模态 | 特征 |
|---|---|
| 文本 | BERT / RoBERTa embedding |
| 视觉 | OpenFace |
| 音频 | COVAREP |

如果端到端训练：

| 模态 | 编码器 |
|---|---|
| 文本 | Transformer / BERT |
| 视觉 | ViT / ResNet / Video Transformer |
| 音频 | CNN / Transformer / wav2vec 特征 |

---

## 4.3 公共特征与残差特征分离

对每个模态特征 $h_m^n$，使用两个投影头：

$$
z_m^{c,n}=P_m^c(h_m^n)
$$

$$
z_m^{r,n}=P_m^r(h_m^n)
$$

其中：

- $z_m^{c,n}$：公共语义特征，用于可靠一致关系的对齐；
- $z_m^{r,n}$：残差/私有特征，用于可靠不一致关系的差异保留。

通常进行 L2 归一化：

$$
z_m^{c,n} \leftarrow \frac{z_m^{c,n}}{\|z_m^{c,n}\|_2}
$$

$$
z_m^{r,n} \leftarrow \frac{z_m^{r,n}}{\|z_m^{r,n}\|_2}
$$

---

# 5. 模块 1：跨模态关系判别模块

## 5.1 单模态预测分布

对每个模态接一个轻量单模态分类头：

$$
p_m^n = \text{Softmax}(C_m(z_m^{c,n}))
$$

其中：

$$
p_m^n \in \mathbb{R}^{K}
$$

表示第 $m$ 个模态对第 $n$ 个样本的类别预测分布。

---

## 5.2 模态可靠性估计

使用预测熵估计模态可靠性：

$$
H(p_m^n) = -\sum_{k=1}^{K}p_{m,k}^n\log p_{m,k}^n
$$

归一化后定义可靠性：

$$
R_m^n = 1-\frac{H(p_m^n)}{\log K}
$$

其中：

$$
R_m^n \in [0,1]
$$

含义：

- $R_m^n$ 越接近 1，说明该模态预测越确定，可靠性越高；
- $R_m^n$ 越接近 0，说明该模态预测越混乱，可能受噪声、缺失或弱语义影响。

---

## 5.3 跨模态语义一致性估计

对于任意两个模态 $i,j$，计算预测分布之间的 Jensen-Shannon Divergence：

$$
\text{JSD}(p_i^n,p_j^n)
=
\frac{1}{2}\text{KL}(p_i^n\|q^n)
+
\frac{1}{2}\text{KL}(p_j^n\|q^n)
$$

其中：

$$
q^n=\frac{1}{2}(p_i^n+p_j^n)
$$

然后定义语义一致性：

$$
A_{ij}^n
=
\exp\left(
-\frac{\text{JSD}(p_i^n,p_j^n)}{\tau_A}
\right)
$$

其中：

- $\tau_A$：温度系数；
- $A_{ij}^n \in [0,1]$。

含义：

- $A_{ij}^n$ 高：两个模态预测分布接近，语义判断一致；
- $A_{ij}^n$ 低：两个模态预测分布差异大，可能存在冲突或互补信息。

---

## 5.4 三类跨模态关系权重

### 5.4.1 Agreement-positive 权重

可靠且一致：

$$
g_{ij}^{agr,n}=R_i^nR_j^nA_{ij}^n
$$

含义：

> 两个模态都可靠，并且语义判断一致，因此适合进行强对齐。

---

### 5.4.2 Complementary-positive 权重

可靠但不一致：

$$
g_{ij}^{comp,n}=R_i^nR_j^n(1-A_{ij}^n)
$$

含义：

> 两个模态都可靠，但语义判断不一致，因此不应强行对齐，而应保留差异。

---

### 5.4.3 Noisy-positive 权重

不可靠关系：

$$
g_{ij}^{noise,n}=1-R_i^nR_j^n
$$

含义：

> 至少一个模态不可靠，应降低其对齐和融合影响。

---

## 5.5 模态级关系权重

对于每个模态 $m$，可以将它与其他模态的关系求平均，得到模态级权重：

$$
g_m^{agr,n}
=
\frac{1}{M-1}
\sum_{j\neq m}g_{mj}^{agr,n}
$$

$$
g_m^{comp,n}
=
\frac{1}{M-1}
\sum_{j\neq m}g_{mj}^{comp,n}
$$

其中：

$$
M=3
$$

表示文本、视觉、音频三种模态。

实际训练时建议：

$$
g_m^{agr,n},g_m^{comp,n}
$$

使用 `stop-gradient`，避免模型通过操纵关系权重来逃避损失。

---

# 6. 模块 2：关系感知类别原型构建模块

## 6.1 为什么不用普通样本对齐？

如果直接使用样本级对齐：

$$
D(z_i^{c,n},z_j^{c,n})
$$

容易受到单个样本噪声、单模态预测错误和局部偶然关系的影响。

因此，本文使用类别原型作为稳定语义锚点。

---

## 6.2 为什么不用普通类别原型？

普通类别原型通常是：

$$
P_{m,c}
=
\frac{1}{N_c}
\sum_{n:y_n=c}z_m^n
$$

但同一类别内部也可能包含不同跨模态关系：

- 有些样本模态一致；
- 有些样本模态冲突；
- 有些样本某个模态受噪声影响；
- 有些样本存在文本-语音-视觉反差。

如果使用单一类别原型，会把一致信息和互补差异全部平均掉。

因此，本文提出 **关系感知类别原型**。

---

## 6.3 两类关系感知类别原型

对每个模态 $m$、每个类别 $c$，维护两个原型：

$$
P_{m,c}^{agr}
$$

$$
P_{m,c}^{comp}
$$

其中：

| 原型 | 来源 | 表示含义 |
|---|---|---|
| $P_{m,c}^{agr}$ | 可靠且一致样本 | 第 $m$ 个模态、第 $c$ 类中可对齐的公共语义中心 |
| $P_{m,c}^{comp}$ | 可靠但不一致样本 | 第 $m$ 个模态、第 $c$ 类中应保留的互补差异中心 |

---

## 6.4 Agreement prototype 更新

在一个 mini-batch 中，对类别为 $c$ 的样本，用 $g_m^{agr,n}$ 加权更新：

$$
\bar{P}_{m,c}^{agr}
=
\frac{
\sum_{n:y_n=c}g_m^{agr,n}z_m^{c,n}
}{
\sum_{n:y_n=c}g_m^{agr,n}+\epsilon
}
$$

然后使用 EMA 更新原型：

$$
P_{m,c}^{agr}
\leftarrow
\text{Norm}
\left(
\mu P_{m,c}^{agr}
+
(1-\mu)\bar{P}_{m,c}^{agr}
\right)
$$

其中：

- $\mu$：EMA 动量系数，建议取 $0.9\sim0.99$；
- $\epsilon$：防止除零；
- $\text{Norm}(\cdot)$：L2 归一化。

如果当前 batch 中某个类别 $c$ 的 agreement 权重过小：

$$
\sum_{n:y_n=c}g_m^{agr,n}<\epsilon
$$

则该类别原型本轮不更新。

---

## 6.5 Complementary prototype 更新

互补原型使用残差特征 $z_m^{r,n}$ 更新：

$$
\bar{P}_{m,c}^{comp}
=
\frac{
\sum_{n:y_n=c}g_m^{comp,n}z_m^{r,n}
}{
\sum_{n:y_n=c}g_m^{comp,n}+\epsilon
}
$$

EMA 更新：

$$
P_{m,c}^{comp}
\leftarrow
\text{Norm}
\left(
\mu P_{m,c}^{comp}
+
(1-\mu)\bar{P}_{m,c}^{comp}
\right)
$$

注意：

- agreement prototype 使用公共特征 $z_m^c$；
- complementary prototype 使用残差特征 $z_m^r$。

这样可以避免可靠不一致的信息污染公共对齐空间。

---

# 7. 模块 3：条件原型对齐与差异保留

## 7.1 对可靠一致关系进行原型对齐

可靠一致关系应该进入公共语义空间，并通过类别原型实现稳定对齐。

---

### 7.1.1 样本到本模态 agreement prototype 的对齐

使用 prototype contrastive loss：

$$
\mathcal{L}_{agr}^{proto}
=
-
\frac{
\sum_{n}\sum_m
 g_m^{agr,n}
\log
\frac{
\exp(\text{sim}(z_m^{c,n},P_{m,y_n}^{agr})/\tau)
}{
\sum_{c=1}^{K}
\exp(\text{sim}(z_m^{c,n},P_{m,c}^{agr})/\tau)
}
}{
\sum_n\sum_m g_m^{agr,n}+\epsilon
}
$$

其中：

- $\text{sim}(\cdot,\cdot)$：余弦相似度；
- $\tau$：对比学习温度系数；
- $P_{m,y_n}^{agr}$：第 $m$ 个模态中真实类别 $y_n$ 对应的一致性原型。

作用：

> 可靠一致样本的公共特征应靠近本模态、本类别的 agreement prototype，同时远离其他类别的 agreement prototype。

---

### 7.1.2 跨模态 agreement prototype 对齐

对同一类别 $c$，不同模态的 agreement prototype 应该对齐：

$$
\mathcal{L}_{agr}^{cross}
=
\sum_{c=1}^{K}
\sum_{i<j}
\bar{g}_{ij,c}^{agr}
D(P_{i,c}^{agr},P_{j,c}^{agr})
$$

其中：

$$
\bar{g}_{ij,c}^{agr}
=
\frac{1}{N_c}
\sum_{n:y_n=c}g_{ij}^{agr,n}
$$

距离函数可以使用：

$$
D(a,b)=1-\cos(a,b)
$$

该损失的含义：

> 如果某类别中两个模态经常表现为可靠且一致，那么这两个模态在该类别上的 agreement prototype 应该靠近。

---

### 7.1.3 总 agreement loss

$$
\mathcal{L}_{agr}
=
\mathcal{L}_{agr}^{proto}
+
\alpha
\mathcal{L}_{agr}^{cross}
$$

其中：

- $\alpha$：控制跨模态原型对齐强度。

---

## 7.2 对可靠不一致关系进行差异保留

可靠不一致关系不应被直接当作噪声丢弃，也不应被强行对齐。它可能包含有用的互补信息。

例如：

- 文本表面积极，但语音低沉；
- 表情不明显，但音频情绪强；
- 文本语义与视觉表情存在反差；
- 讽刺和幽默场景中，多模态不一致本身就是判别线索。

---

### 7.2.1 残差特征靠近本模态 complementary prototype

使用互补原型对比损失：

$$
\mathcal{L}_{comp}^{proto}
=
-
\frac{
\sum_{n}\sum_m
 g_m^{comp,n}
\log
\frac{
\exp(\text{sim}(z_m^{r,n},P_{m,y_n}^{comp})/\tau)
}{
\sum_{c=1}^{K}
\exp(\text{sim}(z_m^{r,n},P_{m,c}^{comp})/\tau)
}
}{
\sum_n\sum_m g_m^{comp,n}+\epsilon
}
$$

该损失的作用：

> 可靠但不一致的信息仍然需要类别结构，而不是随意散开。

它保证：

- 同一类别的互补残差信息聚合；
- 不同类别的互补残差信息分离。

---

### 7.2.2 不同模态 complementary prototype 保持间隔

对于可靠但不一致的信息，不同模态的 complementary prototype 不应塌缩到一起。

定义 margin separation loss：

$$
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
$$

其中：

$$
\bar{g}_{ij,c}^{comp}
=
\frac{1}{N_c}
\sum_{n:y_n=c}g_{ij}^{comp,n}
$$

含义：

- 如果某类别中两个模态经常可靠但不一致；
- 那么它们的 complementary prototype 不应被拉到一起；
- 但也不需要无限推远，只要距离大于 margin $\delta$。

---

### 7.2.3 总 complementary loss

$$
\mathcal{L}_{comp}
=
\mathcal{L}_{comp}^{proto}
+
\beta
\mathcal{L}_{comp}^{sep}
$$

其中：

- $\beta$：控制互补原型间隔约束强度。

---

# 8. 模块 4：融合预测模块

最终融合公共特征和残差特征：

$$
z_f^n
=
\text{Fusion}
(
z_t^{c,n},z_v^{c,n},z_a^{c,n},
z_t^{r,n},z_v^{r,n},z_a^{r,n}
)
$$

## 8.1 简单实现

直接拼接：

$$
z_f^n
=
[z_t^{c,n};z_v^{c,n};z_a^{c,n};z_t^{r,n};z_v^{r,n};z_a^{r,n}]
$$

然后：

$$
\hat{y}^n=C(z_f^n)
$$

## 8.2 加权融合实现

也可以根据模态可靠性进行加权：

$$
w_m^n
=
\frac{\exp(R_m^n)}
{\sum_{j}\exp(R_j^n)}
$$

$$
z_f^n
=
\sum_m
w_m^n[z_m^{c,n};z_m^{r,n}]
$$

但为了避免方法过于复杂，初始版本建议使用拼接 + MLP。

---

# 9. 总体训练目标

## 9.1 任务损失

分类任务：

$$
\mathcal{L}_{task}
=
-\sum_n\log p(y_n|z_f^n)
$$

回归任务：

$$
\mathcal{L}_{task}
=
\frac{1}{N}
\sum_n
\|\hat{y}^n-y^n\|_1
$$

或：

$$
\mathcal{L}_{task}
=
\frac{1}{N}
\sum_n
(\hat{y}^n-y^n)^2
$$

---

## 9.2 可选正交解耦约束

为了防止公共特征和残差特征混杂，可以加入轻量正则：

$$
\mathcal{L}_{orth}
=
\sum_n\sum_m
|\cos(z_m^{c,n},z_m^{r,n})|
$$

该项不是核心创新，只是辅助约束。

---

## 9.3 总损失

基础版本：

$$
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{comp}
$$

增强版本：

$$
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{comp}
+
\lambda_3\mathcal{L}_{orth}
$$

其中：

- $\lambda_1$：agreement alignment 权重；
- $\lambda_2$：complementary preservation 权重；
- $\lambda_3$：正交约束权重。

---

# 10. 训练策略

## 10.1 Stage 1：Warm-up

前若干 epoch 只训练：

$$
\mathcal{L}_{task}
$$

以及单模态预测头。

目的：

- 让 $p_m$ 有基本预测能力；
- 让 $R_m$ 和 $A_{ij}$ 不至于完全随机；
- 避免早期错误关系判断污染原型。

建议：

$$
5\sim10 \text{ epochs}
$$

---

## 10.2 Stage 2：原型初始化

使用 warm-up 后的特征，在训练集上初始化：

$$
P_{m,c}^{agr}
$$

$$
P_{m,c}^{comp}
$$

具体方法：

$$
P_{m,c}^{agr}
=
\text{Norm}
\left(
\frac{
\sum_{n:y_n=c}g_m^{agr,n}z_m^{c,n}
}{
\sum_{n:y_n=c}g_m^{agr,n}+\epsilon
}
\right)
$$

$$
P_{m,c}^{comp}
=
\text{Norm}
\left(
\frac{
\sum_{n:y_n=c}g_m^{comp,n}z_m^{r,n}
}{
\sum_{n:y_n=c}g_m^{comp,n}+\epsilon
}
\right)
$$

---

## 10.3 Stage 3：联合训练

每个 mini-batch 执行：

1. 输入三模态数据；
2. 得到 $h_m$、$z_m^c$、$z_m^r$；
3. 计算单模态预测 $p_m$；
4. 计算可靠性 $R_m$；
5. 计算一致性 $A_{ij}$；
6. 计算关系权重 $g^{agr}$、$g^{comp}$、$g^{noise}$；
7. 使用 EMA 更新 $P^{agr}$、$P^{comp}$；
8. 计算 $\mathcal{L}_{task}$、$\mathcal{L}_{agr}$、$\mathcal{L}_{comp}$；
9. 反向传播更新网络参数。

注意：

- 原型作为 memory buffer，不通过梯度更新；
- 关系权重 $g$ 建议 stop-gradient；
- EMA 更新时应跳过权重过低的类别；
- batch size 太小时，原型更新可能不稳定，可以使用 queue 或全局 memory bank。

---

# 11. 最小可行实现版本

为了先跑通实验，建议先实现最小版本。

## 11.1 保留模块

最小版本只保留：

1. 跨模态关系判别；
2. agreement prototype；
3. complementary prototype；
4. $\mathcal{L}_{agr}^{proto}$；
5. $\mathcal{L}_{comp}^{sep}$；
6. 任务预测损失。

---

## 11.2 最小版本损失

$$
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}^{proto}
+
\lambda_2\mathcal{L}_{comp}^{sep}
$$

暂时不加入：

- $\mathcal{L}_{agr}^{cross}$；
- $\mathcal{L}_{comp}^{proto}$；
- $\mathcal{L}_{orth}$。

跑通后再逐步加入增强项。

---

# 12. 实验设计

## 12.1 主实验数据集

推荐使用：

| 数据集 | 模态 | 任务 |
|---|---|---|
| CMU-MOSI | 文本、视觉、音频 | 情感分析 |
| CMU-MOSEI | 文本、视觉、音频 | 情感分析 |
| CH-SIMS / CH-SIMS-v2 | 文本、视觉、音频 | 中文多模态情感分析 |

可选补充：

| 数据集 | 作用 |
|---|---|
| MUStARD | 多模态讽刺检测，适合验证可靠不一致信息 |
| UR-FUNNY | 多模态幽默检测，适合验证跨模态反差信息 |

---

## 12.2 对比方法

### 基础方法

- Text-only
- Audio-only
- Visual-only
- Early Fusion
- Late Fusion
- Concatenation + MLP

### 经典多模态方法

- TFN
- LMF
- MulT
- MISA
- Self-MM
- MMIM
- DMD

### 相关最新方法

- DecAlign
- CaReFlow
- UDML
- ARL

---

## 12.3 主性能指标

对于 MOSI / MOSEI：

| 指标 | 含义 |
|---|---|
| Acc-2 | 二分类准确率 |
| Acc-7 | 七分类准确率 |
| F1 | 分类 F1 |
| MAE | 平均绝对误差 |
| Corr | 相关系数 |

对于 CH-SIMS / CH-SIMS-v2：

| 指标 | 含义 |
|---|---|
| Acc-2 | 二分类准确率 |
| Acc-3 | 三分类准确率 |
| Acc-5 | 五分类准确率 |
| F1 | 分类 F1 |
| MAE | 回归误差 |
| Corr | 相关系数 |

---

# 13. 关键诊断实验

本文不能只做主性能表，必须证明“paired 不一定 alignment-positive”这个问题真实存在。

## 13.1 Disagreement 分组实验

根据跨模态 disagreement 将测试样本分成三组：

$$
D_{sample}^n
=
\frac{1}{|\mathcal{P}|}
\sum_{i<j}
\text{JSD}(p_i^n,p_j^n)
$$

其中：

$$
\mathcal{P}=\{(t,v),(t,a),(v,a)\}
$$

按照 $D_{sample}$ 排序，分成：

| 分组 | 含义 |
|---|---|
| Low disagreement | 模态语义高度一致 |
| Medium disagreement | 模态部分不一致 |
| High disagreement | 模态冲突明显 |

比较不同方法在三组上的表现：

| 方法 | Low | Medium | High |
|---|---:|---:|---:|
| 普通融合 |  |  |  |
| 强制样本对齐 |  |  |  |
| 普通类别原型 |  |  |  |
| 本文方法 |  |  |  |

预期现象：

- 强制对齐在 Low disagreement 样本上可能有效；
- 强制对齐在 High disagreement 样本上可能伤害性能；
- 本文方法在 Low disagreement 上保持对齐收益，在 High disagreement 上避免错误对齐。

---

## 13.2 样本对齐 vs 普通类别原型 vs 关系感知类别原型

对比：

| 版本 | 含义 |
|---|---|
| Sample Alignment | 直接做样本点对点对齐 |
| Standard Class Prototype | 每个类别一个普通原型 |
| Relation-Aware Class Prototype | 本文的一致性/互补性双原型 |

要证明：

> 关系感知类别原型比样本对齐更稳定，比普通类别原型更能保留跨模态差异。

---

## 13.3 可靠不一致是否有用

对比：

| 版本 | 含义 |
|---|---|
| Only common feature | 只使用公共特征 |
| Only residual feature | 只使用残差特征 |
| Common + residual | 使用完整融合 |
| w/o complementary prototype | 去掉互补原型 |
| Full model | 完整方法 |

预期：

- 在 high-disagreement 样本上，残差信息应有明显贡献；
- 去掉 complementary prototype 后，性能应下降；
- 完整模型在讽刺、幽默或情感反差样本上更强。

---

# 14. 消融实验设计

| 消融版本 | 去掉内容 | 目的 |
|---|---|---|
| Full Model | 完整模型 | 最终效果 |
| w/o Relation Identification | 不区分 agreement/comp/noise | 验证关系判别重要性 |
| w/o Agreement Prototype | 去掉一致性原型 | 验证可靠一致对齐作用 |
| w/o Complementary Prototype | 去掉互补性原型 | 验证可靠差异保留作用 |
| w/o Comp Separation | 去掉互补原型间隔约束 | 验证差异不塌缩的重要性 |
| w/o Reliability | 只用一致性，不用可靠性 | 验证区分噪声与互补差异的必要性 |
| w/o Agreement | 只保留互补分支 | 验证公共语义对齐必要性 |
| Standard Prototype | 用普通类别原型替代关系原型 | 验证关系感知原型优越性 |
| Sample Alignment | 用样本对齐替代原型对齐 | 验证原型级训练更稳定 |

---

# 15. 噪声鲁棒性实验

## 15.1 单模态加噪

对每个模态分别加噪：

| 模态 | 加噪方式 |
|---|---|
| 文本 | token mask / word dropout / embedding dropout |
| 视觉 | Gaussian noise / frame dropout / feature dropout |
| 音频 | Gaussian noise / time masking / feature dropout |

如果使用预提取特征，可以简单使用：

$$
\tilde{x}_m=x_m+\epsilon
$$

$$
\epsilon\sim\mathcal{N}(0,\sigma^2I)
$$

噪声强度：

$$
\sigma\in\{0.1,0.3,0.5,0.7,1.0\}
$$

---

## 15.2 预期分析

当某个模态受噪声影响时：

- 该模态预测熵升高；
- 可靠性 $R_m$ 下降；
- 与该模态相关的 $g^{agr}$、$g^{comp}$ 都应下降；
- $g^{noise}$ 上升；
- 该模态不会严重污染 agreement prototype 或 complementary prototype。

---

# 16. 可视化实验

建议至少做四类可视化。

## 16.1 原型空间可视化

使用 t-SNE / UMAP 展示：

- 普通类别原型；
- agreement prototype；
- complementary prototype。

目标：

> agreement prototype 应更跨模态接近，complementary prototype 应保留模态间差异。

---

## 16.2 Disagreement 分组可视化

画出 Low / Medium / High disagreement 样本的特征分布。

观察：

- 强制对齐是否使 high-disagreement 样本混乱；
- 本文方法是否能保持互补差异结构。

---

## 16.3 关系权重变化可视化

在不同噪声强度下画：

$$
g^{agr}, g^{comp}, g^{noise}
$$

预期：

- clean 且一致样本：$g^{agr}$ 高；
- 可靠反差样本：$g^{comp}$ 高；
- 加噪样本：$g^{noise}$ 高。

---

## 16.4 案例分析

挑选具体样本：

1. 文本、语音、视觉都一致；
2. 文本和语音/视觉冲突；
3. 某个模态受噪声影响；
4. 讽刺或幽默样本。

展示：

- 单模态预测；
- $R_m$；
- $A_{ij}$；
- $g^{agr}$、$g^{comp}$、$g^{noise}$；
- 最终预测变化。

---

# 17. 实现注意事项

## 17.1 关系权重需要 stop-gradient

如果不 stop-gradient，模型可能通过降低 $g^{agr}$ 或 $g^{comp}$ 来逃避对齐损失。

建议实现：

```python
g_agr = g_agr.detach()
g_comp = g_comp.detach()
```

---

## 17.2 原型不建议作为可学习参数

建议使用 memory buffer + EMA 更新。

原因：

- 类别原型代表统计中心；
- EMA 更稳定；
- 可避免原型被梯度过度牵引。

---

## 17.3 Early training 关系判断不稳定

需要 warm-up。

建议：

- 前 5 到 10 个 epoch 不启用原型损失；
- 只训练主任务和单模态预测；
- warm-up 后再初始化原型。

---

## 17.4 Batch size 太小的问题

如果 batch size 太小，某些类别在 batch 中样本不足，原型更新会不稳定。

解决方法：

1. 使用较大 batch size；
2. 使用 memory queue；
3. 多 batch 累积后更新原型；
4. 使用全局训练集周期性刷新原型。

---

## 17.5 回归任务如何构造类别原型

对于 MOSI / MOSEI 这类情感回归任务，可以将连续标签离散化。

例如：

$$
[-3,-2),[-2,-1),[-1,0),(0,1],(1,2],(2,3]
$$

或者使用原始 Acc-7 的七分类划分。

原型类别使用离散标签，主任务仍可使用回归损失。

---

# 18. 论文贡献总结

## 18.1 贡献 1：提出条件正样本对齐问题

本文指出，同一样本内的多模态配对并不总是可靠的可对齐正样本。传统无条件对齐可能在跨模态冲突、弱相关和噪声污染场景中造成错误对齐。

---

## 18.2 贡献 2：提出跨模态关系三分机制

本文根据模态可靠性和跨模态语义一致性，将配对模态关系划分为：

1. agreement-positive；
2. complementary-positive；
3. noisy-positive。

这使模型能够区分：

- 应该对齐的信息；
- 应该保留的互补差异；
- 应该降低影响的噪声信息。

---

## 18.3 贡献 3：提出关系感知类别原型

本文为每个类别、每个模态分别构建：

$$
P_{m,c}^{agr}
$$

$$
P_{m,c}^{comp}
$$

分别建模：

- 可对齐公共语义；
- 应保留互补差异。

该设计比样本级对齐更稳定，也比普通类别原型更能保留跨模态关系结构。

---

## 18.4 贡献 4：提出条件原型对齐与差异保留目标

本文对 agreement-positive 样本进行类别原型对齐，对 complementary-positive 样本进行残差信息结构化保留，对 noisy-positive 样本降低训练影响，从而避免错误对齐，同时保留有判别力的跨模态差异。

---

# 19. 一句话总结

本文提出一种关系感知类别原型的条件正样本对齐框架，核心观点是：同一样本内的多模态配对不一定都是可对齐正样本。模型应先根据模态可靠性和跨模态语义一致性判断关系，再对可靠一致信息进行原型对齐，对可靠不一致信息进行差异保留，并降低不可靠模态的影响，从而同时避免错误对齐和互补信息损失。
