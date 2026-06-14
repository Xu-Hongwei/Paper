# 高斯核、MMD、GMM 的关系与区别说明文档  
## ——结合 CS-Aligner、ProtoMM 与 DecAlign 的具体用法

## 0. 总体结论

高斯核、MMD 和 GMM 都经常出现在“分布建模 / 分布对齐 / 跨模态对齐”中，但它们不是同一种东西。

一句话区分：

```text
高斯核 Gaussian Kernel：
    一个样本相似度函数，用来衡量两个特征点近不近。

MMD Maximum Mean Discrepancy：
    一个基于核函数的分布距离，用来比较两个样本集合 / 两个分布是否一致。

GMM Gaussian Mixture Model：
    一个参数化分布模型，用多个可学习高斯分量拟合数据分布，可用于聚类、原型生成或类别语义建模。
```

放到三篇论文中：

```text
CS-Aligner：
    用高斯核 KDE 估计 CS divergence，
    对齐图像分布 p(x) 和文本分布 p(y)。
    它不是 GMM，也不是 MMD 主损失。

ProtoMM / Dynamic Multimodal Prototype Learning：
    把类别原型建模为“文本描述 + 视觉粒子”的离散分布。
    核心方法是 OT，MMD 只是用于图中评价原型分布与 ground truth 分布差异是否下降。
    它不是 GMM，也不是用 MMD 训练。

DecAlign：
    同时使用 GMM 和 MMD，但作用不同。
    GMM 用于异质性特征的类别原型生成，是 heterogeneity alignment 的核心之一；
    MMD 用于同质性特征的分布匹配，是 homogeneity alignment 的一部分。
```

最关键的纠正是：

```text
DecAlign 不能简单说成 MMD 方法。
DecAlign 的异质性对齐核心是 GMM + prototype-guided OT；
DecAlign 的同质性对齐才使用 MMD 做 distribution matching。
```

---

# 1. 高斯核 Gaussian Kernel

## 1.1 定义

高斯核通常写作：

\[
k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(x,y\) | 两个特征点 |
| \(\|x-y\|^2\) | 两个特征点之间的平方欧氏距离 |
| \(\sigma\) | 核宽度，也叫 bandwidth |
| \(k(x,y)\) | 两个点之间的核相似度 |

如果 \(x\) 和 \(y\) 很近：

\[
k(x,y)\approx 1
\]

如果 \(x\) 和 \(y\) 很远：

\[
k(x,y)\approx 0
\]

所以高斯核本质上是一个“软相似度函数”。

---

## 1.2 高斯核不等于高斯分布假设

这一点非常重要。

使用高斯核并不表示：

```text
整个图像特征分布或文本特征分布服从一个高斯分布。
```

它只是表示：

```text
用一个高斯形状的函数衡量两个样本点之间的局部相似度。
```

例如在 KDE 中，每个样本点周围放一个小高斯核：

\[
\hat p(x)=\frac{1}{N}\sum_{i=1}^{N}k(x,x_i)
\]

这不是说整体分布是一个高斯，而是：

```text
每个样本点贡献一个局部高斯小山包；
所有小山包叠加起来，形成一个复杂的非参数分布估计。
```

因此要区分：

| 概念 | 含义 |
|---|---|
| 高斯分布假设 | 假设整个分布是一个 Gaussian，例如 \(p(x)=\mathcal{N}(\mu,\Sigma)\) |
| 高斯核 KDE | 每个样本点放一个 Gaussian-shaped kernel，整体分布由所有样本共同叠加形成 |

所以：

```text
高斯核只是局部相似度工具；
不是整体分布假设。
```

---

# 2. KDE：Kernel Density Estimation

## 2.1 KDE 是什么？

KDE 是核密度估计，是一种非参数分布估计方法。

给定样本集合：

\[
X=\{x_i\}_{i=1}^{N}
\]

KDE 估计分布为：

\[
\hat p(x)=\frac{1}{N}\sum_{i=1}^{N}K_\sigma(x,x_i)
\]

其中 \(K_\sigma\) 可以是高斯核。

直观理解：

```text
每个样本点 x_i 是一个核中心；
每个样本点周围放一个小高斯核；
所有核叠加起来，得到整体分布估计。
```

---

## 2.2 KDE 和 GMM 的根本区别

KDE 和 GMM 都可能由多个“高斯形状”组成，但本质不同。

| 对比项 | KDE with Gaussian Kernel | GMM |
|---|---|---|
| 方法类型 | 非参数密度估计 | 参数化混合模型 |
| 高斯的作用 | 每个样本点周围的局部核函数 | 每个高斯分量表示一个簇 / 原型 |
| 中心位置 | 样本点本身 \(x_i\) | 学出来的 \(\mu_k\) |
| 数量 | 通常等于样本数 \(N\) | 人为设定 \(K\)，通常远小于样本数 |
| 权重 | 常见为均匀权重 \(1/N\) | 学习混合权重 \(\pi_k\) |
| 是否学习参数 | 通常只选择 bandwidth \(\sigma\) | 需要学习 \(\pi_k,\mu_k,\Sigma_k\) |
| 适合场景 | 估计分布重叠、核分布距离 | 聚类、类别原型、语义簇建模 |

一句话：

```text
KDE 是“每个样本一个小核”；
GMM 是“少数几个可学习高斯分量解释整体数据”。
```

---

# 3. MMD：Maximum Mean Discrepancy

## 3.1 MMD 是什么？

MMD 是一种基于核函数的分布距离，用来衡量两个分布是否一致。

给定两个样本集合：

\[
X=\{x_i\}_{i=1}^{M}
\]

\[
Y=\{y_j\}_{j=1}^{N}
\]

MMD 的经验形式为：

\[
\text{MMD}^2(X,Y)
=
\frac{1}{M^2}\sum_{i,j}k(x_i,x_j)
+
\frac{1}{N^2}\sum_{i,j}k(y_i,y_j)
-
\frac{2}{MN}\sum_{i,j}k(x_i,y_j)
\]

为了便于理解，可以记成：

\[
\text{MMD}^2=A+B-2C
\]

其中：

| 项 | 含义 |
|---|---|
| \(A=\frac{1}{M^2}\sum k(x_i,x_j)\) | \(X\) 分布内部相似度 |
| \(B=\frac{1}{N^2}\sum k(y_i,y_j)\) | \(Y\) 分布内部相似度 |
| \(C=\frac{1}{MN}\sum k(x_i,y_j)\) | \(X\) 与 \(Y\) 跨分布相似度 |

如果两个分布很接近，跨分布相似度 \(C\) 会较大，MMD 会较小。

如果两个分布差异很大，\(C\) 会较小，MMD 会较大。

---

## 3.2 MMD 的 RKHS 理解

MMD 可以理解为：

```text
把两个分布映射到 RKHS 中；
比较它们 mean embedding 的距离。
```

公式为：

\[
\text{MMD}^2(p,q)=\|\mu_p-\mu_q\|_{\mathcal{H}}^2
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\mathcal{H}\) | RKHS，再生核希尔伯特空间 |
| \(\mu_p\) | 分布 \(p\) 在 RKHS 中的均值嵌入 |
| \(\mu_q\) | 分布 \(q\) 在 RKHS 中的均值嵌入 |

因此，MMD 的核心问题是：

```text
两个分布在核空间中的平均表示离得远不远？
```

---

## 3.3 MMD 和高斯核的关系

MMD 经常使用高斯核：

\[
k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
\]

但这里仍然不是假设分布整体是高斯。

MMD 中的“高斯”来自核函数，而不是来自分布模型。

因此：

```text
MMD + Gaussian kernel：
    用高斯核计算样本之间的相似度，
    再根据核相似度比较两个分布。

GMM：
    假设整体分布由多个高斯分量混合而成，
    并学习这些高斯分量的参数。
```

---

# 4. GMM：Gaussian Mixture Model

## 4.1 GMM 是什么？

GMM 是高斯混合模型。它假设一个复杂分布可以由多个高斯分量混合而成：

\[
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x;\mu_k,\Sigma_k)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(K\) | 高斯分量数量 |
| \(\pi_k\) | 第 \(k\) 个高斯分量的混合权重 |
| \(\mu_k\) | 第 \(k\) 个高斯分量的均值 / 中心 |
| \(\Sigma_k\) | 第 \(k\) 个高斯分量的协方差 / 形状 |
| \(\mathcal{N}(x;\mu_k,\Sigma_k)\) | 第 \(k\) 个高斯分量的概率密度 |

GMM 需要学习参数：

\[
\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^{K}
\]

通常使用 EM 算法拟合。

---

## 4.2 GMM 的直观理解

GMM 可以理解为：

```text
我认为数据不是一个整体高斯；
而是由 K 个高斯簇组成。
```

例如在多模态情感或分类任务中，不同类别可能形成不同语义簇：

```text
positive 类别一簇；
negative 类别一簇；
neutral 类别一簇。
```

GMM 可以用多个高斯分量分别表示这些簇。

因此，GMM 适合用于：

```text
聚类；
类别原型建模；
语义簇建模；
复杂分布的参数化近似。
```

---

## 4.3 GMM 的软分配

GMM 的一个重要特点是 soft assignment。

一个样本 \(x_n\) 属于第 \(k\) 个高斯分量的概率可以写为：

\[
w_n(k)=
\frac{
\pi_k\mathcal{N}(x_n;\mu_k,\Sigma_k)
}{
\sum_{j=1}^{K}\pi_j\mathcal{N}(x_n;\mu_j,\Sigma_j)
}
\]

这表示：

```text
一个样本不是硬分配给唯一一个原型；
而是以不同概率属于多个原型。
```

这在多模态中很有用，因为一个样本可能同时包含多个语义成分。

---

# 5. CS-Aligner 中的高斯核、KDE 与 CS divergence

## 5.1 CS-Aligner 的目标

CS-Aligner 的目标是解决 CLIP / InfoNCE 的一个问题：

```text
InfoNCE 能让匹配的图文对靠近，
但不能保证图像 embedding 分布 p(x) 和文本 embedding 分布 p(y) 整体重合。
```

因此 CS-Aligner 的目标函数写为：

\[
\min -I(x;y)+\lambda D_{CS}(p(x),p(y))
\]

其中：

| 项 | 含义 |
|---|---|
| \(-I(x;y)\) | 等价于最大化图像和文本之间的互信息，保证样本级语义匹配 |
| \(D_{CS}(p(x),p(y))\) | 最小化图像分布和文本分布之间的差异，保证模态级分布对齐 |
| \(\lambda\) | 平衡互信息项和分布散度项 |

可以理解为：

```text
InfoNCE / MI：
    负责 pairwise semantic alignment。

CS divergence：
    负责 distributional alignment。
```

---

## 5.2 CS divergence 的经验估计

CS-Aligner 用 KDE 估计 CS divergence：

\[
\hat D_{CS}
=
\log
\frac{1}{M^2}\sum_{i,j}k(x_i,x_j)
+
\log
\frac{1}{N^2}\sum_{i,j}k(y_i,y_j)
-
2\log
\frac{1}{MN}\sum_{i,j}k(x_i,y_j)
\]

其中 \(k(\cdot,\cdot)\) 可以使用高斯核：

\[
k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
\]

三项分别是：

| 项 | 含义 |
|---|---|
| \(\sum k(x_i,x_j)\) | 图像模态内部相似度 |
| \(\sum k(y_i,y_j)\) | 文本模态内部相似度 |
| \(\sum k(x_i,y_j)\) | 图像-文本跨模态相似度 |

CS-Aligner 希望：

```text
图像和文本的跨模态相似度增大；
图像分布和文本分布整体重叠增加；
从而缓解 modality gap。
```

---

## 5.3 CS-Aligner 中高斯核的角色

CS-Aligner 中的高斯核不是 GMM，也不是全局高斯分布假设。

它只是用于：

```text
计算样本点之间的局部相似度；
通过所有样本两两核相似度估计分布重叠程度。
```

因此 CS-Aligner 是：

```text
Gaussian kernel + KDE + CS divergence
```

不是：

```text
GMM
```

也不是：

```text
MMD 主损失
```

---

## 5.4 CS divergence 和 MMD 的关系

CS divergence 和 MMD 都使用三类核相似度：

```text
x-x；
y-y；
x-y。
```

但它们的公式结构不同。

MMD 是：

\[
\text{MMD}^2=A+B-2C
\]

CS divergence 是：

\[
D_{CS}=\log A+\log B-2\log C
\]

也可以写成：

\[
D_{CS}=-2\log\frac{C}{\sqrt{AB}}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(A\) | \(x\) 分布内部核相似度 |
| \(B\) | \(y\) 分布内部核相似度 |
| \(C\) | \(x,y\) 跨分布核相似度 |

所以：

```text
MMD：
    比较两个分布 mean embedding 的距离。

CS divergence：
    比较两个分布 mean embedding 的归一化余弦相似度 / 重叠程度。
```

更直观地说：

```text
MMD 问：两个分布离多远？
CS divergence 问：两个分布重叠 / 方向一致程度有多高？
```

---

# 6. ProtoMM / Dynamic Multimodal Prototype Learning 中的 MMD

## 6.1 ProtoMM 的核心问题

ProtoMM 关注的是 CLIP / VLM 测试时适应中的类别原型问题。

传统 CLIP 用文本 prompt 构造类别原型：

```text
类别原型 = 文本描述特征
```

例如：

```text
"a photo of a laptop"
"a photo of a desktop computer"
```

但类别名称可能存在语义歧义。ProtoMM 认为，单纯文本原型无法充分表达视觉类别概念，因此需要把视觉信息也加入原型。

---

## 6.2 ProtoMM 的核心方法：离散多模态原型 + OT

ProtoMM 把类别原型从单一文本向量扩展为：

```text
类别原型 = 文本描述特征 + 视觉粒子
```

它将当前测试图像建模为离散分布：

\[
P_t=\sum_{n=1}^{N}a_t^n\delta_{x_t^n}
\]

其中 \(x_t^n\) 是第 \(n\) 个图像增强视角的特征。

它将第 \(c\) 类多模态原型建模为：

\[
Q_c=
\sum_{m=1}^{M}w_c^m\delta_{z_c^m}
+
\sum_{s=1}^{S}w_c^{M+s}\delta_{e_c^s}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(z_c^m\) | 第 \(c\) 类的第 \(m\) 个文本描述特征 |
| \(e_c^s\) | 第 \(c\) 类的第 \(s\) 个视觉粒子 |
| \(w_c\) | 原型点的重要性权重 |
| \(Q_c\) | 第 \(c\) 类的多模态原型分布 |

然后 ProtoMM 使用 Optimal Transport 计算：

\[
d_{OT}(P_t,Q_c;C_{tc})
\]

其中 \(C_{tc}\) 是图像增强特征和原型点之间的 cosine distance 代价矩阵。

预测时：

```text
哪个类别原型 Q_c 与测试图像分布 P_t 的 OT 距离最小，
测试图像就更可能属于哪个类别。
```

更新时：

```text
通过 transport plan 选择高质量视觉增强特征；
用这些特征更新类别原型中的 visual particles。
```

所以 ProtoMM 的核心是：

```text
离散分布建模 + OT 预测 + OT 引导视觉粒子更新。
```

---

## 6.3 ProtoMM 中 MMD 的作用

ProtoMM 中确实出现 MMD，但它不是训练损失，也不是预测方法。

它主要出现在论文 Fig. 2(b)，用于评价动态原型更新效果。

论文比较了动态更新过程中：

```text
多模态原型分布
和
ground truth 分布
```

之间的分布差异。

结果显示：

```text
KL divergence 从 18.7 降到 9.5；
MMD 从 0.97 降到 0.29。
```

这说明：

```text
随着测试流推进，视觉粒子不断更新；
类别多模态原型越来越接近真实类别分布。
```

因此，ProtoMM 中 MMD 的角色是：

```text
评价指标 / 分析指标
```

不是：

```text
训练损失；
预测距离；
原型生成方法。
```

---

## 6.4 ProtoMM 与 GMM、MMD 的关系总结

```text
ProtoMM：
    不用 GMM。
    不用 MMD 做训练目标。
    用 OT 做图像分布和类别原型分布的匹配。
    用 MMD 作为图示评价指标，说明原型分布逐渐接近 ground truth。
```

---

# 7. DecAlign 中的 GMM 和 MMD

## 7.1 DecAlign 的整体框架

DecAlign 的目标是解决多模态表示中的一个核心问题：

```text
模态独有信息和跨模态共享语义纠缠在一起，
直接融合容易造成语义干扰或过度对齐。
```

因此 DecAlign 先将每个模态的表示解耦为两部分：

```text
1. modality-unique / heterogeneous features：
   模态独有特征，保留文本、视觉、音频等模态自身特性。

2. modality-common / homogeneous features：
   模态共有特征，表达跨模态共享语义。
```

然后分别对齐：

```text
异质性对齐 heterogeneity alignment：
    GMM + prototype-guided multi-marginal OT。

同质性对齐 homogeneity alignment：
    latent semantic alignment + MMD。
```

---

## 7.2 DecAlign 中 GMM 的作用：异质性原型生成

DecAlign 对 modality-unique features 使用 GMM 建模。

每个模态 \(m\) 的原型集合为：

\[
P_m=
\{(\mu_m^1,\Sigma_m^1),(\mu_m^2,\Sigma_m^2),...,(\mu_m^K,\Sigma_m^K)\}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(K\) | 高斯分量数量，设置为下游任务类别数 |
| \(\mu_m^k\) | 第 \(m\) 个模态第 \(k\) 个高斯原型中心 |
| \(\Sigma_m^k\) | 第 \(m\) 个模态第 \(k\) 个高斯原型协方差 |
| \(P_m\) | 第 \(m\) 个模态的 GMM 原型集合 |

每个高斯分量可以理解为一个类别 / 语义原型。

---

## 7.3 DecAlign 中 GMM 的软分配

样本 \(x_m^n\) 属于第 \(k\) 个高斯分量的概率为：

\[
w_m^n(k)=
\frac{
\pi_k\mathcal{N}(x_m^n;\mu_m^k,\Sigma_m^k)
}{
\sum_{j=1}^{K}\pi_j\mathcal{N}(x_m^n;\mu_m^j,\Sigma_m^j)
}
\]

这表示：

```text
一个样本不是硬分给某一个类别原型；
而是以不同概率属于多个高斯原型。
```

这种软分配能更灵活地表示复杂语义结构。

---

## 7.4 DecAlign 为什么用 GMM？

DecAlign 处理的是 modality-unique features，这些特征存在很强异质性：

```text
文本：句法结构、词义、上下文；
视觉：空间布局、物体、场景；
音频：节奏、语调、能量变化。
```

如果直接做点对点对齐，很容易不稳定。

所以 DecAlign 先用 GMM 提取语义原型：

```text
把复杂样本分布压缩成 K 个类别 / 语义原型；
每个原型由均值和协方差描述；
再在原型层面做跨模态 OT 对齐。
```

这比直接样本级对齐更稳健。

---

## 7.5 DecAlign 中 OT 的作用：对齐不同模态的 GMM 原型

GMM 生成各模态原型后，DecAlign 使用 multi-marginal OT 对齐不同模态的原型。

原型间代价不只看中心距离，也看协方差差异：

\[
C_{i,j}(k_i,k_j)
=
\|\mu_i^{k_i}-\mu_j^{k_j}\|^2
+
Tr(\Sigma_i^{k_i}+\Sigma_j^{k_j}-2(\Sigma_i^{k_i}\Sigma_j^{k_j})^{1/2})
\]

这表示：

```text
两个原型中心是否接近；
两个原型分布形状是否相似。
```

因此，DecAlign 的异质性对齐可以总结为：

```text
GMM 生成每个模态的类别原型；
OT 对齐不同模态的类别原型；
样本到原型校准做局部细化。
```

---

## 7.6 DecAlign 中 MMD 的作用：同质性分布对齐

DecAlign 的 MMD 用在 homogeneity alignment，也就是 modality-common features 上。

这部分不是为了构建类别原型，而是为了让不同模态的共享语义分布一致。

MMD 损失为：

\[
L_{MMD}
=
\frac{2}{M(M-1)}
\sum_{1\le i<j\le M}
[
E_{x,x'\sim Z_{com}^{m_i}}k(x,x')
+
E_{y,y'\sim Z_{com}^{m_j}}k(y,y')
-
2E_{x\sim Z_{com}^{m_i},y\sim Z_{com}^{m_j}}k(x,y)
]
\]

其中高斯核为：

\[
k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
\]

这部分作用是：

```text
让不同模态的 common features 在分布层面更加一致；
补充均值、协方差、偏度等显式统计量对齐。
```

---

## 7.7 DecAlign 的同质性对齐由两部分组成

DecAlign 的 homogeneity alignment 不是只有 MMD。

它包括：

```text
1. latent semantic alignment：
   对齐 modality-common features 的均值 μ、协方差 Σ、偏度 Γ。

2. MMD-based distribution alignment：
   在 RKHS 中比较不同模态 common features 的分布差异。
```

因此：

```text
MMD 是 DecAlign 同质性对齐中的分布正则项；
GMM 才是 DecAlign 异质性原型建模的核心。
```

---

# 8. 三篇论文中的方法角色对比

| 论文 | 主要对象 | 是否用高斯核 | 是否用 MMD | 是否用 GMM | 核心工具 | 作用 |
|---|---|---|---|---|---|---|
| CS-Aligner | 图像分布 \(p(x)\) 与文本分布 \(p(y)\) | 是，用于 KDE / CS divergence | 否，主方法不是 MMD | 否 | CS divergence + InfoNCE | 同时做样本级语义对齐和模态级分布对齐 |
| ProtoMM | 测试图像分布 \(P_t\) 与类别多模态原型分布 \(Q_c\) | 不作为核心 | 是，但只是评价指标 | 否 | OT + visual particles | 动态更新类别原型，缓解文本原型歧义 |
| DecAlign | unique features 与 common features | 是，用于 MMD | 是，用于 common features 分布对齐 | 是，用于 unique features 原型生成 | GMM + OT；MMD | 解耦异质/同质特征并分别对齐 |

---

# 9. 三者关系的统一理解

## 9.1 高斯核

```text
高斯核是基础工具。
它本身不是一个分布模型，也不是一个完整分布距离。
它只是计算两个点之间的局部相似度。
```

可以被用于：

```text
KDE；
MMD；
CS divergence；
其他核方法。
```

---

## 9.2 MMD

```text
MMD 是核分布距离。
它通常使用高斯核计算样本两两相似度，
再比较两个分布在 RKHS 中的 mean embedding。
```

在论文中的用法：

```text
DecAlign：
    用 MMD 对齐 modality-common features 的分布。

ProtoMM：
    用 MMD 作为评价指标，说明动态原型更接近 ground truth。

CS-Aligner：
    不用 MMD 作为主方法，而是用 CS divergence。
```

---

## 9.3 GMM

```text
GMM 是参数化分布模型。
它显式学习 K 个高斯分量，
每个分量可以表示一个簇或语义原型。
```

在论文中的用法：

```text
DecAlign：
    用 GMM 为 modality-unique features 生成类别原型；
    再用 OT 对齐不同模态的 GMM 原型。

ProtoMM：
    不用 GMM。

CS-Aligner：
    不用 GMM。
```

---

# 10. 最终总结

## 10.1 按工具总结

```text
高斯核：
    点对点相似度函数。
    可以用于 KDE、MMD、CS divergence。
    不等于整体分布服从高斯。

MMD：
    基于核的分布距离。
    衡量两个分布在 RKHS 中 mean embedding 的距离。
    常用于分布对齐或分布差异评价。

GMM：
    参数化混合模型。
    用少数几个高斯分量拟合复杂分布。
    常用于聚类、原型生成、类别结构建模。
```

## 10.2 按论文总结

```text
CS-Aligner：
    Gaussian kernel KDE + CS divergence。
    每个样本作为核中心，估计图像分布和文本分布的重叠。
    目标是缓解 CLIP / InfoNCE 的 modality gap。

ProtoMM：
    Discrete prototype distribution + OT。
    类别原型由文本描述和视觉粒子组成。
    MMD 只是评价动态更新后原型分布是否更接近 ground truth。

DecAlign：
    Heterogeneity branch：
        GMM 生成 modality-unique features 的类别原型；
        OT 对齐不同模态原型。

    Homogeneity branch：
        统计量对齐 μ、Σ、Γ；
        MMD 对齐 modality-common features 的潜在分布。
```

## 10.3 最重要的记法

```text
CS-Aligner：
    高斯核 KDE + CS divergence，不是 GMM，不是 MMD。

ProtoMM：
    OT 是核心，MMD 只是评价指标，不是 GMM。

DecAlign：
    GMM + OT 处理异质特征；
    MMD 处理同质特征。
```
