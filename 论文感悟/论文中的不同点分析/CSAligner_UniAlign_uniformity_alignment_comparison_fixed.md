# CS-Aligner 与 UniAlign 中 Uniformity（一致性/均匀性）和 Alignment（对齐性）的对比说明

本文整理两篇论文中关于 **uniformity（一致性/均匀性）** 与 **alignment（对齐性）** 的核心思想与差异：

1. **Distributional Vision-Language Alignment by Cauchy-Schwarz Divergence**，下文简称 **CS-Aligner**。
2. **Towards Uniformity and Alignment for Multimodal Representation Learning**，下文简称 **UniAlign**。

两篇论文讨论的是同一个大问题：

> 多模态表示学习希望把不同模态映射到共享空间中，使语义相关样本靠近，同时保持表示空间的可分性。然而，InfoNCE 同时包含 alignment 和 uniformity 两类目标，在多模态场景下二者可能冲突，从而产生 modality gap / distribution gap。

但两篇论文的切入点不同：

- **CS-Aligner** 主要面向 **image-text 双模态视觉语言对齐**。
- **UniAlign** 主要面向 **$M \ge 3$ 的通用多模态表示学习**，例如 image-text-audio-video。

---

## 1. 总体对比

| 对比维度 | CS-Aligner | UniAlign |
|---|---|---|
| 主要对象 | 视觉-语言双模态，即 image-text | 通用多模态，尤其是 $M \ge 3$ 的场景 |
| 问题来源 | CLIP / InfoNCE 主要做 pairwise alignment，忽略图像分布和文本分布的整体差异 | 多模态 InfoNCE 中存在 alignment-uniformity conflict 和 intra-alignment conflict |
| 核心思想 | 在 InfoNCE 外加入 CS divergence，显式对齐 $p(x)$ 和 $p(y)$ | 解耦 alignment 和 uniformity：模态内 uniformity + conflict-free alignment |
| Alignment 方式 | InfoNCE 样本级对齐 + CS divergence 分布级对齐 | Anchor-based alignment + volume-based alignment |
| Uniformity 方式 | 通过 CS divergence 分解，把 uniformity 限制在每个模态内部 | 显式定义每个模态内部的 uniformity loss |
| 分布散度 | Cauchy-Schwarz divergence | Global Hölder divergence |
| 是否适合 unpaired data | 是。CS divergence 可以利用 unpaired image/text、multi-caption 和 token-level data | 主要关注成组多模态表示学习，unpaired data 不是核心 |
| 主要任务 | 图文检索、文生图、视觉语言表示对齐 | 多模态检索、UnCLIP-style generation、通用多模态共享空间学习 |

一句话概括：

```text
CS-Aligner：
    在 CLIP / InfoNCE 外加入 CS divergence，
    让 image distribution 和 text distribution 直接重合。

UniAlign：
    重新设计多模态对比学习目标，
    把 uniformity 和 alignment 拆开，
    从根源上避免多模态 InfoNCE 的内部冲突。
```

---

# 2. 两篇论文共同关注的问题

## 2.1 InfoNCE 的基本作用

CLIP 类方法使用 InfoNCE 做图文对齐，其基本目标是：

```text
匹配的图文对靠近；
不匹配的图文对远离。
```

对于一个图文 batch：

$$
\{(x_i,y_i)\}_{i=1}^{N}
$$

其中 $x_i$ 是图像特征，$y_i$ 是文本特征。

InfoNCE 会让：

$$
x_i \leftrightarrow y_i
$$

靠近，同时让：

$$
x_i \leftrightarrow y_j,\quad j \ne i
$$

远离。

这在样本级语义匹配上很有效，但会带来两个问题：

1. **只做 pairwise alignment，不保证整体分布对齐。**
2. **uniformity 可能和 alignment 发生冲突。**

---

## 2.2 Alignment 和 Uniformity 是什么？

在对比学习中，InfoNCE 可以从两个角度理解：

```text
Alignment：
    让正样本对靠近。

Uniformity：
    让所有样本在表示空间中均匀分散，避免坍塌。
```

在单模态对比学习中，uniformity 通常是好事，因为它能防止所有样本挤在一起。

但在多模态学习中，如果 uniformity 对跨模态样本也产生排斥，就会出现问题：

```text
alignment：
    想让 image 和 text 靠近；

uniformity：
    可能又把 image 和 text 推开。
```

这就是 **alignment-uniformity conflict**。

---

# 3. CS-Aligner 中的一致性与对齐性

## 3.1 CS-Aligner 的核心目标

CS-Aligner 的目标函数为：

$$
\min\; -I(x;y)+\lambda D_{CS}(p(x),p(y))
$$

其中：

| 项 | 含义 | 作用 |
|---|---|---|
| $-I(x;y)$ | 负互信息项 | 最大化图像和文本之间的语义相关性 |
| $D_{CS}(p(x),p(y))$ | CS divergence | 最小化图像特征分布和文本特征分布的差异 |
| $\lambda$ | 权重系数 | 平衡样本级语义对齐和分布级对齐 |

所以 CS-Aligner 的核心思想是：

```text
InfoNCE / MI：
    保证 x_i 和 y_i 有语义对应关系。

CS divergence：
    保证 p(x) 和 p(y) 的整体分布接近。
```

---

## 3.2 CS-Aligner 中的 Alignment

CS-Aligner 中的 alignment 有两层。

### 第一层：样本级对齐

这一层由 InfoNCE 负责：

$$
x_i \leftrightarrow y_i
$$

也就是：

```text
一张狗的图片应该靠近 "a dog"；
一张火车图片应该靠近 "a train"。
```

这保证的是 **pairwise semantic alignment**。

### 第二层：分布级对齐

CS-Aligner 认为，样本级对齐不够。即使每对图文语义相关，也可能出现：

```text
图像 embedding 整体分布在一个区域；
文本 embedding 整体分布在另一个区域。
```

因此，CS-Aligner 显式最小化：

$$
D_{CS}(p(x),p(y))
$$

让：

$$
p(x) \approx p(y)
$$

这保证的是 **distributional alignment**。

---

## 3.3 CS divergence 如何计算？

CS divergence 理论形式为：

$$
D_{CS}(p;q)
=
-\log
\frac{
\left(\int p(\omega)q(\omega)d\omega\right)^2
}{
\left(\int p(\omega)^2d\omega\right)
\left(\int q(\omega)^2d\omega\right)
}
$$

直观理解：

```text
分子：
    p 和 q 的重叠程度。

分母：
    p 自身大小和 q 自身大小的归一化项。
```

所以它类似于：

```text
两个分布函数之间的归一化余弦相似度。
```

如果 $p=q$，则：

$$
D_{CS}=0
$$

如果两个分布几乎不重叠，则 $D_{CS}$ 会很大。

---

## 3.4 CS-Aligner 中的 Gaussian Kernel KDE

实际训练中不知道真实的 $p(x)$ 和 $p(y)$，所以 CS-Aligner 用 KDE 估计：

$$
\hat D_{CS}
=
\log
\left(
\frac{1}{M^2}
\sum_{i,j=1}^{M}
k(x_i,x_j)
\right)
+
\log
\left(
\frac{1}{N^2}
\sum_{i,j=1}^{N}
k(y_i,y_j)
\right)
-
2\log
\left(
\frac{1}{MN}
\sum_{i=1}^{M}
\sum_{j=1}^{N}
k(x_i,y_j)
\right)
$$

其中 $k$ 可以是高斯核：

$$
k(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
$$

三项含义如下：

| 项 | 含义 |
|---|---|
| $\sum k(x_i,x_j)$ | 图像模态内部相似度 |
| $\sum k(y_i,y_j)$ | 文本模态内部相似度 |
| $\sum k(x_i,y_j)$ | 图像-文本跨模态相似度 |

注意：

```text
这里使用高斯核，不等于假设整体分布是高斯分布。
```

它只是用高斯形状的核函数衡量样本点之间的局部相似度。

---

## 3.5 CS-Aligner 中的 Uniformity

CS-Aligner 的关键理论结果是：当 $\lambda=1$ 时，目标可以分解为：

```text
pairwise alignment
+
global cross-modal alignment
+
uniformity on image
+
uniformity on text
```

也就是说，CS-Aligner 把 uniformity 限制在每个模态内部：

```text
image 内部样本保持分散；
text 内部样本保持分散；
image 和 text 之间则通过 CS divergence 对齐。
```

这与原始 InfoNCE 不同。

原始 InfoNCE 中，uniformity 可能对跨模态样本也产生排斥：

```text
image 和 text 本来应该靠近，
但 uniformity 可能把它们推远。
```

CS-Aligner 避免了这一点。

---

## 3.6 CS-Aligner 的核心理解

CS-Aligner 可以总结为：

```text
样本级：
    InfoNCE 保证具体图文对匹配。

分布级：
    CS divergence 保证图像分布和文本分布重合。

Uniformity：
    只在每个模态内部保持分散，
    不再跨模态互相排斥。
```

因此，它解决的是双模态视觉语言对齐中的：

```text
alignment-uniformity conflict
+
modality distribution gap
```

---

# 4. UniAlign 中的一致性与对齐性

## 4.1 UniAlign 的出发点

UniAlign 认为，标准 InfoNCE 在多模态场景中有两个冲突：

```text
1. Alignment-uniformity conflict
2. Intra-alignment conflict
```

其中第一个冲突在 CS-Aligner 中也讨论过。第二个冲突是 UniAlign 针对 $M \ge 3$ 多模态场景提出的。

---

## 4.2 Alignment-uniformity conflict

这个冲突指的是：

```text
alignment 想让正样本跨模态靠近；
uniformity 又可能把跨模态样本推开。
```

例如对于同一个样本：

```text
image_i、text_i、audio_i
```

它们应该靠近。但 InfoNCE 的 uniformity 会推动样本在全局空间中分散，这可能让不同模态之间产生排斥。

---

## 4.3 Intra-alignment conflict

这是 UniAlign 相比 CS-Aligner 更进一步的地方。

当模态数 $M \ge 3$ 时，一个 anchor 模态会同时受到多个正样本的拉动。例如以 image 为 anchor：

```text
image_i 要靠近 text_i；
image_i 要靠近 audio_i；
image_i 要靠近 video_i。
```

如果这些正样本方向不一致，就会出现拉力抵消：

```text
text 往一个方向拉；
audio 往另一个方向拉；
video 往第三个方向拉。
```

最终结果是：

```text
正样本对齐信号变弱；
跨模态对齐不稳定；
modality gap 变大。
```

这就是 **intra-alignment conflict**。

---

## 4.4 UniAlign 的 General Principle

UniAlign 提出两个原则：

```text
1. Intra-modality uniformity
2. Conflict-free alignment
```

也就是：

```text
只在模态内部做 uniformity；
跨模态对齐时避免多个正样本方向互相冲突。
```

---

# 5. UniAlign 中的 Uniformity

## 5.1 模态内 Uniformity

UniAlign 对每个模态单独定义 uniformity：

$$
U(Z^{(m)})
$$

其中：

$$
Z^{(m)}=\{z_i^{(m)}\}_{i=1}^{B}
$$

表示第 $m$ 个模态在一个 batch 中的 embedding。

它只计算同一模态内部样本之间的分散性：

```text
text-text 内部保持均匀；
image-image 内部保持均匀；
audio-audio 内部保持均匀。
```

而不是跨模态做 uniformity：

```text
不再用 uniformity 推开 image-text；
不再用 uniformity 推开 image-audio；
不再用 uniformity 推开 text-audio。
```

---

## 5.2 为什么要模态内 Uniformity？

模态内 uniformity 有两个作用：

```text
1. 防止每个模态内部表示坍塌；
2. 保持检索任务需要的可分性。
```

例如在图文检索中，如果所有图像 embedding 都挤在一起，就无法区分不同图像。

所以 uniformity 仍然是必要的。关键是：

```text
uniformity 应该作用在模态内部；
不应该跨模态破坏正样本对齐。
```

---

# 6. UniAlign 中的 Alignment

## 6.1 Anchor-based Alignment

UniAlign 选择一个模态作为 anchor。例如选择 text 作为 anchor：

```text
image → text
audio → text
video → text
```

这样每个非 anchor 模态都有一个明确的对齐目标。

损失可以理解为：

$$
L_{align}
=
\frac{1}{B(M-1)}
\sum_{i=1}^{B}
\sum_{n\ne a}
\|z_i^{(a)}-z_i^{(n)}\|_2^2
$$

其中：

```text
a：anchor 模态；
n：其他模态；
B：batch size；
M：模态数。
```

这样做可以避免多个正样本方向互相拉扯。

---

## 6.2 Volume-based Alignment

UniAlign 还提出 volume-based alignment。

对于同一个样本的多个模态 embedding：

$$
z_i^{(1)},z_i^{(2)},\ldots,z_i^{(M)}
$$

如果它们完全共线，则它们张成的体积为 0。

如果它们方向差异很大，则张成的体积较大。

所以 UniAlign 通过最小化体积，让同一样本的多个模态表示更共线：

```text
同一样本的 image、text、audio 表示方向更加一致；
减少 intra-alignment conflict。
```

---

## 6.3 UniAlign 的核心理解

UniAlign 可以总结为：

```text
Uniformity：
    每个模态内部自己保持分散。

Alignment：
    跨模态采用 anchor-based alignment，
    或 volume-based alignment，
    避免多正样本非共线导致的拉力冲突。
```

因此，它解决的是多模态场景中的：

```text
alignment-uniformity conflict
+
intra-alignment conflict
```

---

# 7. 分布散度角度的对比

## 7.1 CS-Aligner：Cauchy-Schwarz Divergence

CS-Aligner 直接最小化：

$$
D_{CS}(p(x),p(y))
$$

它适合两个模态分布之间的对齐：

```text
image distribution
vs
text distribution
```

它的优势是：

```text
1. 对称；
2. 可用 KDE 非参数估计；
3. 不需要假设整体分布是高斯；
4. 可以利用 unpaired data；
5. 适合视觉语言双模态分布对齐。
```

---

## 7.2 UniAlign：Global Hölder Divergence

UniAlign 面向多个模态，因此需要一个多分布散度。它提出 global Hölder divergence，用于衡量多个模态分布之间的整体差异：

```text
p_1(z), p_2(z), ..., p_M(z)
```

相比 CS divergence 主要处理两个分布，Hölder divergence 更适合：

```text
text-image-audio-video 等多个模态同时对齐。
```

UniAlign 的理论说明是：

```text
模态内 uniformity
+
anchor-based alignment
```

可以看作最小化 global Hölder divergence 的计算代理。

---

# 8. Alignment 对比总结

| 对比点 | CS-Aligner | UniAlign |
|---|---|---|
| 样本级对齐 | 使用 InfoNCE 保留图文 pair 语义关系 | 使用 anchor-based alignment 对齐多模态正样本 |
| 分布级对齐 | 使用 CS divergence 直接对齐 $p(x)$ 和 $p(y)$ | 通过解耦 loss 间接最小化 global Hölder divergence |
| 是否解决多正样本冲突 | 不主要讨论，因为主要是双模态 | 明确解决 intra-alignment conflict |
| 对齐对象 | image-text pair + image/text distribution | 多模态样本组 + 多模态分布 |
| 适用模态数 | 主要 2 个模态 | 适合 3 个及以上模态 |

---

# 9. Uniformity 对比总结

| 对比点 | CS-Aligner | UniAlign |
|---|---|---|
| Uniformity 来源 | 从 InfoNCE + CS divergence 分解中得到 | 显式设计 intra-modality uniformity loss |
| 作用范围 | 图像内部、文本内部 | 每个模态内部 |
| 是否跨模态排斥 | 避免跨模态排斥 | 明确避免跨模态排斥 |
| 目的 | 保持每个模态内部结构，同时对齐图文分布 | 防止坍塌并保持多模态检索可分性 |
| 关键思想 | CS divergence 把 uniformity 转化为模态内分散 | uniformity 和 alignment 从目标层面解耦 |

---

# 10. 二者关系

两篇论文并不是互相矛盾，而是层级不同。

## 10.1 CS-Aligner 更像双模态版本

CS-Aligner 适合说明：

```text
为什么 CLIP 的 image-text InfoNCE 不足；
为什么 MI 高不等于分布对齐好；
为什么需要加入分布级 CS divergence。
```

它的核心贡献是：

```text
InfoNCE + CS divergence
=
样本级语义对齐 + 分布级图文对齐
```

---

## 10.2 UniAlign 更像多模态扩展

UniAlign 适合说明：

```text
当模态数从 2 个扩展到 3 个或更多时，
InfoNCE 的冲突会更严重。
```

它不仅处理 alignment-uniformity conflict，还处理：

```text
intra-alignment conflict
```

也就是多个正样本方向不一致的问题。

它的核心贡献是：

```text
intra-modality uniformity
+
conflict-free alignment
+
global Hölder divergence theoretical guarantee
```

---

# 11. 适用场景建议

## 11.1 什么时候更适合引用 CS-Aligner？

如果你的研究重点是：

```text
图像-文本对齐；
CLIP modality gap；
文生图；
图文检索；
unpaired image-text；
multi-caption；
token-level image-text alignment。
```

那么 CS-Aligner 更直接。它可以作为：

```text
双模态分布级对齐方法
```

来引用。

---

## 11.2 什么时候更适合引用 UniAlign？

如果你的研究重点是：

```text
三模态或更多模态；
image-text-audio；
多模态共享表示空间；
多模态 InfoNCE 冲突；
模态内 uniformity；
多正样本 alignment conflict。
```

那么 UniAlign 更合适。它可以作为：

```text
多模态 uniformity-alignment 解耦方法
```

来引用。

---

# 12. 最终总结

## 12.1 一句话总结 CS-Aligner

```text
CS-Aligner 在 InfoNCE 的样本级图文对齐之外，
加入 CS divergence 直接对齐 image distribution 和 text distribution，
并将 uniformity 限制在模态内部，
从而缓解双模态视觉语言对齐中的 modality gap。
```

## 12.2 一句话总结 UniAlign

```text
UniAlign 从多模态 InfoNCE 的内部冲突出发，
将 uniformity 和 alignment 显式解耦：
每个模态内部保持 uniformity，
跨模态采用 anchor / volume 方式实现 conflict-free alignment，
并用 global Hölder divergence 给出多模态分布对齐的理论保证。
```

## 12.3 最核心区别

```text
CS-Aligner：
    更关注双模态 image-text 的分布级对齐。
    方法是 InfoNCE + CS divergence。

UniAlign：
    更关注 M ≥ 3 多模态中的冲突解耦。
    方法是 intra-modality uniformity + conflict-free alignment。
```

最终可以概括为：

> **CS-Aligner 是“在 CLIP 对齐上补充分布级 CS divergence”；UniAlign 是“从目标函数层面重构多模态 alignment 和 uniformity”。**
