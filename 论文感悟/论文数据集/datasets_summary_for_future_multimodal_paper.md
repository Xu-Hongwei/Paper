# 多模态论文数据集总结与后续使用建议

> 目的：概括已参考的八篇多模态论文中使用的数据集，并给出后续论文实验设计时如何选择和使用这些数据集的建议。本文重点面向“图像/文本/音频多模态融合、可靠一致关系原型对齐、可靠不一致关系差异保留、动态融合与鲁棒性分析”等后续研究方向。

---

## 1. 数据集总体分类

从八篇参考论文的实验设计来看，相关数据集大致可以分为六类：

| 类别 | 代表数据集 | 主要模态 | 主要任务 | 对后续论文的价值 |
|---|---|---|---|---|
| 三模态情感/情绪数据集 | CMU-MOSI、CMU-MOSEI、CH-SIMS、IEMOCAP | Text + Visual + Audio | 情感分析、情绪识别 | 最适合作为你后续论文的主实验数据集 |
| 动态融合与鲁棒性数据集 | MVSA-Single、CREMA-D、Kinetics-Sounds | Image/Text 或 Audio/Visual | 情感分类、情绪识别、音视频分类 | 适合验证模态可靠性估计、噪声鲁棒性、动态融合 |
| 模态不平衡/训练优化数据集 | CREMA-D、Kinetics-Sounds、AVE、UCF101、MOSI | Audio + Visual 或多模态 | 分类、事件识别、动作识别 | 适合验证强弱模态不平衡、梯度调制、优化依赖关系 |
| 视觉语言原型/零样本分类数据集 | ImageNet、ImageNet-A/V2/R/Sketch、Caltech101、OxfordPets、Flowers102 等 | Image + Class Text | 零样本分类、测试时适应 | 适合原型学习、类别语义歧义、VLM 测试时适应 |
| 图文检索/生成/无配对匹配数据集 | MSCOCO、Flickr30k、Visual Genome、CC3M、CC12M、LAION-HR、Urban-1k、DOCCI | Image + Text | 图文检索、图文生成、无配对匹配 | 适合图文对齐、分布级对齐、词-区域原型知识 |
| 视频-文本-音频共享空间数据集 | VAST150K、MSR-VTT、DiDeMo、ActivityNet、VGGSound | Text + Video + Audio | 多模态预训练、视频检索、跨模态生成 | 适合多模态共享嵌入空间、InfoNCE 改进、检索/生成联合验证 |

---

## 2. 最适合后续论文的主数据集

如果后续论文的主题是：

```text
可靠一致关系原型对齐 + 可靠不一致关系差异保留
```

那么最核心的数据集应该是三模态情感/情绪数据集：

| 数据集 | 模态 | 任务 | 推荐程度 | 使用目的 |
|---|---|---|---|---|
| CMU-MOSI | Text + Visual + Audio | 多模态情感分析 | ★★★★★ | 小规模经典 benchmark，适合快速验证方法、做消融和可视化 |
| CMU-MOSEI | Text + Visual + Audio | 多模态情感分析 | ★★★★★ | 大规模 benchmark，适合证明泛化性和稳定性 |
| CH-SIMS | Text + Visual + Audio | 中文多模态情感分析 | ★★★★☆ | 证明方法在中文/跨语言场景中有效 |
| IEMOCAP | Text + Visual + Audio | 多模态情绪识别 | ★★★★☆ | 证明方法不只适用于 sentiment regression，也适用于 emotion classification |

### 推荐使用方式

#### 最小实验配置

```text
CMU-MOSI + CMU-MOSEI
```

适合第一阶段验证方法是否成立，尤其适合先跑“disagreement 分组实验”。

#### 标准论文配置

```text
CMU-MOSI + CMU-MOSEI + CH-SIMS + IEMOCAP
```

这是比较完整的多模态融合论文配置。MOSI/MOSEI 证明英文情感分析有效，CH-SIMS 证明中文场景有效，IEMOCAP 证明方法可以迁移到情绪识别分类任务。

#### 加强版配置

```text
主实验：
CMU-MOSI、CMU-MOSEI、CH-SIMS、IEMOCAP

鲁棒性补充：
CREMA-D、Kinetics-Sounds 或 MVSA-Single

可选图文扩展：
MSCOCO、Flickr30k
```

这种配置更扎实，但工作量也明显更大。

---

## 3. 各类数据集的具体用途

### 3.1 三模态情感/情绪数据集

#### 代表数据集

```text
CMU-MOSI
CMU-MOSEI
CH-SIMS
IEMOCAP
```

#### 适合验证的问题

这类数据集最适合验证多模态融合方法，因为它们通常同时包含文本、视觉和音频三种模态。对于你的后续论文，它们可以用来验证：

1. 模态一致关系是否能帮助融合；
2. 模态不一致关系是否不应该被强行对齐；
3. 可靠一致样本是否适合做原型对齐；
4. 可靠不一致样本是否应该保留模态差异；
5. 动态融合权重是否比静态融合更有效；
6. 方法是否能同时适用于回归和分类任务。

#### 常用指标

| 数据集 | 常用指标 |
|---|---|
| CMU-MOSI | MAE、Corr、Acc-2、Acc-7、F1 |
| CMU-MOSEI | MAE、Corr、Acc-2、Acc-7、F1 |
| CH-SIMS | MAE、Corr、Acc-2/Acc-3、F1 |
| IEMOCAP | WAcc、WAF1、Accuracy、F1 |

#### 后续论文中的使用建议

可以把它们作为主实验数据集，并围绕它们设计如下实验：

```text
1. 主结果对比：
   与 MulT、MISA、Self-MM、DMD、DecAlign 等方法比较。

2. 关系分组实验：
   将样本划分为 reliable agreement、reliable disagreement、unreliable disagreement 等组，分别报告性能。

3. 模块消融：
   去掉可靠性估计、去掉一致关系原型对齐、去掉不一致关系差异保留、去掉动态融合。

4. 可视化：
   t-SNE 展示不同关系组的模态表示分布；
   原型距离展示一致样本是否更靠近；
   差异保留模块展示不一致样本是否没有被过度压缩。
```

---

### 3.2 动态融合与鲁棒性数据集

#### 代表数据集

```text
MVSA-Single
CREMA-D
Kinetics-Sounds
```

#### 数据集特点

| 数据集 | 模态 | 任务 | 特点 |
|---|---|---|---|
| MVSA-Single | Image + Text | 多模态情感分类 | 图文双模态，适合测试图像或文本噪声影响 |
| CREMA-D | Audio + Visual | 情绪识别 | 音视频模态差异明显，适合分析强弱模态和动态质量变化 |
| Kinetics-Sounds | Audio + Visual | 音视频分类 | 音频和视觉均有判别信息，适合鲁棒性和模态缺失实验 |

#### 适合验证的问题

这类数据集适合用来证明：

```text
1. 当某一模态被噪声污染时，方法是否能降低该模态权重；
2. 当模态质量动态变化时，融合权重是否能自适应调整；
3. 是否可以避免弱模态被双重抑制；
4. 方法是否比静态融合更鲁棒。
```

#### 后续论文中的使用建议

如果你的论文强调“可靠性估计”，这类数据集很适合作为补充实验。可以设计：

```text
噪声鲁棒性实验：
- 给视觉模态加 Gaussian noise / salt noise；
- 给音频模态加噪声或遮挡；
- 比较 clean、low-noise、high-noise 三种条件下的结果。

模态缺失实验：
- 随机 mask 某一模态；
- 比较 static fusion、uncertainty fusion、你的可靠性融合。

动态融合可解释性实验：
- 画出不同噪声强度下各模态权重变化曲线；
- 证明模型确实在降低不可靠模态贡献。
```

---

### 3.3 模态不平衡与训练优化数据集

#### 代表数据集

```text
CREMA-D
Kinetics-Sounds
AVE
UCF101
MOSI
```

#### 适合验证的问题

这些数据集适合分析：

```text
1. 强模态是否压制弱模态；
2. 多模态联合训练是否真的优于单模态；
3. 不同模态的优化速度是否一致；
4. 模态贡献是否应该平衡；
5. 梯度调制或单模态辅助损失是否有效。
```

#### 后续论文中的使用建议

如果你的方法中包含训练阶段的约束，例如：

```text
单模态辅助损失
原型约束
梯度调制
一致/不一致关系分组优化
可靠性监督
```

那么可以使用 CREMA-D、Kinetics-Sounds 或 AVE 做补充实验。

建议的实验设计：

```text
1. 单模态 vs 多模态：
   分别训练 Text-only、Audio-only、Visual-only、Multimodal。

2. 强弱模态分析：
   比较不同模态在训练过程中的 accuracy / loss / gradient norm。

3. 关系分组下的模态贡献：
   分析 reliable agreement 和 reliable disagreement 样本中，哪个模态更可靠。

4. 梯度或权重曲线：
   观察模型是否过度依赖某个强模态。
```

---

### 3.4 视觉语言原型与零样本分类数据集

#### 代表数据集

```text
ImageNet
ImageNet-A
ImageNet-V2
ImageNet-R
ImageNet-Sketch
Caltech101
OxfordPets
StanfordCars
Flowers102
Food101
DTD
EuroSAT
UCF101
```

#### 适合验证的问题

这类数据集主要用于视觉语言模型，例如 CLIP 的 zero-shot classification 和 test-time adaptation。它们适合验证：

```text
1. 类别文本原型是否存在语义歧义；
2. 视觉粒子是否能补充文本原型；
3. 类别原型是否能动态适应测试流；
4. 方法在 ImageNet OOD 变体上是否鲁棒。
```

#### 后续论文中的使用建议

如果你的后续论文仍然是三模态情感融合，这类数据集不是主线，不建议强行加入。

只有当你的方法扩展到下面方向时，才建议使用：

```text
视觉语言模型 VLM
CLIP 测试时适应
类别原型学习
图像-类别文本对齐
zero-shot classification
```

可以设计的实验包括：

```text
1. 原型有效性实验：
   比较 text-only prototype 和 multimodal prototype。

2. 类别语义歧义实验：
   选择语义相近类别，看视觉粒子是否提升区分能力。

3. OOD 泛化实验：
   在 ImageNet-A/R/Sketch/V2 上测试。
```

---

### 3.5 图文检索、生成与无配对匹配数据集

#### 代表数据集

```text
MSCOCO
Flickr30k
Visual Genome
CC3M
CC12M
LAION-HR
Urban-1k
DOCCI
```

#### 数据集用途

| 数据集 | 用途 |
|---|---|
| MSCOCO | 图文检索、图文生成、多 caption 对齐、unpaired alignment |
| Flickr30k | 图文检索、图文匹配、短文本图文对应 |
| Visual Genome | 构建 word-region prototype、区域级视觉语义知识 |
| CC3M / CC12M | 大规模图文对齐训练 |
| LAION-HR | 高分辨率图文生成/对齐 |
| Urban-1k / DOCCI | 长文本图文检索、细粒度语义对齐 |

#### 适合验证的问题

这类数据集适合图文方向的论文，特别是：

```text
1. 图像和文本的整体分布是否对齐；
2. InfoNCE 是否只做样本级匹配；
3. unpaired image-text data 是否可以利用；
4. word embedding 是否可以构造视觉原型；
5. OOD words 是否能找到对应视觉表示；
6. token-level / region-level alignment 是否有效。
```

#### 后续论文中的使用建议

如果你的方法未来扩展到图文对齐，可以这样使用：

```text
1. 图文检索：
   用 MSCOCO / Flickr30k，指标为 R@1、R@5、R@10。

2. 无配对图文匹配：
   用 MSCOCO / Flickr30k，构造 paired / unpaired 对比。

3. 词-区域原型：
   用 Visual Genome 构建 word-region prototype。

4. 分布级图文对齐：
   用 CC3M / CC12M 训练，用 MSCOCO / Flickr30k 测试。
```

如果你的论文主线是情感融合，这类数据集可以作为后续扩展，不建议一开始加入。

---

### 3.6 视频-文本-音频共享空间数据集

#### 代表数据集

```text
VAST150K
MSR-VTT
DiDeMo
ActivityNet
VGGSound
```

#### 适合验证的问题

这类数据集更适合多模态预训练或共享嵌入空间学习，适合验证：

```text
1. 多模态 InfoNCE 是否存在 alignment-uniformity conflict；
2. 多模态正样本是否存在 intra-alignment conflict；
3. 视频、文本、音频是否能进入统一共享空间；
4. 表示是否同时适合检索和生成。
```

#### 后续论文中的使用建议

如果你的方法未来扩展为多模态预训练目标，可以使用这些数据集：

```text
训练：
VAST150K

检索测试：
MSR-VTT、DiDeMo、ActivityNet

生成或跨模态对齐测试：
VGGSound
```

但如果当前目标是写一篇三模态情感融合论文，这类数据集实验成本较高，暂时不建议作为主实验。

---

## 4. 后续论文实验设计建议

如果你的论文主题是“可靠一致关系原型对齐 + 可靠不一致关系差异保留”，可以按以下层次设计实验。

### 4.1 主结果实验

| 实验目的 | 推荐数据集 | 指标 |
|---|---|---|
| 证明整体性能优于 SOTA | CMU-MOSI、CMU-MOSEI、CH-SIMS、IEMOCAP | MAE、Corr、Acc、F1、WAcc、WAF1 |

实验内容：

```text
与经典多模态融合方法比较：
TFN、LMF、MulT、MISA、Self-MM、DMD、DecAlign 等。

报告主指标：
MOSI/MOSEI：MAE、Corr、Acc-2、Acc-7、F1
CH-SIMS：MAE、Corr、Acc、F1
IEMOCAP：WAcc、WAF1
```

---

### 4.2 关系分组实验

| 实验目的 | 推荐数据集 | 关键问题 |
|---|---|---|
| 证明 agreement / disagreement 分组有意义 | CMU-MOSI、CMU-MOSEI | 不同关系组是否表现不同 |

建议分组：

```text
1. Reliable Agreement：
   多模态预测一致，且置信度高。

2. Reliable Disagreement：
   模态预测不一致，但至少部分模态可靠。

3. Unreliable Agreement：
   多模态看似一致，但整体置信度低。

4. Unreliable Disagreement：
   多模态不一致，且可靠性低。
```

可报告：

```text
每组样本比例；
每组单模态性能；
每组多模态融合性能；
去掉某一模块后每组性能变化；
每组模态距离 / 原型距离 / 表示分布。
```

这部分是你论文最关键的实验之一，因为它直接证明“可靠一致”和“可靠不一致”不是人为硬造出来的概念，而是数据中真实存在的现象。

---

### 4.3 模块消融实验

| 模块 | 消融方式 | 预期现象 |
|---|---|---|
| 可靠性估计 | 去掉 reliability score | 模型无法区分可靠/不可靠样本，分组性能下降 |
| 一致关系原型对齐 | 去掉 agreement prototype alignment | reliable agreement 样本性能下降明显 |
| 不一致关系差异保留 | 去掉 disagreement difference preservation | reliable disagreement 样本性能下降明显 |
| 动态融合权重 | 改成 static fusion | 噪声或模态退化下性能下降 |
| 单模态辅助监督 | 去掉 unimodal loss | 单模态表示质量下降，关系判断不稳定 |

推荐数据集：

```text
CMU-MOSI
CMU-MOSEI
CH-SIMS
```

---

### 4.4 鲁棒性实验

| 实验类型 | 推荐数据集 | 目的 |
|---|---|---|
| 模态噪声 | MOSI、MOSEI、CREMA-D | 验证可靠性估计是否有效 |
| 模态缺失 | MOSI、MOSEI | 验证模型是否依赖单一强模态 |
| 弱模态退化 | CREMA-D、Kinetics-Sounds | 验证动态融合是否避免弱模态双重抑制 |

实验设置可以包括：

```text
Gaussian noise
Salt noise
Random masking
Text token dropout
Audio feature masking
Visual frame dropout
```

可报告：

```text
clean setting 性能；
low-noise setting 性能；
high-noise setting 性能；
不同噪声强度下的模态权重变化；
不同噪声强度下的可靠性分数变化。
```

---

### 4.5 可视化实验

推荐数据集：

```text
CMU-MOSI
CMU-MOSEI
```

建议可视化内容：

```text
1. t-SNE / UMAP：
   展示不同模态表示在对齐前后的分布。

2. 原型距离热力图：
   reliable agreement 样本是否更靠近类别原型。

3. 差异保留可视化：
   reliable disagreement 样本是否保留模态差异，而不是被强行压缩到同一点。

4. 模态权重曲线：
   不同噪声强度下，模型是否自动降低低质量模态权重。

5. 分组比例图：
   展示 agreement/disagreement/reliable/unreliable 的样本数量分布。
```

---

## 5. 推荐实验路线图

### 第一阶段：先验证问题是否成立

```text
数据集：CMU-MOSI、CMU-MOSEI

要做：
1. 单模态预测；
2. 多模态预测；
3. agreement / disagreement 分组；
4. 每组样本比例和性能统计；
5. 初步可视化。
```

目标：证明“可靠一致”和“可靠不一致”在数据中确实存在，并且不同组对融合方法有不同需求。

---

### 第二阶段：验证方法是否有效

```text
数据集：CMU-MOSI、CMU-MOSEI、CH-SIMS、IEMOCAP

要做：
1. 与 SOTA 比主结果；
2. 做完整消融；
3. 分析每个模块对不同关系组的贡献。
```

目标：证明你的方法不仅整体指标好，而且每个模块都对应解决一个明确问题。

---

### 第三阶段：验证鲁棒性

```text
数据集：CMU-MOSI、CMU-MOSEI、CREMA-D 或 Kinetics-Sounds

要做：
1. 加噪声；
2. 模态缺失；
3. 弱模态退化；
4. 动态权重变化分析。
```

目标：证明“可靠性建模”不是装饰模块，而是在模态质量变化时真正有用。

---

### 第四阶段：可选扩展

如果后续想把论文扩展到视觉语言方向：

```text
图文检索：MSCOCO、Flickr30k
词-区域原型：Visual Genome
零样本分类：ImageNet 系列
```

如果后续想把论文扩展到多模态预训练方向：

```text
训练：VAST150K
检索：MSR-VTT、DiDeMo、ActivityNet
生成：VGGSound
```

---

## 6. 最终建议

对于当前论文，不建议一开始使用太多任务范式完全不同的数据集。最合理路线是：

```text
第一优先级：
CMU-MOSI、CMU-MOSEI

第二优先级：
CH-SIMS、IEMOCAP

第三优先级：
CREMA-D、Kinetics-Sounds、MVSA-Single

暂不优先：
ImageNet 系列、MSCOCO、Flickr30k、Visual Genome、VAST150K
```

原因是：

```text
MOSI/MOSEI/CH-SIMS/IEMOCAP 与你的论文主题最贴合，都是典型的文本-视觉-音频多模态融合任务。
CREMA-D/Kinetics-Sounds/MVSA-Single 适合补充鲁棒性和模态质量变化实验。
ImageNet/MSCOCO/Flickr30k/Visual Genome 更偏视觉语言原型或图文匹配，不是当前三模态情感融合论文的主线。
VAST150K/MSR-VTT/DiDeMo/ActivityNet/VGGSound 更偏大规模多模态预训练和检索生成，实验成本高，暂时不建议作为主实验。
```

一句话总结：

> 后续论文主线最好围绕 CMU-MOSI、CMU-MOSEI、CH-SIMS、IEMOCAP 展开；先用 MOSI/MOSEI 跑出 disagreement 分组现象，再用完整数据集验证方法性能，最后用噪声/缺失模态实验证明可靠性建模的必要性。

