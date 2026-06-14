# CoPA 实验设计文档

## 0. 实验总目标

CoPA 的实验不能只证明“整体指标更高”，更重要的是证明论文问题成立：

> **同一样本内的多模态配对不一定都是可对齐正样本。**

因此实验要回答四个层次的问题：

1. 数据中是否真实存在跨模态 disagreement？
2. 无条件强制对齐是否会在 high-disagreement 样本上伤害性能？
3. high-disagreement 是否可以进一步区分为“可靠互补”与“噪声失效”？
4. CoPA 的关系判别、关系原型和差异保留是否分别解决了对应问题？

---

## 1. 数据集选择与实验边界

### 1.1 当前论文主线数据集

当前论文定位为监督式多模态融合任务，因此主实验优先选择具有明确标签的文本-视觉-音频数据集。

| 优先级 | 数据集 | 模态 | 任务 | 用途 |
|---|---|---|---|---|
| 第一优先级 | CMU-MOSI | Text + Visual + Audio | 情感分析 | 小规模诊断、消融、可视化 |
| 第一优先级 | CMU-MOSEI | Text + Visual + Audio | 情感分析 | 大规模主实验、稳定性验证 |
| 第二优先级 | CH-SIMS / CH-SIMS-v2 | Text + Visual + Audio | 中文情感分析 | 跨语言泛化 |
| 第二优先级 | IEMOCAP | Text + Visual + Audio | 情绪识别 | 情绪分类任务泛化 |
| 第三优先级 | CREMA-D | Audio + Visual | 情绪识别 | 可靠性、鲁棒性、强弱模态分析 |
| 第三优先级 | Kinetics-Sounds / AVE | Audio + Visual | 音视频分类 / 事件识别 | 非情感补充验证 |

---

### 1.2 不建议作为当前主实验的数据集

| 数据集类型 | 代表数据集 | 暂不作为主实验的原因 |
|---|---|---|
| 图文检索 | MSCOCO、Flickr30k | 当前 CoPA 依赖类别标签与类别原型，检索任务需改成语义簇原型 |
| 视觉语言零样本分类 | ImageNet 系列 | 更适合 CLIP/VLM 测试时适应，不是当前三模态融合主线 |
| 大规模图文预训练 | CC3M、CC12M、LAION | 任务范式和实验成本都过大 |
| 视频文本检索 | VAST150K、MSR-VTT、DiDeMo | 更偏多模态预训练与检索，不适合第一篇 CoPA |

---

## 2. 实验路线总览

建议实验分四个阶段。

### 阶段 1：问题验证实验

目标：证明“paired 不一定 alignment-positive”。

推荐数据集：

```text
CMU-MOSI + CMU-MOSEI
```

核心实验：

1. 训练诊断模型；
2. 计算跨模态 disagreement；
3. 按 disagreement 分组；
4. 比较普通融合和无条件强制对齐在不同组上的表现。

---

### 阶段 2：方法主实验

目标：证明 CoPA 整体有效。

推荐数据集：

```text
CMU-MOSI + CMU-MOSEI + CH-SIMS + IEMOCAP
```

核心实验：

1. 与 SOTA 方法比较；
2. 做完整消融；
3. 分析不同关系组中的性能变化。

---

### 阶段 3：鲁棒性实验

目标：证明可靠性建模不是装饰模块。

推荐数据集：

```text
CMU-MOSI + CMU-MOSEI + CREMA-D 或 Kinetics-Sounds
```

核心实验：

1. 单模态加噪；
2. 模态缺失；
3. 不同噪声强度下的关系权重变化；
4. 可靠性与 noisy-positive 检测分析。

---

### 阶段 4：可选泛化实验

目标：证明 CoPA 不只是情感任务上的现象。

可选数据集：

```text
UR-FUNNY / MUStARD / Kinetics-Sounds / AVE
```

重点：

- UR-FUNNY / MUStARD 用于验证可靠不一致与互补差异；
- Kinetics-Sounds / AVE 用于验证非情感音视频任务。

---

# 3. 阶段 1：Disagreement 分组诊断实验

这是整个论文最关键的 motivation 实验。

## 3.1 实验目的

验证三个问题：

1. 跨模态 disagreement 是否真实存在？
2. 无条件强制对齐是否在 high-disagreement 样本上有害？
3. high-disagreement 是否包含可靠互补信息，而不仅仅是噪声？

---

## 3.2 诊断模型结构

先不要训练完整 CoPA。先训练一个带单模态预测头的普通融合模型。

单模态编码：

\[
h_m=E_m(x_m),\quad m\in\{t,v,a\}
\]

单模态预测：

\[
p_m=\text{Softmax}(C_m(h_m))
\]

多模态融合：

\[
z_f=\text{Fusion}(h_t,h_v,h_a)
\]

多模态预测：

\[
\hat{y}=C_f(z_f)
\]

训练损失：

\[
\mathcal{L}=\mathcal{L}_{task}^{multi}
+\eta(\mathcal{L}_{task}^{t}+\mathcal{L}_{task}^{v}+\mathcal{L}_{task}^{a})
\]

建议：

\[
\eta\in\{0.2,0.5\}
\]

---

## 3.3 标签处理

### 3.3.1 MOSI / MOSEI 二分类

\[
y_{bin}=1[y>0]
\]

优点：简单、稳定。缺点：掩盖细粒度 disagreement。

---

### 3.3.2 MOSI / MOSEI 三分类

推荐用于第一阶段诊断：

| 连续标签 | 离散类别 |
|---|---|
| \(y<-0.5\) | negative |
| \(-0.5\leq y\leq 0.5\) | neutral |
| \(y>0.5\) | positive |

此时：

\[
p_m\in\mathbb{R}^3
\]

---

### 3.3.3 七分类

可用于正式结果对齐 Acc-7，但不建议第一轮诊断直接使用，因为单模态预测更不稳定。

---

## 3.4 计算 disagreement

对每个测试样本 \(n\)，得到：

\[
p_t^n,p_v^n,p_a^n
\]

计算两两 JSD：

\[
\text{JSD}(p_i^n,p_j^n)
=\frac{1}{2}\text{KL}(p_i^n\|q_{ij}^n)
+\frac{1}{2}\text{KL}(p_j^n\|q_{ij}^n)
\]

其中：

\[
q_{ij}^n=\frac{1}{2}(p_i^n+p_j^n)
\]

样本级 disagreement：

\[
D_{sample}^n
=\frac{1}{3}
[\text{JSD}(p_t^n,p_v^n)
+\text{JSD}(p_t^n,p_a^n)
+\text{JSD}(p_v^n,p_a^n)]
\]

---

## 3.5 计算 reliability

单模态可靠性：

\[
R_m^n=1-\frac{H(p_m^n)}{\log K}
\]

其中：

\[
H(p_m^n)=-\sum_{k=1}^{K}p_{m,k}^n\log p_{m,k}^n
\]

样本级可靠性：

\[
R_{sample}^n=\frac{1}{3}(R_t^n+R_v^n+R_a^n)
\]

---

## 3.6 分组方式

### 3.6.1 快速探索版

直接在测试集按 \(D_{sample}\) 三等分：

| 分组 | 样本位置 | 含义 |
|---|---|---|
| Low disagreement | 前 33% | 模态判断高度一致 |
| Medium disagreement | 中间 33% | 中等分歧 |
| High disagreement | 后 33% | 模态冲突明显 |

---

### 3.6.2 正式论文版

更严谨：

1. 在验证集计算 \(D_{sample}\)；
2. 得到 1/3 和 2/3 分位数阈值；
3. 用验证集阈值划分测试集。

这样避免使用测试集分位数带来的质疑。

---

## 3.7 对比模型

阶段 1 只需要比较少量模型。

### 模型 A：Base Fusion

普通融合，不加对齐：

\[
z_f=[h_t;h_v;h_a]
\]

\[
\mathcal{L}=\mathcal{L}_{task}
\]

---

### 模型 B：Unconditional Sample Alignment

无条件样本级强制对齐：

\[
\mathcal{L}=\mathcal{L}_{task}
+\lambda\sum_{i<j}D(z_i,z_j)
\]

其中：

\[
D(z_i,z_j)=1-\cos(z_i,z_j)
\]

先试：

\[
\lambda\in\{0.01,0.05,0.1,0.5\}
\]

---

### 模型 C：Standard Class Prototype

普通类别原型，不区分 agreement / complementary：

\[
P_{m,c}
\]

样本到正确类别原型：

\[
\mathcal{L}_{proto}
=-\log
\frac{
\exp(\text{sim}(z_m,P_{m,y})/\tau)
}{
\sum_{c=1}^{K}\exp(\text{sim}(z_m,P_{m,c})/\tau)
}
\]

跨模态同类原型对齐：

\[
\mathcal{L}_{cross}=\sum_c\sum_{i<j}D(P_{i,c},P_{j,c})
\]

---

### 模型 D：CoPA-Mini

最小版 CoPA：

\[
\mathcal{L}=\mathcal{L}_{task}
+\lambda_1\mathcal{L}_{agr}^{proto}
+\lambda_2\mathcal{L}_{comp}^{sep}
\]

用于初步验证关系感知原型是否优于普通强制对齐。

---

## 3.8 结果表设计

### 表 1：不同 disagreement 组性能

| Method | Low Dis. | Medium Dis. | High Dis. |
|---|---:|---:|---:|
| Base Fusion |  |  |  |
| Uncond. Sample Align |  |  |  |
| Standard Prototype |  |  |  |
| CoPA-Mini |  |  |  |

推荐指标：

- Acc-2；
- F1；
- MAE；
- Corr。

---

### 表 2：强制对齐收益

\[
\Delta_{align}=\text{Perf}_{UncondAlign}-\text{Perf}_{Base}
\]

| Group | \(\Delta_{align}\) |
|---|---:|
| Low disagreement |  |
| Medium disagreement |  |
| High disagreement |  |

理想现象：

\[
\Delta_{Low}>0,\quad \Delta_{High}<0
\]

这说明：

> 无条件对齐不是总有益。

---

## 3.9 进一步区分 high disagreement

将 high-disagreement 样本继续按 reliability 分为：

| 分组 | 条件 | 解释 |
|---|---|---|
| High-D + High-R | disagreement 高，reliability 高 | 可靠不一致，可能是互补/冲突信息 |
| High-D + Low-R | disagreement 高，reliability 低 | 可能是噪声或模态失效 |

具体做法：

在 High disagreement 组内，按 \(R_{sample}\) 中位数二分。

结果表：

| Method | High-D + High-R | High-D + Low-R |
|---|---:|---:|
| Base Fusion |  |  |
| Uncond. Align |  |  |
| CoPA-Mini |  |  |

预期：

- CoPA-Mini 在 High-D + High-R 中提升明显；
- CoPA-Mini 在 High-D + Low-R 中不被噪声严重影响；
- Unconditional Alignment 在两类 high-D 样本中都可能表现较差。

---

# 4. 阶段 2：主性能实验

## 4.1 数据集

标准配置：

```text
CMU-MOSI
CMU-MOSEI
CH-SIMS / CH-SIMS-v2
IEMOCAP
```

---

## 4.2 指标

| 数据集 | 指标 |
|---|---|
| CMU-MOSI | MAE, Corr, Acc-2, Acc-7, F1 |
| CMU-MOSEI | MAE, Corr, Acc-2, Acc-7, F1 |
| CH-SIMS / CH-SIMS-v2 | MAE, Corr, Acc-2 / Acc-3 / Acc-5, F1 |
| IEMOCAP | WAcc, WAF1, Accuracy, F1 |

---

## 4.3 对比方法

### 基础模型

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
- UDML / ARL 相关方法

---

## 4.4 主结果表

### MOSI / MOSEI

| Method | MOSI Acc-7 | MOSI Acc-2 | MOSI F1 | MOSI MAE | MOSI Corr | MOSEI Acc-7 | MOSEI Acc-2 | MOSEI F1 | MOSEI MAE | MOSEI Corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |  |  |  |
| SOTA Methods |  |  |  |  |  |  |  |  |  |  |
| CoPA |  |  |  |  |  |  |  |  |  |  |

---

### CH-SIMS / IEMOCAP

| Method | CH-SIMS Acc | CH-SIMS F1 | CH-SIMS MAE | CH-SIMS Corr | IEMOCAP WAcc | IEMOCAP WAF1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |
| SOTA Methods |  |  |  |  |  |  |
| CoPA |  |  |  |  |  |  |

---

# 5. 阶段 3：消融实验

## 5.1 模块消融

| 版本 | 去掉内容 | 目的 |
|---|---|---|
| Full CoPA | 完整模型 | 最终效果 |
| w/o Relation Identification | 不区分 agreement / comp / noise | 验证关系判别必要性 |
| w/o Reliability | 只用一致性，不用可靠性 | 验证区分噪声与互补的重要性 |
| w/o Agreement Prototype | 去掉一致性原型 | 验证可靠一致对齐作用 |
| w/o Complementary Prototype | 去掉互补原型 | 验证可靠不一致建模作用 |
| w/o Comp Separation | 去掉互补原型间隔约束 | 验证差异不塌缩的重要性 |
| Standard Prototype | 用普通类别原型替代关系原型 | 验证关系感知原型优越性 |
| Sample Alignment | 用样本级对齐替代原型对齐 | 验证原型级训练更稳定 |
| w/o Residual Branch | 去掉残差分支 | 验证互补差异是否有判别力 |
| w/o Orthogonal Reg. | 去掉可选正交正则 | 验证公共/残差分离程度 |

---

## 5.2 分组消融

在 Low / Medium / High disagreement 三组上分别报告消融结果：

| Variant | Low Dis. | Medium Dis. | High Dis. |
|---|---:|---:|---:|
| Full CoPA |  |  |  |
| w/o Agreement Prototype |  |  |  |
| w/o Complementary Prototype |  |  |  |
| w/o Reliability |  |  |  |
| Standard Prototype |  |  |  |

预期：

- w/o Agreement Prototype 主要伤害 Low-disagreement；
- w/o Complementary Prototype 主要伤害 High-D + High-R；
- w/o Reliability 在噪声或 High-D + Low-R 中下降明显。

---

# 6. 阶段 4：鲁棒性实验

## 6.1 单模态加噪

| 模态 | 加噪方式 |
|---|---|
| 文本 | token mask / word dropout / embedding dropout |
| 视觉 | Gaussian noise / frame dropout / feature dropout |
| 音频 | Gaussian noise / time masking / feature dropout |

如果使用预提取特征：

\[
\tilde{x}_m=x_m+\epsilon
\]

\[
\epsilon\sim\mathcal{N}(0,\sigma^2I)
\]

噪声强度：

\[
\sigma\in\{0.1,0.3,0.5,0.7,1.0\}
\]

---

## 6.2 噪声比例

可以设置：

\[
r\in\{25\%,50\%,75\%\}
\]

表示测试集中有多少比例样本的某个模态被污染。

---

## 6.3 结果表

| Method | Clean | Text Noise | Audio Noise | Visual Noise | All Noise |
|---|---:|---:|---:|---:|---:|
| Base Fusion |  |  |  |  |  |
| Dynamic Fusion Baseline |  |  |  |  |  |
| Standard Prototype |  |  |  |  |  |
| CoPA |  |  |  |  |  |

---

## 6.4 关系权重分析

在不同噪声强度下记录：

\[
g^{agr},\quad g^{comp},\quad g^{noise}
\]

预期现象：

- clean 且一致样本：\(g^{agr}\) 高；
- 可靠反差样本：\(g^{comp}\) 高；
- 加噪样本：\(g^{noise}\) 高。

---

# 7. 可视化实验

## 7.1 Motivation Figure

画出：

\[
\Delta_{align}=\text{Perf}_{UncondAlign}-\text{Perf}_{Base}
\]

随 disagreement 增大的变化曲线。

横轴：disagreement bins。  
纵轴：alignment gain。

理想趋势：

```text
Alignment Gain
   ^
   |  positive
   |    *
   |      *
   |        *
   |          0
   |             *
   |                * negative
   +--------------------------> disagreement
       low              high
```

这张图可以作为 Introduction / Motivation 图。

---

## 7.2 原型空间可视化

使用 t-SNE / UMAP 展示：

1. 普通类别原型；
2. agreement prototype；
3. complementary prototype。

预期：

- agreement prototype 跨模态更接近；
- complementary prototype 保留模态间差异；
- noisy 样本不应主导原型分布。

---

## 7.3 关系分组可视化

展示 Low / Medium / High disagreement 样本的表示分布。

对比：

- Base Fusion；
- Unconditional Alignment；
- Standard Prototype；
- CoPA。

---

## 7.4 案例分析

挑选典型样本：

1. reliable agreement：三模态预测一致；
2. reliable disagreement：文本与音频/视觉冲突；
3. noisy-positive：某模态受噪声影响；
4. 讽刺或幽默样本。

展示：

- 单模态预测 \(p_t,p_v,p_a\)；
- 可靠性 \(R_t,R_v,R_a\)；
- 一致性 \(A_{ij}\)；
- 关系权重 \(g^{agr},g^{comp},g^{noise}\)；
- 最终预测。

---

# 8. 实现细节建议

## 8.1 统一分组索引

Disagreement 分组必须由同一个中立诊断模型计算，所有方法使用同一组 Low / Medium / High 索引进行评估。

不要每个模型自己计算分组。

---

## 8.2 诊断模型不能是 CoPA

建议使用：

- Concat + unimodal heads；
- 或 MulT + unimodal heads。

不要用完整 CoPA 来定义 disagreement 分组，否则容易产生偏置。

---

## 8.3 不使用测试标签分组

分组只使用：

\[
p_t,p_v,p_a
\]

测试标签只用于最终计算指标。

---

## 8.4 多随机种子

建议至少：

```text
3 seeds: 1, 2, 3
```

完整论文最好：

```text
5 seeds: 1, 2, 3, 4, 5
```

报告平均值与标准差。

---

## 8.5 超参数搜索

建议搜索：

| 超参数 | 建议范围 |
|---|---|
| \(\lambda_1\) | 0.01, 0.05, 0.1, 0.5 |
| \(\lambda_2\) | 0.01, 0.05, 0.1, 0.5 |
| \(\tau_A\) | 0.1, 0.3, 0.5, 1.0 |
| prototype temperature \(\tau\) | 0.05, 0.1, 0.2 |
| EMA momentum \(\mu\) | 0.9, 0.95, 0.99 |
| margin \(\delta\) | 0.2, 0.5, 1.0 |

---

# 9. 第一周最小任务清单

如果现在开始做，第一周只做最小验证：

1. 下载并跑通 CMU-MOSI；
2. 训练 Concat baseline + unimodal heads；
3. 得到 \(p_t,p_v,p_a\)；
4. 计算 \(D_{sample}\)；
5. 划分 Low / Medium / High disagreement；
6. 训练 Unconditional Sample Alignment；
7. 对比两者在三组上的 Acc-2 / F1 / MAE；
8. 画出 alignment gain 随 disagreement 变化的曲线。

如果观察到：

\[
\Delta_{Low}>0,
\quad
\Delta_{High}<0
\]

说明论文问题基本成立。

如果没有观察到，需要重新检查：

- disagreement 计算是否可靠；
- 单模态预测头是否训练充分；
- 对齐损失是否过强或过弱；
- 是否需要使用三分类而不是二分类；
- 是否需要换到 MOSEI 或讽刺/幽默数据。

---

# 10. 最终实验叙事

论文实验部分应形成如下逻辑：

1. 先证明问题存在：无条件对齐在 high-disagreement 样本上有害；
2. 再证明关系区分必要：high-disagreement 可以分为可靠互补和不可靠噪声；
3. 再证明方法有效：CoPA 在主数据集上优于现有方法；
4. 再证明模块对应问题：agreement prototype 帮助一致样本，complementary prototype 帮助可靠不一致样本，reliability 抑制噪声样本；
5. 最后证明鲁棒性和可解释性：噪声、模态缺失、原型可视化和案例分析。
