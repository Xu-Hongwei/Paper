# CoPA 实验设置与执行方案

> 论文方法：**CoPA: Conditional Positive Alignment via Relation-Aware Prototypes for Multimodal Fusion**  
> 实验目标：验证“paired multimodal samples are not always alignment-positive”是否成立，并验证 CoPA 在监督式多模态融合任务中的有效性。

---

## 1. 实验总目标

本文实验不应只做主结果表，而应分层验证以下问题：

1. **现象是否成立**  
   无条件跨模态对齐是否在 high-disagreement 样本上伤害性能？

2. **方法是否有效**  
   CoPA 是否能在主任务上优于经典融合、强制对齐、普通类别原型和相关 SOTA？

3. **模块是否必要**  
   Agreement prototype、complementary prototype、reliability、relation identification 是否各自有效？

4. **可靠不一致是否有用**  
   High disagreement 中是否存在 high reliability 样本，这部分是否包含有用互补信息？

5. **鲁棒性是否提升**  
   噪声、模态缺失、模态退化下，CoPA 是否避免不可靠模态污染原型和融合结果？

---

## 2. 数据集选择

### 2.1 当前论文定位

当前论文定位为：

> **监督式多模态融合中的条件正样本对齐。**

因此主数据集应具备：

- 明确监督标签；
- 多模态输入；
- 能训练单模态预测头；
- 能构造类别原型；
- 存在跨模态一致、互补和噪声关系。

---

### 2.2 主实验数据集

| 数据集 | 模态 | 任务 | 使用目的 |
|---|---|---|---|
| **CMU-MOSI** | Text + Visual + Audio | 情感分析 | 小规模经典 benchmark，适合快速验证 disagreement 现象、消融和可视化 |
| **CMU-MOSEI** | Text + Visual + Audio | 情感分析 | 大规模 benchmark，验证现象和方法稳定性 |
| **CH-SIMS / CH-SIMS-v2** | Text + Visual + Audio | 中文情感分析 | 验证跨语言/中文场景泛化 |
| **IEMOCAP** | Text + Visual + Audio | 情绪识别 | 验证方法不只适用于 sentiment regression，也适用于 emotion classification |

### 推荐主实验组合

最小配置：

```text
CMU-MOSI + CMU-MOSEI
```

标准配置：

```text
CMU-MOSI + CMU-MOSEI + CH-SIMS + IEMOCAP
```

---

### 2.3 鲁棒性补充数据集

| 数据集 | 模态 | 任务 | 使用目的 |
|---|---|---|---|
| **CREMA-D** | Audio + Visual | 情绪识别 | 验证音视频模态质量变化、强弱模态、噪声鲁棒性 |
| **Kinetics-Sounds** | Audio + Visual | 音视频分类 | 验证非文本三模态外的泛化和鲁棒性 |
| **AVE** | Audio + Visual | 音视频事件识别 | 验证事件识别场景下的模态互补 |
| **MVSA-Single** | Image + Text | 图文情感分类 | 验证双模态图文监督融合场景 |

建议优先：

```text
CREMA-D 或 Kinetics-Sounds
```

---

### 2.4 暂不作为主实验的数据集

| 数据集类型 | 示例 | 暂不优先原因 |
|---|---|---|
| 图文检索 | MSCOCO、Flickr30k | 当前方法依赖类别标签和单模态预测分布，图文检索需要改成语义簇原型或检索分布 |
| 视觉语言零样本 | ImageNet 系列 | 更适合 CLIP/VLM 测试时适应，不是当前监督式融合主线 |
| 大规模图文预训练 | CC3M、CC12M、LAION | 任务成本高，目标不同 |
| 视频文本检索 | MSR-VTT、DiDeMo、ActivityNet | 更偏预训练/检索，不适合第一篇 CoPA 论文主线 |

---

## 3. 实验整体路线

### 第一阶段：现象诊断

目的：证明 paired 不一定 alignment-positive。

数据集：

```text
CMU-MOSI
CMU-MOSEI
```

实验：

1. 训练诊断模型；
2. 计算单模态预测；
3. 计算 disagreement；
4. 按 disagreement 分组；
5. 比较 Concat 与 Unconditional Alignment 的分组性能。

---

### 第二阶段：方法有效性

数据集：

```text
CMU-MOSI
CMU-MOSEI
CH-SIMS
IEMOCAP
```

实验：

1. 主结果对比；
2. 消融；
3. 原型对比；
4. 关系分组性能；
5. 可视化。

---

### 第三阶段：鲁棒性补充

数据集：

```text
CREMA-D
Kinetics-Sounds
```

实验：

1. 单模态加噪；
2. 模态缺失；
3. 弱模态退化；
4. 关系权重变化曲线。

---

## 4. 实验 1：Disagreement 诊断实验

这是最关键的 motivation 实验。

### 4.1 实验目的

验证：

> 无条件对齐在 low-disagreement 样本上可能有效，但在 high-disagreement 样本上可能伤害性能。

如果该现象成立，说明：

\[
\text{paired} \neq \text{always alignment-positive}
\]

---

### 4.2 诊断模型

先训练一个简单的多模态模型，带单模态预测头。

#### 结构

\[
h_m=E_m(x_m)
\]

\[
p_m=\text{Softmax}(C_m(h_m))
\]

\[
z_f=\text{Fusion}(h_t,h_v,h_a)
\]

\[
\hat{y}=C_f(z_f)
\]

#### 训练损失

\[
\mathcal{L}
=
\mathcal{L}_{task}^{multi}
+
\eta
(\mathcal{L}_{task}^{t}
+
\mathcal{L}_{task}^{v}
+
\mathcal{L}_{task}^{a})
\]

其中：

\[
\eta \in [0.2,0.5]
\]

目的：

- 得到可靠的 \(p_t,p_v,p_a\)；
- 用于后续 disagreement 分组。

---

### 4.3 标签处理

#### MOSI / MOSEI

原始标签为连续情感值 \([-3,3]\)。

建议三分类用于 disagreement 计算：

| 连续标签 | 类别 |
|---|---|
| \(y<-0.5\) | negative |
| \(-0.5\leq y\leq0.5\) | neutral |
| \(y>0.5\) | positive |

单模态头输出：

\[
p_m\in\mathbb{R}^{3}
\]

主任务仍可以保留回归损失或分类损失。

---

### 4.4 计算 disagreement

对测试集每个样本，得到：

\[
p_t^n,p_v^n,p_a^n
\]

计算两两 JSD：

\[
D_{tv}^n=\text{JSD}(p_t^n,p_v^n)
\]

\[
D_{ta}^n=\text{JSD}(p_t^n,p_a^n)
\]

\[
D_{va}^n=\text{JSD}(p_v^n,p_a^n)
\]

样本级 disagreement：

\[
D_{sample}^n
=
\frac{1}{3}(D_{tv}^n+D_{ta}^n+D_{va}^n)
\]

---

### 4.5 计算 reliability

每个模态：

\[
R_m^n=
1-\frac{H(p_m^n)}{\log K}
\]

样本级可靠性：

\[
R_{sample}^n=
\frac{1}{3}(R_t^n+R_v^n+R_a^n)
\]

---

### 4.6 分组方式

#### 前期探索

按测试集 \(D_{sample}\) 三等分：

| 组 | 含义 |
|---|---|
| Low disagreement | 前 33% |
| Medium disagreement | 中间 33% |
| High disagreement | 后 33% |

#### 正式论文

建议用验证集确定阈值，然后应用到测试集。

步骤：

1. 在 validation set 上计算 \(D_{sample}\)；
2. 得到 1/3 和 2/3 分位数；
3. 用这两个阈值划分 test set。

这样更严谨。

---

### 4.7 比较模型

第一轮只需要：

| 模型 | 说明 |
|---|---|
| **Concat** | 普通拼接融合，无对齐 |
| **Unconditional Sample Alignment** | 所有 paired modalities 强制样本级对齐 |
| **Standard Class Prototype** | 普通类别原型对齐 |
| **CoPA-min** | 最小版 CoPA |

#### Unconditional Sample Alignment

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda
\sum_{i<j}D(z_i,z_j)
\]

其中：

\[
D(z_i,z_j)=1-\cos(z_i,z_j)
\]

建议尝试：

\[
\lambda\in\{0.01,0.05,0.1,0.5\}
\]

---

### 4.8 结果表模板

| Method | Low Dis. | Medium Dis. | High Dis. |
|---|---:|---:|---:|
| Concat |  |  |  |
| Uncond. Align |  |  |  |
| Standard Prototype |  |  |  |
| CoPA-min |  |  |  |

也可报告对齐收益：

\[
\Delta_{align}
=
\text{Perf}_{UncondAlign}
-
\text{Perf}_{Concat}
\]

| Group | \(\Delta_{align}\) |
|---|---:|
| Low disagreement | \(>0\) |
| Medium disagreement | \(\approx 0\) |
| High disagreement | \(<0\) |

理想现象：

- Low disagreement：强制对齐有收益；
- High disagreement：强制对齐有损害；
- CoPA 在 High disagreement 上不下降或提升。

---

## 5. 实验 2：High-D + Reliability 二次分组

### 5.1 实验目的

证明 high disagreement 中有两类：

| 类型 | 含义 |
|---|---|
| High-D + High-R | 可靠不一致，可能是互补信息 |
| High-D + Low-R | 不可靠不一致，可能是噪声 |

这一步用于证明：

> reliable disagreement 不是噪声，而是有潜在判别价值的互补信息。

---

### 5.2 分组方式

在 High disagreement 组内，根据 \(R_{sample}\) 中位数分成：

\[
HighD\_HighR
\]

\[
HighD\_LowR
\]

---

### 5.3 结果表模板

| Method | High-D + High-R | High-D + Low-R |
|---|---:|---:|
| Concat |  |  |
| Uncond. Align |  |  |
| Standard Prototype |  |  |
| CoPA |  |  |

预期：

- Unconditional alignment 对两者都可能不稳定；
- CoPA 在 High-D + High-R 上应更强；
- CoPA 在 High-D + Low-R 上应避免噪声污染。

---

## 6. 实验 3：主性能对比

### 6.1 数据集

标准配置：

```text
CMU-MOSI
CMU-MOSEI
CH-SIMS / CH-SIMS-v2
IEMOCAP
```

---

### 6.2 对比方法

#### 单模态与基础融合

| 方法 | 作用 |
|---|---|
| Text-only | 单模态上限/强模态参考 |
| Visual-only | 弱模态参考 |
| Audio-only | 弱模态参考 |
| Early Fusion | 基础融合 |
| Late Fusion | 基础融合 |
| Concat + MLP | 最基本多模态 baseline |

#### 经典多模态融合

| 方法 | 说明 |
|---|---|
| TFN | Tensor Fusion Network |
| LMF | Low-rank Multimodal Fusion |
| MulT | Multimodal Transformer |
| MISA | Modality-invariant / specific representation |
| Self-MM | Self-supervised multimodal sentiment |
| MMIM | Mutual information based multimodal learning |
| DMD | Distillation / decomposition based method |

#### 相关最新方法

| 方法 | 说明 |
|---|---|
| DecAlign | 解耦 + 分层对齐 |
| CaReFlow | 分布映射 + 信息保留 |
| UDML | 动态可靠性融合 |
| ARL | 不平衡多模态优化 |

---

### 6.3 指标

#### MOSI / MOSEI

| 指标 | 说明 |
|---|---|
| MAE | 越低越好 |
| Corr | 越高越好 |
| Acc-2 | 二分类准确率 |
| Acc-7 | 七分类准确率 |
| F1 | 二分类 F1 |

#### CH-SIMS / CH-SIMS-v2

| 指标 | 说明 |
|---|---|
| MAE | 越低越好 |
| Corr | 越高越好 |
| Acc-2 / Acc-3 / Acc-5 | 分类准确率 |
| F1 | 分类 F1 |

#### IEMOCAP

| 指标 | 说明 |
|---|---|
| WAcc | Weighted Accuracy |
| WAF1 | Weighted Average F1 |
| Acc | Accuracy |
| F1 | F1 |

---

### 6.4 主表模板

| Method | MOSI MAE ↓ | MOSI Corr ↑ | MOSI Acc-2 ↑ | MOSI F1 ↑ | MOSEI MAE ↓ | MOSEI Corr ↑ | MOSEI Acc-2 ↑ | MOSEI F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TFN |  |  |  |  |  |  |  |  |
| MulT |  |  |  |  |  |  |  |  |
| MISA |  |  |  |  |  |  |  |  |
| DecAlign |  |  |  |  |  |  |  |  |
| CoPA |  |  |  |  |  |  |  |  |

---

## 7. 实验 4：消融实验

### 7.1 消融版本

| 版本 | 去掉什么 | 目的 |
|---|---|---|
| Full CoPA | 完整模型 | 最终效果 |
| w/o Relation | 不区分 agreement / comp / noise | 验证关系判别必要性 |
| w/o Reliability | 去掉 \(R_m\)，只用 agreement | 验证可靠性估计必要性 |
| w/o Agreement Prototype | 去掉 \(P^{agr}\) | 验证可靠一致对齐必要性 |
| w/o Complementary Prototype | 去掉 \(P^{comp}\) | 验证互补差异原型必要性 |
| w/o Comp Separation | 去掉 \(\mathcal{L}_{comp}^{sep}\) | 验证差异不塌缩的重要性 |
| w/o Comp Proto Loss | 去掉 \(\mathcal{L}_{comp}^{proto}\) | 验证互补残差类别结构 |
| w/o Cross Agreement | 去掉 \(\mathcal{L}_{agr}^{cross}\) | 验证跨模态 agreement prototype 对齐 |
| Standard Prototype | 用普通类别原型替代关系原型 | 验证关系感知原型优越性 |
| Sample Alignment | 用样本对齐替代原型对齐 | 验证原型级对齐稳定性 |

---

### 7.2 表格模板

| Variant | Acc-2 ↑ | F1 ↑ | MAE ↓ | Corr ↑ |
|---|---:|---:|---:|---:|
| Full CoPA |  |  |  |  |
| w/o Relation |  |  |  |  |
| w/o Reliability |  |  |  |  |
| w/o \(P^{agr}\) |  |  |  |  |
| w/o \(P^{comp}\) |  |  |  |  |
| Standard Prototype |  |  |  |  |
| Sample Alignment |  |  |  |  |

建议在 MOSI 和 MOSEI 上都做，至少 MOSI 完整、MOSEI 关键消融。

---

## 8. 实验 5：原型层级对比

### 8.1 目的

证明关系感知原型优于：

- 样本点对点对齐；
- 普通类别原型；
- 只做 agreement prototype。

### 8.2 对比设置

| 方法 | 描述 |
|---|---|
| Sample Alignment | 所有 paired samples 直接拉近 |
| Standard Class Prototype | 每模态每类别一个原型 |
| Agreement Prototype only | 只使用 \(P^{agr}\) |
| Complementary Prototype only | 只使用 \(P^{comp}\) |
| Relation-Aware Dual Prototype | 使用 \(P^{agr}+P^{comp}\) |

### 8.3 结果表

| Prototype Strategy | Low Dis. | High Dis. | Overall |
|---|---:|---:|---:|
| Sample Alignment |  |  |  |
| Standard Class Prototype |  |  |  |
| Agreement-only |  |  |  |
| Dual Relation Prototype |  |  |  |

预期：

- Sample Alignment 对 high-disagreement 不稳定；
- Standard Prototype 稳定但可能平均掉差异；
- Dual Relation Prototype 最优。

---

## 9. 实验 6：关系分组性能分析

### 9.1 四类关系分组

根据 \(D_{sample}\) 和 \(R_{sample}\) 分成：

| 分组 | 条件 | 解释 |
|---|---|---|
| Reliable Agreement | Low-D + High-R | 可靠一致 |
| Reliable Disagreement | High-D + High-R | 可靠不一致 / 互补 |
| Unreliable Agreement | Low-D + Low-R | 看似一致但整体不可靠 |
| Unreliable Disagreement | High-D + Low-R | 不可靠不一致 / 噪声 |

---

### 9.2 分析内容

报告：

1. 每组样本比例；
2. 每组单模态性能；
3. 每组多模态性能；
4. 不同模块对每组的影响；
5. 每组的原型距离。

---

### 9.3 表格模板

| Group | Ratio | Concat | Uncond. Align | w/o Comp | CoPA |
|---|---:|---:|---:|---:|---:|
| Reliable Agreement |  |  |  |  |  |
| Reliable Disagreement |  |  |  |  |  |
| Unreliable Agreement |  |  |  |  |  |
| Unreliable Disagreement |  |  |  |  |  |

预期：

- Reliable Agreement：agreement prototype 帮助最大；
- Reliable Disagreement：complementary prototype 帮助最大；
- Unreliable groups：reliability 抑制机制更重要。

---

## 10. 实验 7：鲁棒性实验

### 10.1 噪声设置

#### 文本噪声

| 噪声 | 实现 |
|---|---|
| Token mask | 随机 mask 一定比例 token |
| Word dropout | 随机删除词 |
| Embedding dropout | 对文本特征加 dropout |

#### 视觉噪声

| 噪声 | 实现 |
|---|---|
| Gaussian noise | \(\tilde{x}=x+\epsilon\) |
| Frame dropout | 随机删除视觉帧 |
| Feature dropout | 随机置零部分视觉特征 |

#### 音频噪声

| 噪声 | 实现 |
|---|---|
| Gaussian noise | \(\tilde{x}=x+\epsilon\) |
| Time masking | 遮挡时间片段 |
| Feature dropout | 随机置零部分音频特征 |

---

### 10.2 噪声强度

\[
\sigma\in\{0.1,0.3,0.5,0.7,1.0\}
\]

或污染比例：

\[
r\in\{25\%,50\%,75\%\}
\]

---

### 10.3 结果表

| Method | Clean | Text Noise | Visual Noise | Audio Noise | All Noise |
|---|---:|---:|---:|---:|---:|
| Concat |  |  |  |  |  |
| UDML-style |  |  |  |  |  |
| Uncond. Align |  |  |  |  |  |
| CoPA |  |  |  |  |  |

预期：

- 噪声下 \(R_m\) 下降；
- 与该模态相关的 \(g^{agr},g^{comp}\) 下降；
- \(g^{noise}\) 上升；
- CoPA 不会让噪声样本污染原型。

---

## 11. 实验 8：模态缺失实验

### 11.1 设置

| 实验 | 操作 |
|---|---|
| w/o Text | 文本置零或 mask |
| w/o Visual | 视觉置零或 mask |
| w/o Audio | 音频置零或 mask |
| Random Missing | 随机丢弃一个模态 |
| Partial Missing | 一定比例样本缺失某模态 |

---

### 11.2 结果表

| Method | Full | w/o Text | w/o Visual | w/o Audio | Random Missing |
|---|---:|---:|---:|---:|---:|
| Concat |  |  |  |  |  |
| Dynamic Fusion |  |  |  |  |  |
| CoPA |  |  |  |  |  |

预期：

- 缺失模态可靠性下降；
- CoPA 自动降低该模态贡献；
- 原型更新不受缺失模态严重污染。

---

## 12. 实验 9：可视化

### 12.1 Disagreement motivation figure

横轴：

\[
D_{sample}
\]

纵轴：

\[
\Delta_{align}
=
\text{Perf}_{UncondAlign}
-
\text{Perf}_{Concat}
\]

预期曲线：

```text
Alignment Gain
   ^
   | positive for low disagreement
   |        *
   |      *
   |    *
   | 0 ----------------
   |             *
   |                *
   | negative for high disagreement
   +--------------------------> disagreement
```

这张图可以放在 Introduction 或 Motivation。

---

### 12.2 t-SNE / UMAP 原型可视化

展示：

1. 普通类别原型；
2. agreement prototype；
3. complementary prototype；
4. 不同模态特征分布。

预期：

- agreement prototypes 跨模态更接近；
- complementary prototypes 保持模态间差异；
- noisy samples 不应集中污染原型。

---

### 12.3 关系权重可视化

在不同噪声强度下画：

\[
g^{agr},g^{comp},g^{noise}
\]

预期：

- clean agreement 样本：\(g^{agr}\) 高；
- reliable disagreement 样本：\(g^{comp}\) 高；
- noisy 样本：\(g^{noise}\) 高。

---

### 12.4 案例分析

选择样本：

1. 三模态一致样本；
2. 文本-音频/视觉冲突样本；
3. 某模态噪声样本；
4. 讽刺/幽默样本。

展示：

- 单模态预测；
- \(R_m\)；
- \(A_{ij}\)；
- \(g^{agr},g^{comp},g^{noise}\)；
- 最终预测。

---

## 13. 训练与实现设置

### 13.1 特征

如果使用标准多模态情感处理流程：

| 模态 | 特征 |
|---|---|
| Text | BERT / GloVe |
| Visual | OpenFace |
| Audio | COVAREP |

### 13.2 优化器

| 参数 | 建议 |
|---|---|
| Optimizer | Adam / AdamW |
| Learning rate | \(1e^{-4}\sim5e^{-5}\) |
| Batch size | 32 / 64 |
| Epochs | 50 |
| Dropout | 0.1 - 0.5 |
| Weight decay | 0.001 - 0.01 |
| Seeds | 3 或 5 个随机种子 |

---

### 13.3 关键超参数

| 超参数 | 含义 | 建议范围 |
|---|---|---|
| \(\lambda_1\) | agreement loss 权重 | 0.01, 0.05, 0.1, 0.5 |
| \(\lambda_2\) | complementary loss 权重 | 0.01, 0.05, 0.1, 0.5 |
| \(\alpha\) | cross prototype alignment 权重 | 0.1, 0.5, 1.0 |
| \(\beta\) | complementary separation 权重 | 0.1, 0.5, 1.0 |
| \(\tau_A\) | agreement 计算温度 | 0.1, 0.5, 1.0 |
| \(\tau\) | prototype contrastive 温度 | 0.05, 0.1, 0.2 |
| \(\mu\) | EMA momentum | 0.9, 0.95, 0.99 |
| \(\delta\) | complementary margin | 0.2, 0.5, 1.0 |

---

### 13.4 训练阶段

#### Stage 1：Warm-up

训练：

\[
\mathcal{L}_{task}
+
\eta\mathcal{L}_{unimodal}
\]

持续：

\[
5\sim10\text{ epochs}
\]

#### Stage 2：原型初始化

用训练集初始化：

\[
P^{agr},P^{comp}
\]

#### Stage 3：联合训练

训练：

\[
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}
+
\lambda_2\mathcal{L}_{comp}
\]

---

## 14. 最小可行实验方案

如果先快速验证，按下面做：

### 14.1 数据集

```text
CMU-MOSI
```

### 14.2 模型

```text
Concat
Unconditional Alignment
CoPA-min
```

### 14.3 必做实验

1. 计算 disagreement；
2. Low / Mid / High 分组；
3. 对比 Concat 与 Unconditional Alignment；
4. 跑 CoPA-min；
5. 做一张 motivation 图。

### 14.4 CoPA-min 损失

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}^{proto}
+
\lambda_2\mathcal{L}_{comp}^{sep}
\]

暂时不加：

- \(\mathcal{L}_{agr}^{cross}\)；
- \(\mathcal{L}_{comp}^{proto}\)；
- relation-conditioned fusion；
- \(\mathcal{L}_{orth}\)。

---

## 15. 如果现象不符合预期怎么办？

### 情况 1：强制对齐在所有组都提升

说明：

- 数据 disagreement 不足；
- 对齐损失太弱；
- high-disagreement 不是可靠冲突；
- 当前任务确实适合统一对齐。

应尝试：

- 更强 disagreement 数据，如 MUStARD / UR-FUNNY；
- 更细标签划分；
- 改用单模态真实表现而非预测分布；
- 分析 high-D + high-R 子组。

---

### 情况 2：强制对齐在所有组都下降

说明：

- 对齐实现可能太粗糙；
- \(\lambda\) 太大；
- 表示未归一化；
- 单模态空间不匹配；
- 样本级 L2 拉近过强。

应尝试：

- 降低 \(\lambda\)；
- 使用 cosine distance；
- 加 warm-up；
- 改为 prototype alignment。

---

### 情况 3：disagreement 分组不稳定

说明：

- 单模态预测头太弱；
- early model 不可靠；
- 三分类过粗或过细；
- 测试集样本少。

应尝试：

- 增加 unimodal loss；
- 用 validation 阈值；
- 用多种 disagreement 指标；
- 多随机种子平均。

---

## 16. 实验最终叙事逻辑

论文实验部分应按如下逻辑展开：

1. **先证明问题存在**  
   无条件对齐对不同 disagreement 样本效果不同。

2. **再证明方法有效**  
   CoPA 在主数据集上整体性能提升。

3. **再证明设计合理**  
   双原型、关系判别、可靠性、互补保留各自有贡献。

4. **再证明不是偶然**  
   噪声、模态缺失、跨数据集场景下仍有效。

5. **最后做可解释分析**  
   展示 relation weights、prototype distributions、case study。

---

## 17. 最终实验路线图

```text
Step 1:
CMU-MOSI 跑 disagreement 诊断

Step 2:
CMU-MOSEI 复现 disagreement 诊断

Step 3:
实现 CoPA-min，与 Concat / UncondAlign / StandardPrototype 比较

Step 4:
加入完整 CoPA 模块，做主结果

Step 5:
做消融和关系分组分析

Step 6:
做噪声鲁棒性和模态缺失

Step 7:
补充 CH-SIMS / IEMOCAP

Step 8:
可选加入 CREMA-D / Kinetics-Sounds
```

---

## 18. 实验核心结论模板

如果结果符合预期，可以这样写：

> We first conduct a disagreement-based diagnostic experiment and observe that unconditional multimodal alignment improves low-disagreement samples but degrades high-disagreement samples. This indicates that paired multimodal samples are not always reliable alignment-positive pairs. Moreover, high-disagreement samples with high modality reliability still contain discriminative complementary information, which should not be simply suppressed as noise. Motivated by these observations, CoPA conditionally aligns agreement-positive samples, preserves complementary-positive samples, and reduces the influence of noisy-positive samples via relation-aware prototypes.

中文：

> 我们首先进行基于跨模态分歧的诊断实验，发现无条件多模态对齐在低分歧样本上能够提升性能，但在高分歧样本上反而降低性能。这说明同一样本内的多模态配对并不总是可靠的可对齐正样本。此外，高分歧但高可靠样本仍包含具有判别力的互补信息，不应被简单视为噪声抑制。基于此，本文提出 CoPA，通过关系感知类别原型对可靠一致样本进行条件对齐，对可靠不一致样本进行差异保留，并降低不可靠样本的影响。
