# CoPA version3：改进版实验与执行路线

> 实验目标：先把动机坐实，再实现 CoPA-min，最后决定是否扩展到 Full CoPA。
>
> 核心原则：每一步都要有明确成败标准；如果证据不支持强主张，就及时收缩论文叙事。

---

## 1. 总体路线

version3 实验分五个阶段：

| Phase | 目标 | 关键产物 |
|---|---|---|
| Phase 1 | 整理现有 motivation 诊断 | MOSI/MOSEI 分组 delta 表和结论边界 |
| Phase 2 | 实现并评估 CoPA-min | CoPA-min vs baseline 主结果 |
| Phase 3 | 验证 reliable disagreement | High-D + High-R 专门消融 |
| Phase 4 | 回到原始样本做 case study | 可解释样本证据 |
| Phase 5 | 鲁棒性与扩展数据集 | 噪声/缺失/跨数据集验证 |

实验优先级必须服务论文主线：

1. 先证明 alignment gain 与 relation state 有关；
2. 再证明 High-D + High-R 中存在 reliable complementary disagreement；
3. 最后才做噪声、缺失和弱对齐鲁棒性。

因此，Phase 5 不能反过来主导论文叙事。噪声处理只是 conditional alignment 的边界条件，不是 version3 的主贡献。

---

## 2. Phase 1：整理现有 motivation 诊断

### 2.1 目的

证明一个稳健命题：

> 无条件跨模态对齐的收益依赖 cross-modal relation state。

暂不强行证明：

> High-disagreement 样本一定被无条件对齐伤害。

### 2.2 已有结果整理

当前已有 5 seeds 诊断结果：

| Dataset | Low-D delta Macro-F1 | Mid-D delta Macro-F1 | High-D delta Macro-F1 | 初步结论 |
|---|---:|---:|---:|---|
| MOSI | +0.0468 | +0.0145 | +0.0213 | High-D 收益弱于 Low-D，但未变负 |
| MOSEI | +0.0129 | +0.0033 | -0.0006 | High-D 出现轻微负收益 |

写论文时应表述为：

> Alignment gain is relation-dependent. In high-disagreement groups, unconditional alignment becomes less reliable and can even degrade performance on MOSEI.

### 2.3 需要补做的严谨化

正式实验建议使用 validation set 决定 disagreement 阈值：

1. 在 validation set 上计算 \(D_{sample}\)；
2. 取 1/3 和 2/3 分位数作为 Low/Mid/High 阈值；
3. 将阈值应用到 test set；
4. 报告每组样本数、Acc-2、Macro-F1、delta。

Motivation 分组只使用预测熵和跨模态 JSD，不使用 test label：

\[
R_m^{diag}=1-\frac{H(p_m)}{\log K}
\]

这样可以避免 test label leakage。这里的 \(R_m^{diag}\) 只表示预测确定性，不等价于训练原型时的可靠性。

### 2.4 成功标准

满足任一条件即可支撑 version3 动机：

- High-D 的 alignment gain 明显弱于 Low-D；
- High-D 的方差明显更大；
- High-D 在 MOSEI 或其他数据集上出现负收益；
- 不同 relation group 的 alignment gain 呈现系统性差异。

### 2.5 失败后的调整

如果无条件对齐在所有组都稳定提升，则不能继续主张“条件正样本对齐必要”。应调整为：

- 换更能体现跨模态冲突的数据集，例如讽刺/幽默；
- 或把论文改成 prototype-level regularization，而不是 conditional alignment。

---

## 3. Phase 2：实现并评估 CoPA-min

### 3.1 目的

验证一个最小方法：

> 根据 relation state 选择性使用 agreement alignment 和 complementary preservation，是否优于无条件对齐和普通原型。

### 3.2 模型版本

必须比较以下四类：

| 方法 | 作用 |
|---|---|
| Concat | 基础融合 baseline |
| Unconditional Sample Alignment | 验证无条件样本对齐 |
| Standard Class Prototype | 验证普通类别原型 |
| CoPA-min | 本文核心方法 |

### 3.3 CoPA-min 保留模块

只实现以下内容：

- 单模态预测头；
- diagnostic reliability \(R_m^{diag}\)；
- label-aware reliability \(R_m^{label}\)；
- agreement \(A_{ij}\)；
- relation weights \(g^{agr}, g^{comp}, g^{noise}\)；
- EMA agreement prototype；
- EMA complementary prototype；
- \(\mathcal{L}_{agr}^{proto}\)；
- \(\mathcal{L}_{comp}^{sep}\)；
- 主任务损失 \(\mathcal{L}_{task}\)。

暂不实现：

- relation-conditioned fusion；
- orthogonal loss；
- uniformity loss；
- full cross-prototype alignment；
- full complementary prototype contrastive loss。

### 3.4 训练流程

推荐三阶段：

1. **Warm-up**
   - 训练主任务头和单模态辅助头；
   - 不启用 prototype loss；
   - 目标是让 \(p_m\)、\(R_m\)、\(A_{ij}\) 不完全随机。

2. **Prototype initialization**
   - 使用 warm-up 后的训练集特征；
   - 使用 label-aware reliability：

\[
R_m^{label}=
\left(1-\frac{H(p_m)}{\log K}\right)
\cdot
p_m(y)
\]

   - 按 \(g^{agr}\) 初始化 \(P^{agr}\)；
   - 按 \(g^{comp}\) 初始化 \(P^{comp}\)；
   - 不使用仅基于 entropy 的 reliability 更新训练原型。

3. **Joint training**
   - 每个 batch 计算 relation weights；
   - 训练中的 relation weights 使用 \(R_m^{label}\)；
   - relation weights 使用 detach；
   - 用 EMA 更新 prototype buffer；
   - 优化总损失：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_1\mathcal{L}_{agr}^{proto}
+
\lambda_2\mathcal{L}_{comp}^{sep}
\]

### 3.5 超参数

第一轮搜索：

| 参数 | 候选 |
|---|---|
| \(\lambda_1\) | 0.01, 0.05, 0.1, 0.5 |
| \(\lambda_2\) | 0.01, 0.05, 0.1, 0.5 |
| \(\tau_A\) | 0.1, 0.5, 1.0 |
| \(\mu\) | 0.9, 0.95, 0.99 |
| \(\delta\) | 0.2, 0.5, 1.0 |

第一轮可以先固定 \(\tau_A=0.5\)、\(\mu=0.95\)、\(\delta=0.5\)，主要搜索 \(\lambda_1,\lambda_2\)。

### 3.6 成功标准

CoPA-min 至少需要满足：

- Overall 不弱于 Concat 和 Unconditional Alignment；
- High-D 或 High-D+High-R 上优于 Unconditional Alignment；
- Standard Prototype 不足以替代 Relation-Aware Prototype；
- label-aware reliability 优于或至少不弱于 entropy-only reliability；
- 5 seeds mean/std 稳定。

### 3.7 失败后的调整

如果 CoPA-min 只提升 Overall，但 High-D 不提升：

- 将论文主张改为 relation-aware regularization；
- 弱化 complementary-positive；
- 保留 selective alignment 作为核心。

如果 CoPA-min 全面不如 baseline：

- 先检查 relation weights 是否合理；
- 对比 entropy-only reliability 和 label-aware reliability；
- 检查高置信错误样本是否污染 agreement prototype；
- 检查 prototype 是否被早期错误关系污染；
- 增加 warm-up；
- 降低 \(\lambda_1,\lambda_2\)；
- 暂停 Full CoPA 扩展。

---

## 4. Phase 3：High-D + High-R 专门消融

### 4.1 目的

证明或证伪：

> High-disagreement + high-reliability 样本中存在有判别价值的 complementary information。

这是防止论文被认为“硬凑”的关键实验。

### 4.2 分组方式

在 High-D 样本内部，根据 \(R_{sample}\) 中位数划分：

- High-D + High-R；
- High-D + Low-R。

其中：

\[
R_{sample}=\frac{1}{M}\sum_m R_m
\]

### 4.3 专门消融

必须比较：

| 版本 | 用途 |
|---|---|
| common only | 只使用公共语义 |
| residual only | 检查残差信息是否有判别力 |
| common + residual | 检查残差是否补充公共语义 |
| w/o comp prototype | 检查互补原型是否必要 |
| w/o comp separation | 检查差异保持是否必要 |
| CoPA-min | 完整最小方法 |

### 4.4 成功标准

支持 complementary-positive 的条件：

- High-D + High-R 上 residual branch 有正贡献；
- w/o comp prototype 或 w/o comp separation 明显下降；
- CoPA-min 在 High-D + High-R 上优于 Unconditional Alignment；
- case study 中能看到合理的跨模态反差。

### 4.5 失败后的调整

如果 residual branch 无贡献：

- 不再把 reliable disagreement 强写成互补信息；
- 将 \(g^{comp}\) 改写为避免错误对齐的保护机制；
- 方法贡献收缩为 conditional selective alignment。

---

## 5. Phase 4：原始样本 case study

### 5.1 目的

补上模型分布证据和真实多模态现象之间的桥梁。

仅使用 JSD 和 entropy 会被质疑是模型内部定义，因此需要展示真实样本：

- 原始文本；
- 视频表情或视觉摘要；
- 音频语气或音频特征描述；
- 单模态预测；
- reliability；
- relation weights；
- 最终预测变化。

### 5.2 样本类型

至少选 5 到 10 个样本，覆盖：

| 类型 | 目标 |
|---|---|
| Low-D + High-R | 展示可靠一致，适合对齐 |
| High-D + High-R | 展示可靠不一致，可能互补 |
| High-D + Low-R | 展示噪声或弱模态 |
| Unconditional Alignment 失败样本 | 展示错误对齐问题 |
| CoPA-min 修正样本 | 展示条件对齐收益 |

### 5.3 展示模板

每个 case 建议包含：

| 字段 | 内容 |
|---|---|
| Text | 原始文本或转写 |
| Label | 真实标签 |
| Predictions | text / visual / audio / fusion |
| Reliability | \(R_t,R_v,R_a\) |
| Agreement | \(A_{tv},A_{ta},A_{va}\) |
| Relation weights | \(g^{agr},g^{comp},g^{noise}\) |
| Error analysis | 为什么无条件对齐错，CoPA-min 为什么更合理 |

### 5.4 成功标准

case study 不需要证明统计显著性，但必须说明：

- High-D + High-R 确实包含可解释的跨模态反差；
- High-D + Low-R 更像噪声或弱模态；
- CoPA-min 的 relation weights 与人类直觉大体一致。

---

## 6. Phase 5：鲁棒性与扩展数据集

Phase 5 的定位是补充验证，不是主实验。它回答的是：

> 当模态退化、缺失或弱对应时，CoPA-min 是否能避免不可靠配对污染对齐和原型？

它不负责证明 complementary-positive。真正证明 complementary-positive 的位置仍然是 Phase 3 和 Phase 4。

### 6.1 鲁棒性实验

优先做三类：

| 模态 | 噪声方式 |
|---|---|
| Text | token mask / word dropout / embedding dropout |
| Visual | frame dropout / feature dropout / Gaussian noise |
| Audio | time masking / feature dropout / Gaussian noise |

如果第一阶段只用预提取特征，可以先做 feature dropout 和 Gaussian noise。后续再回到原始模态做更真实的噪声。

### 6.2 模态缺失实验

设置：

- w/o Text；
- w/o Visual；
- w/o Audio；
- Random Missing；
- Partial Missing。

观察：

- 缺失模态的 reliability 是否下降；
- 相关 \(g^{agr}\)、\(g^{comp}\) 是否下降；
- \(g^{noise}\) 是否上升；
- CoPA-min 是否比无条件对齐更稳。

### 6.3 扩展数据集

优先级：

1. MOSI；
2. MOSEI；
3. CH-SIMS 或 CH-SIMS-v2；
4. IEMOCAP；
5. MUStARD / UR-FUNNY。

如果目标是验证 reliable disagreement，MUStARD 和 UR-FUNNY 比普通情感数据更适合，但实现成本和数据处理成本更高。

---

## 7. 最终论文表格安排

### 7.1 Motivation table

| Method | Low-D | Mid-D | High-D | Overall |
|---|---:|---:|---:|---:|
| Concat |  |  |  |  |
| Unconditional Alignment |  |  |  |  |
| Delta |  |  |  |  |

### 7.2 Main comparison

| Method | MOSI Acc-2 | MOSI F1 | MOSEI Acc-2 | MOSEI F1 |
|---|---:|---:|---:|---:|
| Concat |  |  |  |  |
| Uncond. Align |  |  |  |  |
| Standard Prototype |  |  |  |  |
| CoPA-min |  |  |  |  |

### 7.3 Relation group analysis

| Method | Low-D+High-R | High-D+High-R | High-D+Low-R |
|---|---:|---:|---:|
| Concat |  |  |  |
| Uncond. Align |  |  |  |
| w/o comp |  |  |  |
| CoPA-min |  |  |  |

### 7.4 Ablation

| Variant | Acc-2 | Macro-F1 | High-D F1 | High-D+High-R F1 |
|---|---:|---:|---:|---:|
| CoPA-min |  |  |  |  |
| w/o reliability |  |  |  |  |
| w/o agreement prototype |  |  |  |  |
| w/o complementary prototype |  |  |  |  |
| w/o comp separation |  |  |  |  |
| Standard Prototype |  |  |  |  |

---

## 8. 最小可执行清单

建议下一轮实际执行顺序：

1. 把现有 MOSI/MOSEI motivation 结果整理成统一 CSV 和图；
2. 修改分组逻辑为 validation threshold；
3. 实现 Standard Prototype baseline；
4. 实现 CoPA-min；
5. 跑 MOSI 5 seeds；
6. 跑 MOSEI 5 seeds；
7. 生成 High-D + High-R 消融；
8. 挑选 case study；
9. 决定是否加入 Full CoPA 增强项。

---

## 9. 版本推进门槛

### 可以继续推进 Full CoPA 的条件

满足以下多数条件：

- CoPA-min Overall 优于 baseline；
- High-D 或 High-D+High-R 上 CoPA-min 更稳；
- complementary branch 有明确消融贡献；
- case study 能解释 reliable disagreement；
- relation weights 在噪声/缺失下表现合理。

### 不应继续堆模块的条件

出现以下任一情况，应暂停 Full CoPA：

- CoPA-min 无法优于 Standard Prototype；
- High-D + High-R 上 residual branch 没有贡献；
- relation weights 与直觉明显不符；
- 加模块只提升 Overall，无法解释 relation group。

---

## 10. version3 实验一句话总结

> version3 的实验不是为了立刻证明一个复杂 Full CoPA，而是先证明“对齐收益依赖关系状态”，再用 CoPA-min 验证选择性对齐和互补差异保留是否真的必要；如果证据不支持强主张，就及时把论文收缩为 relation-aware alignment regularization。
