# CoPA version5：实验设计与执行路线

> 实验目标：验证 **relation-gated contrastive learning** 是否比 Concat、无条件对齐、DirectAdd 和普通 InfoNCE 更适合监督式多模态融合。  
> 核心问题：**预测分布只作为 relation gate 是否足够？InfoNCE 是否能真正学习 RA 的公共语义和 RD 的判别性残差？**

---

## 0. version5 实验总目标

v5 的方法从：

\[
\text{prediction disagreement} + \text{prototype separation}
\]

改为：

\[
\text{prediction-based relation gate} + \text{relation-gated contrastive learning}
\]

因此实验必须回答五个问题：

1. 无条件对齐、DirectAdd、普通 InfoNCE 是否仍然在 High-D / RD 上不稳定？
2. RA-gated common InfoNCE 是否比普通 same-sample InfoNCE 更稳？
3. RD residual InfoNCE 是否能证明 reliable disagreement 具有判别价值？
4. relation gate 是否真的有用，而不是随机分组也能达到同样效果？
5. CoPA-v5 是否在 overall 和 relation-state subgroup 上优于 v4 / baseline？

---

## 1. 总体实验路线

version5 实验分六个阶段：

| Phase | 目标 | 关键产物 |
|---|---|---|
| Phase 0 | 整理现有 20-seed motivation 证据 | 当前 v4 / prototype 版本的证据边界 |
| Phase 1 | 加入普通 InfoNCE baseline | 判断 same-sample InfoNCE 是否存在 High-D 风险 |
| Phase 2 | Residual probe | 验证 High-D+High-R residual 是否有判别价值 |
| Phase 3 | 实现 CoPA-v5 | RA-NCE + RD-NCE 主方法 |
| Phase 4 | Ablation & error control | 验证每个模块是否必要 |
| Phase 5 | 扩展数据集和鲁棒性 | MOSI/MOSEI 之外的补充验证 |

优先级：

\[
\text{Phase 1} \rightarrow \text{Phase 2} \rightarrow \text{Phase 3}
\]

也就是先确认 InfoNCE baseline 和 residual probe，再全面跑 CoPA-v5。

---

## 2. 数据集设置

### 2.1 主数据集

优先使用：

1. **CMU-MOSI**；
2. **CMU-MOSEI**。

原因：

- 是多模态情感分析主流 benchmark；
- 已有 Concat / UncondAlign / DirectAdd / CoPA 20-seed 结果；
- 便于和 v4 结果连续比较。

### 2.2 扩展数据集

建议后续使用：

| 数据集 | 用途 |
|---|---|
| CH-SIMS / CH-SIMS-v2 | 更适合分析模态级情感差异 |
| IEMOCAP | 情绪识别，多类别场景 |
| MUStARD | 讽刺检测，更容易出现 reliable disagreement |
| UR-FUNNY | 幽默检测，更容易验证反差 residual |

### 2.3 标签设置

对于 MOSI/MOSEI：

- 主指标可以使用二分类情感标签；
- 辅助可报告 7-class 或 regression 指标；
- relation gate 和 InfoNCE 第一版建议使用离散分类标签，减少复杂性。

二分类标签：

\[
y=\mathbb{1}(score>0)
\]

可选三分类：

\[
y\in\{negative,neutral,positive\}
\]

但第一版建议先使用二分类，保证稳定。

---

## 3. 基础模型与特征来源

### 3.1 特征来源

第一版建议继续使用：

- MultiBench / MMSA 已提取的文本、音频、视觉特征；
- 不重新训练大型编码器；
- 只训练轻量 encoder、projection head、fusion head、contrastive head。

### 3.2 模型输入

\[
x_t^n,x_v^n,x_a^n
\]

编码后：

\[
h_m^n=E_m(x_m^n),\quad m\in\{t,v,a\}
\]

再得到：

\[
z_m^{c,n}=P_m^c(h_m^n)
\]

\[
z_m^{r,n}=P_m^r(h_m^n)
\]

### 3.3 主任务融合

保留原始融合路径：

\[
z_{fusion}^n=F([h_t^n;h_v^n;h_a^n])
\]

\[
\hat y^n=C_{fusion}(z_{fusion}^n)
\]

主任务损失：

\[
\mathcal{L}_{task}=CE(\hat y^n,y^n)
\]

---

## 4. Relation gate 计算

### 4.1 单模态预测分布

每个模态有单模态预测头：

\[
p_m^n=\text{Softmax}(C_m(z_m^{c,n}))
\]

### 4.2 Prediction disagreement

\[
D_{ij}^n=JSD(p_i^n,p_j^n)
\]

样本级分歧：

\[
D_{pred}^n=rac{1}{3}(D_{ta}^n+D_{tv}^n+D_{av}^n)
\]

### 4.3 Agreement score

\[
A_{ij}^n=\exp(-D_{ij}^n/\tau_A)
\]

### 4.4 Reliability

诊断版本：

\[
Q_m^n=1-\frac{H(p_m^n)}{\log K}
\]

训练 label-aware 版本：

\[
S_m^n=p_m^n(y^n)
\]

\[
R_m^{label,n}=Q_m^nS_m^n
\]

### 4.5 RA gate

第一版训练建议：

\[
g_{ij}^{agr,n}=Q_i^nQ_j^nS_i^nS_j^nA_{ij}^n
\]

### 4.6 RD gate

推荐：

\[
B_{ij}^{label,n}=\max(S_i^n,S_j^n)
\]

\[
g_{ij}^{dis,n}=Q_i^nQ_j^nB_{ij}^{label,n}(1-A_{ij}^n)
\]

所有 gate 使用：

\[
\text{detach}(g)
\]

---

## 5. Phase 0：整理现有 20-seed 证据

### 5.1 目的

把现有结果作为 motivation evidence，而不是最终方法证据。

### 5.2 需要整理的表

#### Table 1：Overall performance

| Method | Macro-F1 | Acc | Delta vs Concat | 95% CI | Error Control |
|---|---:|---:|---:|---:|---|
| Concat |  |  |  |  |  |
| UncondAlign |  |  |  |  |  |
| DirectAdd |  |  |  |  |  |
| CoPA-v4 |  |  |  |  |  |

#### Table 2：Low/Mid/High-D group delta

| Method | Low-D | Mid-D | High-D | High-D Error Control |
|---|---:|---:|---:|---|
| UncondAlign |  |  |  |  |
| DirectAdd |  |  |  |  |
| CoPA-v4 |  |  |  |  |

#### Table 3：Fixed strength analysis

| Method | Strength | Group | Delta | 95% CI | Positive Rate | Error Control |
|---|---:|---|---:|---:|---:|---|
| UncondAlign | \(\lambda\) | High-D |  |  |  |  |
| DirectAdd | \(\alpha\) | High-D |  |  |  |  |

### 5.3 预期结论

只能保守写：

> Alignment gain is relation-dependent. Unconditional alignment and direct addition are not uniformly reliable, especially in high-disagreement groups.

不要写：

> CoPA-v4 has solved reliable disagreement.

因为当前 RD 证据还不充分。

---

## 6. Phase 1：普通 InfoNCE baseline

### 6.1 目的

验证：

> 如果使用普通 same-sample InfoNCE，会不会重复无条件对齐的问题？

### 6.2 Baseline 定义

对模态对 \((i,j)\)：

\[
\ell_{i\rightarrow j}^{NCE,n}
=
-\log
\frac{
\exp(\text{sim}(z_i^n,z_j^n)/\tau)
}{
\sum_{k=1}^{B}\exp(\text{sim}(z_i^n,z_j^k)/\tau)
}
\]

双向：

\[
\mathcal{L}_{ij}^{NCE}
=\frac{1}{2}(\mathcal{L}_{i\rightarrow j}^{NCE}+\mathcal{L}_{j\rightarrow i}^{NCE})
\]

三模态：

\[
\mathcal{L}_{uncond}^{NCE}
=\frac{1}{3}(\mathcal{L}_{ta}^{NCE}+\mathcal{L}_{tv}^{NCE}+\mathcal{L}_{av}^{NCE})
\]

训练：

\[
\mathcal{L}=\mathcal{L}_{task}+\lambda_{nce}\mathcal{L}_{uncond}^{NCE}
\]

### 6.3 对照方法

| 方法 | 作用 |
|---|---|
| Concat | 基础监督融合 |
| UncondAlign L2/cosine | 无条件距离拉近 |
| DirectAdd | 保留原路径 + 直接混入对齐摘要 |
| Uncond InfoNCE | same-sample positive 对比学习 |

### 6.4 超参数

\[
\lambda_{nce}\in\{0.01,0.05,0.1,0.5\}
\]

\[
\tau\in\{0.05,0.1,0.2,0.5\}
\]

第一版固定：

\[
\tau=0.1
\]

主要搜索：

\[
\lambda_{nce}
\]

### 6.5 成功标准

如果普通 InfoNCE 在 High-D 或 RD 上不稳定，说明：

\[
\text{same-sample positive} \text{ 仍然过强}
\]

如果普通 InfoNCE overall 提升但 High-D 不稳定，也支持 CoPA-v5 的 relation-gated 设计。

### 6.6 失败情况

如果普通 InfoNCE 在所有组都稳定提升，则不能说 InfoNCE 有害，只能说：

> relation-gated InfoNCE further improves stability and subgroup robustness.

---

## 7. Phase 2：Residual probe

### 7.1 目的

证明或证伪：

\[
High\text{-}D+High\text{-}R
\]

中的 residual 是否具有判别价值。

这是决定论文能否继续主打 “discriminative disagreement” 的关键。

### 7.2 分组方式

先按 validation 阈值划分 Low/Mid/High-D：

\[
D_{pred}^n
\]

再在 High-D 内部按 reliability 中位数划分：

\[
R_{sample}^n=\frac{1}{3}(Q_t^n+Q_v^n+Q_a^n)
\]

得到：

- High-D + High-R；
- High-D + Low-R。

### 7.3 Probe 特征

公共特征：

\[
z_{common}^n=[z_t^{c,n};z_v^{c,n};z_a^{c,n}]
\]

残差特征：

\[
r_{ta}^n=|z_t^{r,n}-z_a^{r,n}|
\]

\[
r_{tv}^n=|z_t^{r,n}-z_v^{r,n}|
\]

\[
r_{av}^n=|z_a^{r,n}-z_v^{r,n}|
\]

组合残差：

\[
z_{res}^n=[r_{ta}^n;r_{tv}^n;r_{av}^n]
\]

### 7.4 Probe 模型

使用轻量模型，避免过拟合：

- Logistic Regression；
- Linear SVM；
- 1-layer MLP。

对比：

| Probe | 输入 | 目的 |
|---|---|---|
| Common-only | \(z_{common}\) | 公共语义是否足够 |
| Residual-only | \(z_{res}\) | residual 是否有判别力 |
| Common+Residual | \([z_{common};z_{res}]\) | residual 是否提供增益 |
| Shuffled residual | 打乱 residual-label | 负对照 |
| Random group residual | 随机 High-R / Low-R | 检查分组是否有效 |

### 7.5 成功标准

满足任一即可支撑 RD：

1. High-D+High-R 中 residual-only 明显高于 shuffled residual；
2. common+residual 明显优于 common-only；
3. High-D+High-R residual 增益大于 High-D+Low-R；
4. RD residual 的类内聚合 / 类间分离更明显。

### 7.6 如果失败

如果 residual probe 失败，论文主张应收缩：

从：

> Selective Agreement and Discriminative Disagreement

收缩为：

> Relation-aware Selective Alignment

RD 只作为“不应强制对齐”的证据，不强调“可学习判别残差”。

---

## 8. Phase 3：实现 CoPA-v5

### 8.1 主方法

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

### 8.2 RA-NCE

对 RA 权重高的样本：

\[
\mathcal{L}_{agr}^{NCE}
\]

作用：

> 只对可靠一致样本做 common representation alignment。

### 8.3 RD-NCE

对 RD 权重高的样本：

\[
\mathcal{L}_{dis}^{NCE}
\]

作用：

> 不拉近模态原特征，而是学习 residual representation 的类别结构。

### 8.4 第一版实现建议

为了降低复杂度，第一版使用：

| 模块 | 第一版选择 |
|---|---|
| RA loss | Gated common InfoNCE |
| RD loss | Prototype residual NCE |
| Gate | label-aware gate |
| Prototype update | EMA |
| Gate gradient | detach |
| Negative set | 不同类别样本作为主要负样本 |

### 8.5 训练阶段

#### Stage 1：Warm-up

\[
\mathcal{L}_{task}+\lambda_u\mathcal{L}_{uni}
\]

Epoch：

\[
5\sim 10
\]

#### Stage 2：Prototype initialization

使用 warm-up 后训练集特征初始化：

\[
P_c^{dis}
\]

可选：

\[
P_c^{agr}
\]

#### Stage 3：Joint training

启用完整损失，并使用 ramp-up：

\[
\lambda_a(t),\lambda_d(t):0\rightarrow \lambda^{max}
\]

---

## 9. Phase 4：消融实验

### 9.1 必做消融

| 方法 | 目的 |
|---|---|
| Concat | 基线 |
| UncondAlign | 无条件距离对齐 |
| DirectAdd | 直接相加对齐摘要 |
| Uncond InfoNCE | 普通 same-sample NCE |
| RA-NCE only | 只验证 reliable agreement 对齐 |
| RD-NCE only | 只验证 residual disagreement learning |
| CoPA-v5 full | RA-NCE + RD-NCE |
| CoPA-v5 w/o label-aware gate | 验证 label-aware reliability |
| CoPA-v5 w/o detach | 验证 stop-gradient gate |
| CoPA-v5 shuffled gate | 验证 relation gate 非随机 |
| CoPA-v5 shuffled residual-label | 验证 residual 类别结构 |

### 9.2 推荐表格

#### Table A：Overall results

| Method | Macro-F1 | Acc | Delta | 95% CI | Positive Rate | Error Control |
|---|---:|---:|---:|---:|---:|---|
| Concat |  |  |  |  |  |  |
| UncondAlign |  |  |  |  |  |  |
| DirectAdd |  |  |  |  |  |  |
| Uncond InfoNCE |  |  |  |  |  |  |
| CoPA-v5 |  |  |  |  |  |  |

#### Table B：Disagreement group results

| Method | Low-D Delta | Mid-D Delta | High-D Delta | High-D Error Control |
|---|---:|---:|---:|---|
| UncondAlign |  |  |  |  |
| DirectAdd |  |  |  |  |
| Uncond InfoNCE |  |  |  |  |
| CoPA-v5 |  |  |  |  |

#### Table C：Relation-state results

| Method | RA | UA | RD | ND | RD Error Control |
|---|---:|---:|---:|---:|---|
| Uncond InfoNCE |  |  |  |  |  |
| RA-NCE only |  |  |  |  |  |
| RD-NCE only |  |  |  |  |  |
| CoPA-v5 |  |  |  |  |  |

#### Table D：Residual probe

| Group | Common-only | Residual-only | Common+Residual | Shuffled residual |
|---|---:|---:|---:|---:|
| High-D+High-R |  |  |  |  |
| High-D+Low-R |  |  |  |  |
| Low-D+High-R |  |  |  |  |

---

## 10. Phase 5：扩展验证

### 10.1 数据集扩展

优先顺序：

1. MOSI；
2. MOSEI；
3. CH-SIMS / CH-SIMS-v2；
4. MUStARD；
5. UR-FUNNY；
6. IEMOCAP。

### 10.2 为什么需要讽刺 / 幽默数据集

MOSI/MOSEI 的 reliable disagreement 可能不够强。讽刺和幽默任务天然包含：

- 文本表面语义和语气反差；
- 视觉表情和语言内容反差；
- 音频情绪和文本语义反差。

因此更适合验证：

\[
RD \Rightarrow \text{discriminative residual}
\]

### 10.3 噪声 / 缺失鲁棒性

可选实验：

- audio noise；
- visual masking；
- text word dropout；
- modality missing。

但注意：

> 噪声鲁棒性不是 CoPA-v5 主贡献，只是边界条件。

---

## 11. 指标与误差控制

### 11.1 主指标

分类：

- Macro-F1；
- Accuracy；
- Weighted-F1；
- subgroup Macro-F1。

回归：

- MAE；
- Corr；
- Acc-2；
- F1。

第一版主指标建议：

\[
\text{Macro-F1}
\]

### 11.2 多 seed

正式结果建议：

\[
20\ \text{seeds}
\]

初筛可以用：

\[
5\ \text{seeds}
\]

### 11.3 Error control

沿用当前规则：

1. \(min\_seeds\geq 5\)；
2. positive rate \(\geq 0.8\)；
3. 95% CI 不跨 0。

即：

\[
\text{Error Control}=True
\]

当且仅当：

\[
CI_{low}>0
\]

且：

\[
\text{positive rate}\geq 80\%
\]

---

## 12. 超参数设置

### 12.1 Gate 参数

| 参数 | 候选 |
|---|---|
| \(\tau_A\) | 0.1, 0.3, 0.5, 1.0 |
| reliability type | entropy-only, label-aware |
| gate detach | True / False |

第一版：

\[
\tau_A=0.5
\]

\[
\text{label-aware gate + detach}
\]

### 12.2 InfoNCE 参数

| 参数 | 候选 |
|---|---|
| \(\tau_c\) | 0.05, 0.1, 0.2 |
| \(\tau_r\) | 0.05, 0.1, 0.2 |
| \(\lambda_a\) | 0.01, 0.05, 0.1, 0.2 |
| \(\lambda_d\) | 0.01, 0.05, 0.1, 0.2 |
| EMA \(\mu\) | 0.9, 0.95, 0.99 |

第一轮建议：

\[
\tau_c=0.1,\quad \tau_r=0.1
\]

\[
\lambda_a,\lambda_d\in\{0.01,0.05,0.1\}
\]

\[
\mu=0.95
\]

### 12.3 Batch size

InfoNCE 对 batch size 敏感。

建议：

\[
B\geq 128
\]

如果显存允许：

\[
B=512\text{ or }1024
\]

如果 batch 太小，优先使用 prototype NCE。

---

## 13. 代码实现建议

### 13.1 新增文件结构

建议新增：

```text
code/
  copa_v5/
    models/
      copa_nce_model.py
      projection_heads.py
      residual_heads.py
    losses/
      gated_infonce.py
      residual_nce.py
      prototype_nce.py
    utils/
      relation_gate.py
      prototype_memory.py
      subgroup_eval.py
    scripts/
      run_copa_v5.py
      run_infonce_baseline.py
      run_residual_probe.py
```

### 13.2 relation_gate.py

需要实现：

- entropy reliability；
- label-support reliability；
- JSD；
- agreement score；
- \(g^{agr}\)；
- \(g^{dis}\)；
- detach option。

### 13.3 gated_infonce.py

需要实现：

- cross-modal bidirectional InfoNCE；
- gate-weighted loss；
- supervised negative mask；
- same-class false negative removal。

### 13.4 residual_nce.py

需要实现：

- residual construction；
- residual SupCon；
- residual prototype NCE；
- shuffled residual negative control。

### 13.5 prototype_memory.py

需要实现：

- class prototype 初始化；
- EMA 更新；
- gate-weighted update；
- empty class protection。

---

## 14. 预期结果解释

### 14.1 最理想结果

| 现象 | 解释 |
|---|---|
| CoPA-v5 overall > Concat / UncondInfoNCE | 方法有效 |
| CoPA-v5 High-D > UncondInfoNCE | relation gate 有价值 |
| RD-NCE 在 High-D+High-R 有提升 | reliable disagreement 有判别价值 |
| shuffled gate 下降 | gate 不是随机有效 |
| shuffled residual-label 下降 | residual 类别结构真实存在 |

### 14.2 中等结果

如果 CoPA-v5 只提升 overall，但 RD 不稳定：

论文收缩为：

> relation-gated contrastive regularization

不要强说：

> reliable disagreement 被稳定利用。

### 14.3 不理想结果

如果普通 InfoNCE 和 CoPA-v5 差不多：

需要检查：

- relation gate 是否过于稀疏；
- RA/RD 样本比例是否失衡；
- residual 表示是否太弱；
- batch 内正样本是否不足；
- \(\lambda_d\) 是否过大或过小；
- RD 是否在 MOSI/MOSEI 中本来就不明显。

可转向 MUStARD / UR-FUNNY 验证。

---

## 15. 写作时的结论边界

### 15.1 可以写

> Prediction disagreement is used as a relation signal rather than a direct semantic target.

> CoPA uses relation gates to determine whether paired modalities should be aligned or modeled through residual contrastive learning.

> Reliable agreement is treated as alignment-positive, while reliable disagreement is treated as residual-positive.

### 15.2 不建议写

> We directly use prediction disagreement as discriminative information.

> We estimate mutual information between modalities using JSD.

> All high-disagreement samples are useful.

> All same-sample modality pairs should be aligned by InfoNCE.

### 15.3 关于互信息的写法

如果使用 InfoNCE，可以谨慎写：

> InfoNCE can be interpreted as maximizing a lower bound of mutual information under certain assumptions, but in CoPA it is mainly used as a relation-gated contrastive representation learning objective.

中文：

> InfoNCE 在一定假设下可以被解释为互信息下界最大化，但本文主要将其作为关系门控的对比表示学习目标，而不是直接用来估计跨模态互信息。

---

## 16. 最小可执行实验清单

第一轮最小实验只做 MOSI / MOSEI：

1. Concat；
2. UncondAlign；
3. DirectAdd；
4. Uncond InfoNCE；
5. RA-NCE only；
6. RD-prototype-NCE only；
7. CoPA-v5 full；
8. shuffled gate；
9. residual probe。

每个先跑：

\[
5\ \text{seeds}
\]

筛掉明显不行的配置后，再跑：

\[
20\ \text{seeds}
\]

---

## 17. 最终实验主线

论文实验顺序建议：

1. **Motivation Analysis**  
   证明无条件对齐、DirectAdd、普通 InfoNCE 的收益依赖 relation state。

2. **Residual Probe**  
   证明 High-D+High-R residual 可能具有判别价值。

3. **Main Results**  
   CoPA-v5 vs Concat / UncondAlign / DirectAdd / UncondInfoNCE。

4. **Ablation Study**  
   RA-NCE、RD-NCE、label-aware gate、detach、prototype NCE。

5. **Subgroup Analysis**  
   Low/Mid/High-D、RA/UA/RD/ND、High-D+High-R。

6. **Case Study**  
   展示文本-音频-视觉反差样本。

7. **Robustness / Extension**  
   噪声、缺失、讽刺/幽默数据集。

---

## 18. 一句话总结

> v5 实验的关键不是证明“预测不一致一定有用”，而是证明：预测分布可以作为 relation gate，帮助 InfoNCE 区分哪些 same-sample pairs 应该对齐，哪些 reliable disagreement pairs 应该学习残差结构；如果 residual probe 和 RD-NCE 成功，CoPA 就可以稳健地主张 selective agreement and discriminative disagreement learning。
