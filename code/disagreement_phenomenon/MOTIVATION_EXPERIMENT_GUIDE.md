# CoPA 动机实验详细说明

这份文档解释 `code/disagreement_phenomenon` 当前动机实验的完整流程、术语、表格读法和常见误解。它的目标不是介绍完整 CoPA-v5 方法，而是把目前这套“动机证据闭环”讲清楚：

```text
同一个多模态样本不一定天然适合无条件对齐。
普通 same-sample alignment / InfoNCE 的收益可能依赖样本内部的关系状态。
```

这里的“关系状态”不是人工标注的额外标签，而是由参考诊断模型的单模态预测分布、跨模态分歧和预测置信度构造出来的分析分组。

## 1. 这套实验到底想证明什么

### 1.1 主命题

多模态情感任务里，一个样本通常包含 text、audio、vision 三个模态。很多 alignment 方法默认认为：

```text
同一个样本里的 text/audio/vision 是正样本对，应该被拉近。
```

这叫 same-sample positive assumption。动机实验要检验的是：这个假设是否总是合理。

更具体地说，当前实验不直接证明“CoPA-v5 主方法一定最好”，而是先证明一个更小、更稳的动机：

```text
普通无条件对齐的收益不是全局稳定的；
它在 Low-D、High-D、RA、RD 等关系状态上可能表现不同。
```

如果这个现象成立，后续才有理由设计 relation-aware / conditional alignment 方法。否则，论文直接上复杂 CoPA 主方法会显得动机不稳。

### 1.2 不主张什么

当前动机实验不主张：

```text
Concat 学不到互补信息。
```

Concat 是监督分类器，它当然可以从 `[h_text; h_audio; h_vision]` 学到标签边界。

当前实验也不主张：

```text
residual probe 一定证明了真正的互补语义。
```

Residual probe 只是补充诊断。它可以说明某些差异特征有判别相关性，但不能单独证明这些差异就是干净、有效、因果的互补信息。

当前实验更准确的主线是：

```text
同样本对齐是否有效，取决于样本内部的跨模态关系状态。
```

## 2. 总流程一览

一次 `run_phenomenon.py` 单 seed 实验大概做这些事：

```text
1. 读取 MOSI/MOSEI aligned npz
2. 把连续情感分数转成 three_class 或 binary 标签
3. 训练 reference diagnostic model
4. 用 reference model 输出每个模态的概率分布 p_t/p_a/p_v
5. 计算跨模态分歧 D
6. 用验证集 D 的 q33/q66 划分 Low-D/Mid-D/High-D
7. 计算每个模态的可靠性 R_m，再得到样本可靠性 R_sample
8. 用验证集可靠性阈值划分 RA/UA/RD/ND
9. 训练 Concat、UncondAlign、UncondInfoNCE、TextInject、BalancedDirectAdd
10. 对每个 method 在同一组 test relation states 上算 Macro-F1
11. 输出 delta 表：method - Concat
12. 输出 calibration 表和 residual probe 表
```

`run_multi_seed.py` 做的是把多个 seed 的单 seed 输出合并：

```text
1. 调用多次 run_phenomenon.py
2. 读取每个 seed 的 CSV
3. 计算 mean/std/sem/95% CI/positive rate/error-control
4. 输出 summary CSV 和图
```

## 3. 数据和标签

### 3.1 输入数据

默认数据路径：

```text
E:\Xu\data\MultiBench\mosi\mosi_aligned.npz
E:\Xu\data\MultiBench\mosei\mosei_aligned.npz
```

每个 `.npz` 需要包含：

```text
train_text, train_vision, train_audio, train_label
valid_text, valid_vision, valid_audio, valid_label
test_text, test_vision, test_audio, test_label
```

`text/vision/audio` 可以是：

```text
[N, D]
```

也可以是：

```text
[N, T, D]
```

当前 encoder 对 `[N, T, D]` 的处理很简单：先对时间维 `T` 做平均池化，再过 MLP。

### 3.2 `label_mode`

默认是三分类：

```text
--label_mode three_class
```

转换规则：

```text
score < -0.5          -> class 0, negative
-0.5 <= score <= 0.5  -> class 1, neutral
score > 0.5           -> class 2, positive
```

补充鲁棒性实验可以用二分类：

```text
--label_mode binary
```

转换规则：

```text
score <= 0 -> class 0
score > 0  -> class 1
```

当前论文主线仍建议用 `three_class`。`binary` 是 appendix/robustness，不是默认主线。

## 4. Reference diagnostic model

### 4.1 为什么需要 reference model

我们需要先得到每个样本内部的模态关系状态。但真实数据没有直接标注：

```text
这个样本是可靠一致，还是可靠冲突，还是噪声冲突。
```

所以代码先训练一个参考诊断模型，用它的单模态头输出：

```text
p_text
p_audio
p_vision
```

这些概率分布用来计算：

```text
D: 模态之间预测分布是否分歧
R: 每个模态预测是否自信
```

### 4.2 训练目标

Reference model 是 joint training，不是三个独立单模态模型。

公式：

```text
L_ref = CE(logits_fusion, y)
        + eta_unimodal * mean(
            CE(logits_text, y),
            CE(logits_audio, y),
            CE(logits_vision, y)
          )
```

默认：

```text
eta_unimodal = 0.1
```

其中：

```text
h_t, h_a, h_v
```

是三个模态 encoder 输出的 hidden states。

单模态头直接接在：

```text
logits_t = C_t(h_t)
logits_a = C_a(h_a)
logits_v = C_v(h_v)
```

融合头接在：

```text
logits_f = C_f([h_t; h_v; h_a])
```

注意：reference model 的作用是分组诊断，不是最终要比较的主方法。

## 5. Disagreement: D 是什么

### 5.1 默认 D: probability JSD

默认设置：

```text
--disagreement_metric prob_jsd
```

Reference model 输出每个模态的预测概率：

```text
p_t = softmax(logits_t)
p_a = softmax(logits_a)
p_v = softmax(logits_v)
```

然后计算 Jensen-Shannon divergence：

```text
D_ta = JSD(p_t, p_a)
D_tv = JSD(p_t, p_v)
D_av = JSD(p_a, p_v)
```

JSD 可以理解为两个概率分布“意见不一致”的程度。越大，两个模态的预测分布越不一样。

### 5.2 text-anchor pair mode

当前 v5 主线以 text 为语义锚点：

```text
--pair_mode text_anchor
```

所以样本级分歧是：

```text
D_sample = mean(D_ta, D_tv)
```

也就是只看：

```text
text-audio
text-vision
```

不把 `audio-vision` 放进主线分组。

原因是用户已经明确决定：

```text
音频向文字对齐即可，图像也向文字对齐即可。
三者全对齐方向可能冲突。
```

所以 `full_pair` 只作为 appendix/diagnostic：

```text
--pair_mode full_pair
D_sample = mean(D_ta, D_tv, D_av)
```

### 5.3 一个简单例子

假设是三分类，某个样本的三个模态预测为：

```text
p_text   = [0.05, 0.10, 0.85]
p_audio  = [0.10, 0.20, 0.70]
p_vision = [0.80, 0.15, 0.05]
```

text 和 audio 都认为是 positive，分布接近：

```text
D_ta 较小
```

text 和 vision 一个认为 positive，一个认为 negative：

```text
D_tv 较大
```

在 text-anchor 下：

```text
D_sample = mean(D_ta, D_tv)
```

这个样本可能被分到 High-D，因为 vision 和 text 冲突强。

## 6. Low-D / Mid-D / High-D 如何划分

代码只用验证集的 D 分布定阈值：

```text
q33, q66 = quantile(valid_D_sample, [1/3, 2/3])
```

然后应用到 train/test：

```text
D <= q33          -> Low-D
q33 < D <= q66    -> Mid-D
D > q66           -> High-D
```

重要细节：

```text
Low-D/Mid-D/High-D 是 validation-quantile 分桶。
```

所以 Low/Mid/High 的样本数接近分位切分，不等于“数据天然三等分质量”。解释时不能只看 bucket size，而要看：

```text
D_sample 的绝对值
各组的 delta pattern
多 seed 统计
```

## 7. Reliability: R 是什么

### 7.1 单模态可靠性

每个模态的可靠性来自预测熵：

```text
R_m = 1 - H(p_m) / log(K)
```

其中：

```text
K = 类别数
H(p_m) = -sum_k p_m(k) log p_m(k)
```

直觉：

```text
分布越尖锐 -> 熵越低 -> R_m 越高
分布越平均 -> 熵越高 -> R_m 越低
```

例子，三分类：

```text
p = [0.90, 0.05, 0.05] -> 高可靠性
p = [0.34, 0.33, 0.33] -> 低可靠性
```

### 7.2 样本级可靠性

在 text-anchor 下，样本可靠性是 text-audio 和 text-vision 两条边的平均：

```text
R_ta = R_text * R_audio
R_tv = R_text * R_vision
R_sample = mean(R_ta, R_tv)
```

如果 `R_text` 很低，那么即使 audio/vision 很自信，text-anchor 关系也会被压低。这符合“text 是语义锚点”的设定。

在 full-pair 下：

```text
R_sample = mean(R_ta, R_tv, R_av)
```

## 8. Relation states: RA / UA / RD / ND

### 8.1 为什么需要 relation states

只看 D 不够。

`Low-D` 表示模态预测一致，但这种一致可能有两种情况：

```text
RA: 大家都自信且一致
UA: 大家都不太自信，只是一起模糊
```

`High-D` 也有两种情况：

```text
RD: 大家都自信但互相冲突
ND: 有模态不自信，冲突可能是噪声
```

所以当前 relation states 是：

```text
RA = Low-D  + High-R  reliable agreement
UA = Low-D  + Low-R   uncertain agreement
RD = High-D + High-R  reliable disagreement
ND = High-D + Low-R   noisy disagreement
Mid-D = 中间分歧组，保留为过渡组
```

### 8.2 默认 `balanced_within_d`

默认：

```text
--relation_split balanced_within_d
```

它不是用一个全局 R 中位数切所有样本，而是：

```text
Low-D 内部用 Low-D validation R 中位数切 RA/UA
High-D 内部用 High-D validation R 中位数切 RD/ND
```

这样做的原因：

```text
Low-D 和 High-D 的 R 分布可能不同。
如果用一个全局 R 阈值，某些 relation state 可能极端稀疏。
```

兼容旧逻辑：

```text
--relation_split global_r
```

会用一个 validation global R median 切所有组。

### 8.3 一个 relation state 例子

假设验证集阈值已经算好：

```text
q33_D = 0.002
q66_D = 0.010
Low-D 内 R threshold = 0.20
High-D 内 R threshold = 0.35
```

某个 test 样本：

```text
D_sample = 0.001
R_sample = 0.50
```

因为 `D_sample <= q33_D`，它是 Low-D；又因为 `R_sample >= 0.20`，它是 High-R。所以：

```text
relation_state = RA
```

另一个样本：

```text
D_sample = 0.030
R_sample = 0.60
```

它是 High-D，并且在 High-D 内是 High-R，所以：

```text
relation_state = RD
```

解释上它表示：

```text
模态之间意见冲突很强，但这种冲突来自比较自信的预测。
```

## 9. Calibration 表怎么看

输出：

```text
relation_state_distribution_calibration.csv
relation_state_distribution_calibration_summary.csv
```

核心列：

```text
group
n
class_0_ratio / class_1_ratio / class_2_ratio
avg_R
avg_R_text
avg_R_audio
avg_R_vision
text_acc
audio_acc
vision_acc
fusion_acc
```

它回答两个问题：

```text
1. RA/UA/RD/ND 是否为空或极端稀疏？
2. High-R 组是否真的更像“可靠”？
```

如果 High-R 组的单模态 accuracy 没有明显高于 Low-R，论文里就不要强称“true reliability”。更稳的说法是：

```text
confidence-based reliability
```

也就是“基于预测置信度的可靠性诊断”。

## 10. 主线 baseline

### 10.1 Concat

Concat 是监督融合基线：

```text
[h_t; h_v; h_a] -> fusion classifier
```

它没有额外 alignment loss。

在 delta 表里，其他方法都和它比较：

```text
delta_macro_f1 = method_macro_f1 - concat_macro_f1
```

### 10.2 UncondAlign

UncondAlign 是普通无条件余弦对齐：

```text
L = CE_fusion + lambda_align * L_align
```

在 text-anchor 下：

```text
L_align = mean(
    1 - cos(h_t, h_a),
    1 - cos(h_t, h_v)
)
```

在 full-pair 下：

```text
L_align = mean(
    1 - cos(h_t, h_a),
    1 - cos(h_t, h_v),
    1 - cos(h_a, h_v)
)
```

关键修正：

```text
UncondAlign 必须使用 h_* hidden states。
它不应该使用 InfoNCE projection head 的 z_*。
```

当前代码已经按这个原则修复。

### 10.3 UncondInfoNCE

UncondInfoNCE 是普通 same-sample InfoNCE baseline：

```text
L = CE_fusion + lambda_nce * L_InfoNCE
```

当前版本默认使用 projection head：

```text
z_t = P_t(h_t)
z_a = P_a(h_a)
z_v = P_v(h_v)
```

分类仍然使用：

```text
h_t, h_a, h_v
```

InfoNCE 使用：

```text
z_t, z_a, z_v
```

这样做是为了避免 contrastive loss 直接压分类 hidden，让 baseline 更公平。

在 text-anchor 下：

```text
L_InfoNCE = mean(
    bidirectional_InfoNCE(z_t, z_a),
    bidirectional_InfoNCE(z_t, z_v)
)
```

在 full-pair 下：

```text
L_InfoNCE = mean(
    bidirectional_InfoNCE(z_t, z_a),
    bidirectional_InfoNCE(z_t, z_v),
    bidirectional_InfoNCE(z_a, z_v)
)
```

默认参数：

```text
--run_infonce
--lambda_nce_values 0.01 0.05 0.1 0.5
--nce_temperature 0.1
--use_nce_projection
--nce_proj_dim 128
```

### 10.4 TextInject

`direct_add_pair_mode=text_anchor` 时，代码把 DirectAdd 标记为：

```text
TextInject
```

CLI 里 `--direct_add_pair_mode` 只接受：

```text
text_anchor
full_pair
```

`balanced` 不作为 primary DirectAdd 的入口，避免和独立的 BalancedDirectAdd appendix 输出重复。

它做的是：

```text
fuse_text   = h_t
fuse_audio  = h_a + alpha * h_t
fuse_vision = h_v + alpha * h_t
```

这不是公平 alignment baseline，因为 text 被保留，audio/vision 被注入 text 信息。它更像 text-enhancement trick。

所以它不应该放主动机表，只能作为 appendix/diagnostic。

### 10.5 BalancedDirectAdd

BalancedDirectAdd 是更公平的 appendix baseline：

```text
n_t = LayerNorm(h_t)
n_a = LayerNorm(h_a)
n_v = LayerNorm(h_v)
h_avg = mean(n_t, n_a, n_v)

fuse_text   = h_t + alpha * h_avg
fuse_audio  = h_a + alpha * h_avg
fuse_vision = h_v + alpha * h_avg
```

它对三个模态一视同仁，不把 text 单独注入到别的模态。

代码会始终把它作为单独 baseline 训练和输出，而不是通过
`--direct_add_pair_mode balanced` 选择。这样 TextInject/DirectAdd 和
BalancedDirectAdd 的含义不会混在一起。

注意：

```text
BalancedDirectAdd 可以很强，但它不是主动机证据。
```

如果它很强，说明“公平融合增强”有价值；但论文主线仍然应该看：

```text
Concat / UncondAlign / UncondInfoNCE
```

## 11. 为什么要做 lambda sweep

UncondAlign 和 UncondInfoNCE 都有超参数：

```text
lambda_align
lambda_nce
```

代码不是固定一个值，而是在验证集上选最优：

```text
选择 valid Macro-F1 最高的 lambda
```

然后用这个 best lambda 报 test delta。

这避免了一个不公平问题：

```text
如果固定 lambda，baseline 可能只是因为超参数没调好而差。
```

为了让 sweep 更接近“只改变 lambda/alpha”，代码在每个候选训练前都会重置随机种子和
train DataLoader generator。这样同一轮 sweep 中的候选不会因为初始化或 batch shuffle
不同而额外引入噪声。

相关输出：

```text
lambda_sweep_valid.csv
lambda_test_delta_metrics.csv
infonce_lambda_sweep_valid.csv
infonce_lambda_test_delta_metrics.csv
```

## 12. Delta 表怎么读

最常用输出：

```text
delta_metrics.csv
infonce_delta_metrics.csv
relation_state_delta.csv
infonce_relation_state_delta.csv
```

含义：

```text
delta_macro_f1 = method_macro_f1 - concat_macro_f1
```

例子：

```text
group = RD
delta_macro_f1 = -0.06
```

表示在 RD 组上，该 method 的 Macro-F1 比 Concat 低 0.06。

解释重点不是单个数，而是 pattern：

```text
Low-D / RA 上是否更容易正收益？
High-D / RD 上是否更容易负收益或不稳定？
UncondInfoNCE 是否没有稳定优于 Concat？
```

多 seed 时看：

```text
*_mean
*_ci95_low
*_ci95_high
*_positive_rate
*_passes_error_control
```

## 13. Error control 是什么

Multi-seed summary 里有：

```text
delta_macro_f1_positive_rate
delta_macro_f1_ci95_low
delta_macro_f1_ci95_high
delta_macro_f1_passes_error_control
```

默认通过条件：

```text
seed count >= 5
same-sign seed ratio >= 0.8
95% CI 不跨 0
```

所以如果某个方法 mean 是正的，但 CI 跨 0，或者正向 seed 只有 3/5，就不要说它稳定有效。

更稳的说法：

```text
observed positive trend, but not error-control stable
```

## 14. Residual probe 是什么

### 14.1 为什么 residual probe 降级为补充诊断

Residual 特征可能有冗余信息。比如：

```text
|h_t - h_a|
```

不一定只表示“差异信息”。它可能还携带：

```text
h_t 的强度
h_a 的强度
类别相关共性
encoder 的尺度偏差
```

所以 residual probe 不能作为主命题，只能做补充：

```text
RD 中是否存在可判别的差异相关信号？
```

### 14.2 当前 residual 特征

默认先 L2-normalize hidden：

```text
h_m_norm = h_m / ||h_m||
```

text-anchor residual：

```text
|h_t - h_a|
|h_t - h_v|
```

full-pair diagnostic residual：

```text
|h_t - h_a|
|h_t - h_v|
|h_a - h_v|
```

by-mode 表还支持：

```text
abs    = |left - right|
signed = left - right
prod   = left * right
all    = concat(abs, signed, prod)
```

### 14.3 probe 如何训练

对每个 relation state 单独训练一个 LogisticRegression probe：

```text
train samples in RA -> test samples in RA
train samples in RD -> test samples in RD
...
```

特征组合：

```text
common-only:      [h_t; h_a; h_v]
residual-only:    residual features
common+residual:  [h_t; h_a; h_v; residual]
```

### 14.4 控制组

当前有两类控制：

```text
label-shuffled residual:
  训练 residual-only probe 时打乱训练标签。

sample-shuffled residual:
  保留 common 特征和标签，但打乱 residual 与样本的对应关系。
```

更强的证据不是“residual-only 高”，而是：

```text
common+matched_residual > common-only
common+matched_residual > common+sample_shuffled_residual
matched_residual_only > label_shuffled_residual_only
```

如果这些都不成立，就不要写“RD residual 有明显判别价值”。

## 15. 输出文件地图

### 15.1 单 seed 核心输出

```text
config.json
summary.json
test_groups.csv
group_metrics.csv
delta_metrics.csv
lambda_sweep_valid.csv
lambda_test_delta_metrics.csv
relation_state_distribution_calibration.csv
relation_state_metrics.csv
relation_state_delta.csv
uncond_align_relation_delta.csv
direct_add_delta_metrics.csv
direct_add_relation_state_delta.csv
balanced_direct_add_delta_metrics.csv
balanced_direct_add_relation_state_delta.csv
residual_discriminative_probe.csv
residual_probe_by_mode.csv
concat_aware_motivation.csv
```

启用 `--run_infonce` 后额外有：

```text
infonce_lambda_sweep_valid.csv
infonce_lambda_test_delta_metrics.csv
infonce_delta_metrics.csv
infonce_high_d_reliability_delta.csv
infonce_relation_state_delta.csv
infonce_relation_delta.csv
```

### 15.2 multi-seed 核心输出

```text
multi_seed_config.json
multi_seed_group_metrics_summary.csv
multi_seed_delta_summary.csv
relation_state_delta_summary.csv
relation_state_distribution_calibration_summary.csv
uncond_align_relation_delta_summary.csv
infonce_delta_summary.csv
infonce_relation_state_delta_summary.csv
error_control_report.csv
residual_discriminative_probe_summary.csv
residual_probe_by_mode_summary.csv
concat_aware_motivation_summary.csv
```

其中：

```text
*_all.csv      = 每个 seed 的原始合并
*_summary.csv  = mean/std/CI/error-control 汇总
```

## 16. 推荐读表顺序

如果你现在有点混乱，建议按这个顺序看：

### 第一步：看分组是否合理

```text
relation_state_distribution_calibration_summary.csv
```

检查：

```text
RA/UA/RD/ND 的 n 是否太小
avg_R 是否符合 High-R/Low-R 的直觉
High-R accuracy 是否真的更高
class ratio 是否严重偏斜
```

### 第二步：看主线 baseline

```text
multi_seed_group_metrics_summary.csv
multi_seed_delta_summary.csv
infonce_delta_summary.csv
```

先看：

```text
Concat
UncondAlign
UncondInfoNCE
```

不要先看 DirectAdd。

### 第三步：看 relation-dependent pattern

```text
relation_state_delta_summary.csv
infonce_relation_state_delta_summary.csv
```

重点看：

```text
RA vs UA
RD vs ND
Low-D vs High-D
```

如果 InfoNCE overall 不差，但 RD/High-D 不稳定或变差，这反而支持“普通 same-sample positive 不够细”。

### 第四步：看 appendix baseline

```text
direct_add_relation_state_delta_summary.csv
balanced_direct_add_relation_state_delta_summary.csv
```

这一步只回答：

```text
TextInject 和 BalancedDirectAdd 是否作为补充对照有意义？
```

不要让它们替代主线。

### 第五步：看 residual probe

```text
residual_probe_by_mode_summary.csv
```

重点不是最高分，而是：

```text
matched 是否超过 shuffled
common+residual 是否超过 common-only
RD 上是否特别明显
```

## 17. 常见误解

### 17.1 “D 大就是差异信息多吗？”

不一定。

`D` 是预测分布分歧，不是直接的语义互补量。它只是把样本分成不同诊断区域。

更稳的解释：

```text
D 大表示模态预测意见不一致。
这种不一致可能是可靠互补，也可能是噪声。
```

所以才需要 R，把 High-D 再分成：

```text
RD / ND
```

### 17.2 “R 高就一定是真可靠吗？”

不一定。

R 来自预测熵，是 confidence-based reliability。模型可能自信但错。

所以 calibration 表必须报告：

```text
Text Acc / Audio Acc / Vision Acc / Fusion Acc
```

如果 High-R accuracy 不高，就只能说 confidence-based，不要说 ground-truth reliability。

### 17.3 “Residual 正交或去冗余后就是互补信息吗？”

不保证。

正交只能减少线性冗余，不能保证剩下的信息就是有效互补，也不能保证它和标签有因果关系。

当前代码没有把 residual probe 当作主线成功条件，这是对的。

### 17.4 “BalancedDirectAdd 很强，是否说明主线应该改成它？”

不一定。

BalancedDirectAdd 是融合增强 baseline，不是普通 alignment objective。它强说明：

```text
共享平均表示可能改善监督融合。
```

但它不能直接证明：

```text
same-sample InfoNCE 是不是最优
relation-aware alignment 是否必要
```

所以它适合作为 appendix baseline。

### 17.5 “为什么不用 A-V？”

当前主线是 text-anchor：

```text
audio -> text
vision -> text
```

因为 text 是语义锚点，audio 和 vision 未必应该互相直接对齐。比如语音语调和面部表情可能表达不同侧面，强行 A-V 对齐可能冲突。

`full_pair` 仍然可以跑，但建议作为 appendix。

## 18. 代码检查中发现并修复的细节

本轮检查发现 projection head 的使用边界之前不够干净：

```text
计划要求：
  UncondAlign 使用 h_t/h_a/h_v
  UncondInfoNCE 使用 z_t/z_a/z_v = P_m(h_m)

修复前细节：
  loss 边界存在错位风险。

当前修复：
  unconditional_alignment_loss 固定读取 h_*
  unconditional_infonce_loss 优先读取 z_*，没有 z_* 时才回退 h_*
```

同时 smoke test 增加了边界断言：

```text
修改 z_* 不应改变 UncondAlign loss
修改 z_* 应该改变 UncondInfoNCE loss
```

注意：在这次修复前跑出的 MOSI 5-seed 结果，尤其是 `UncondAlign` 和 `UncondInfoNCE`，不建议作为最终论文表直接引用。建议用修复后的代码重跑主线结果。

后续又修复了两个实验边界问题：

```text
1. direct_add_pair_mode 不再接受 balanced，BalancedDirectAdd 始终是独立 appendix baseline。
2. lambda/alpha sweep 的每个候选训练前重置随机种子，减少候选之间的初始化和 shuffle 噪声。
```

## 19. 推荐命令

### 19.1 单 seed 快速检查

```powershell
python -B code\disagreement_phenomenon\scripts\run_phenomenon.py --dataset mosi --data_root E:\Xu\data\MultiBench --seed 1 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --pair_mode text_anchor --relation_split balanced_within_d --deterministic
```

### 19.2 MOSI 5 seeds 主线

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosi --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --pair_mode text_anchor --relation_split balanced_within_d --deterministic
```

### 19.3 Binary robustness

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosi --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --pair_mode text_anchor --relation_split balanced_within_d --label_mode binary --deterministic
```

### 19.4 MOSEI 5 seeds

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --pair_mode text_anchor --relation_split balanced_within_d --deterministic
```

## 20. 当前最稳的论文叙事

如果后续重跑结果继续支持当前趋势，建议叙事保持克制：

```text
1. 普通 same-sample alignment / InfoNCE 在不同 relation states 上收益不稳定。
2. 低分歧不等于可靠一致，高分歧也不等于有效互补。
3. 需要把 agreement / disagreement 与 confidence-based reliability 结合起来。
4. 因此，relation-aware selective alignment 比无条件拉近所有同样本模态更合理。
```

不要把 residual probe 写成主命题；不要让 BalancedDirectAdd 抢主线；不要把 confidence-based R 写成真实可靠性标签。
