# CoPA-v6 Motivation Section Draft

> Draft purpose: this section is written for the paper body. It uses the frozen MOSEI 15-seed motivation evidence and deliberately avoids claiming final method success.

## 1. Motivation Setup

We first ask whether high cross-modal disagreement is necessarily a difficult or harmful regime. For each sample, disagreement is measured from unimodal prediction distributions with `prob_jsd` under the `text_anchor` pair graph, using text-audio and text-vision pairs. We denote this quantity as `D_pred`: a supervised model-induced, task-aware disagreement signal. It is not constructed from validation or test labels, so it is not direct label leakage. However, it is also not a fully unsupervised or intrinsic data-only disagreement, because the unimodal predictors are trained under downstream task supervision. The Low-D, Mid-D, and High-D groups are defined by validation-set `D_pred` quantiles, and the same validation thresholds are applied to test samples.

Reliability is not a ground-truth quality label. It is a prediction-certainty diagnostic derived from the reference diagnostic model. Within Low-D and High-D, validation-set reliability medians define relation states: RA, UA, RD, and ND. Test labels are never used to assign test relation states.

Unless otherwise stated, the main evidence uses MOSEI, three-class sentiment classification, `balanced_within_d` relation splitting, and 15 independent seeds.

## 2. High-D Is Not Difficulty

The Concat baseline shows a monotonic increase from Low-D to High-D:

| Group | N | avg D | avg R | Concat Macro-F1 | 95% CI | Order Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Low-D | 1547.3 | 0.0137 | 0.0048 | 0.505 | [0.500, 0.510] | 15/15 |
| Mid-D | 1645.6 | 0.0392 | 0.0103 | 0.596 | [0.588, 0.603] | 15/15 |
| High-D | 1450.1 | 0.1015 | 0.0156 | 0.687 | [0.680, 0.695] | 15/15 |

This result contradicts the simple assumption that larger task-aware prediction disagreement always indicates harder or lower-quality samples. Across all 15 seeds, High-D consistently outperforms Mid-D and Low-D.

The claim should stop there: under downstream task semantics, model-induced High-D is not automatically difficult. This result alone does not prove that High-D contains intrinsic cross-modal complementarity.

## 3. Why High-D Looks Easier

To understand why High-D is strong, we inspect label statistics and unimodal branch behavior:

| Group | N | avg \|label\| | Label Entropy | Class 0/1/2 | Text Acc | Audio Acc | Vision Acc | Concat Macro-F1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Low-D | 1547.3 | 0.732 | 1.519 | 0.216/0.460/0.324 | 0.531 | 0.472 | 0.501 | 0.505 |
| Mid-D | 1645.6 | 0.759 | 1.483 | 0.170/0.450/0.380 | 0.587 | 0.468 | 0.498 | 0.596 |
| High-D | 1450.1 | 1.040 | 1.568 | 0.280/0.323/0.397 | 0.696 | 0.368 | 0.377 | 0.687 |

High-D is not simply easier because of lower label entropy; its label entropy is not the lowest among the three groups. Instead, High-D contains stronger sentiment intensity and much stronger text-branch predictability, while audio and vision branches are weaker.

We further check whether this pattern is merely a class-prior artifact. The largest class ratio is actually lowest in High-D:

| Group | N | Class Ratios | Majority Acc | Concat Acc | Concat Macro-F1 |
|---|---:|---|---:|---:|---:|
| Low-D | 1547.3 | 0.216/0.460/0.324 | 0.460 | 0.540 | 0.505 |
| Mid-D | 1645.6 | 0.170/0.450/0.380 | 0.450 | 0.605 | 0.596 |
| High-D | 1450.1 | 0.280/0.323/0.397 | 0.397 | 0.695 | 0.687 |

Class-wise accuracy gives a more nuanced picture:

| Class | Low-D Concat Acc | Mid-D Concat Acc | High-D Concat Acc | High-Low | High-Mid |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.340 | 0.552 | 0.863 | 0.522 | 0.311 |
| 1 | 0.666 | 0.618 | 0.491 | -0.175 | -0.127 |
| 2 | 0.493 | 0.614 | 0.741 | 0.248 | 0.127 |

Thus, High-D is not strong because its largest class dominates the group. Instead, the effect is class-dependent: High-D is much easier for the polar sentiment classes but harder for the neutral class. This supports a more precise motivation: task-aware high disagreement is a mixed regime. It may contain strong text evidence, weak non-text evidence, class-specific sentiment effects, and possibly usable cross-modal information. Therefore, the learning problem should not be framed as uniformly reducing disagreement, but as identifying the relation state under which disagreement should be used, suppressed, or rebalanced.

We also directly test whether `D_pred` is correlated with polarity and confidence signals:

| Signal | Pearson | Spearman |
|---|---:|---:|
| label_abs_polarity | 0.2423 | 0.1630 |
| pred_polarity_conf | 0.4470 | 0.4081 |
| pred_confidence | 0.4631 | 0.4407 |
| pred_margin | 0.4355 | 0.3727 |
| R_sample | 0.3700 | 0.5557 |

This confirms that `D_pred` is not a pure modality-conflict variable. It is more strongly coupled with model-induced confidence and predicted polarity than with the ground-truth sentiment magnitude alone.

Finally, we control for polarity strength by splitting samples into tertiles of `|label_reg|` and comparing Low-D, Mid-D, and High-D within each bin:

| Polarity Bin | Low-D Acc | Mid-D Acc | High-D Acc | High-Low | High-Mid |
|---|---:|---:|---:|---:|---:|
| Low-P | 0.665 | 0.617 | 0.492 | -0.173 | -0.126 |
| Mid-P | 0.334 | 0.501 | 0.666 | 0.332 | 0.165 |
| High-P | 0.547 | 0.701 | 0.874 | 0.327 | 0.173 |

After controlling for polarity strength, High-D is not uniformly easier. It is worse in the low-polarity bin and strongest in mid/high-polarity bins. The correct conclusion is therefore not that High-D is inherently easy, but that task-aware disagreement interacts strongly with sentiment polarity.

We also reduce circularity by decoupling the model that defines D groups from the model used for evaluation. Specifically, D groups are taken from one diagnostic seed, while Concat performance is evaluated with a different seed; same-seed pairs are excluded. The monotonic pattern remains:

| Group | N | Cross-seed Concat Acc | Cross-seed Macro-F1 |
|---|---:|---:|---:|
| Low-D | 1547.3 | 0.539 | 0.504 |
| Mid-D | 1645.6 | 0.606 | 0.596 |
| High-D | 1450.1 | 0.694 | 0.687 |
| High-Low |  | 0.155 | 0.183 |
| High-Mid |  | 0.088 | 0.091 |

This does not remove the polarity/class confound, but it addresses a different concern: the High-D ordering is not only a same-model self-validation artifact. The stronger interpretation is that `D_pred` captures a reproducible task-induced structure, while its meaning remains conditional on class and polarity.

## 4. Reliability Splits High-D

RD and ND are both high-disagreement states, but they differ in prediction certainty:

| Group | N | avg D | avg R | Text Macro-F1 | Audio Macro-F1 | Vision Macro-F1 | Fusion Macro-F1 | Concat Macro-F1 | Oracle Macro-F1 | Oracle - Fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RD | 754.1 | 0.1175 | 0.0247 | 0.688 | 0.295 | 0.370 | 0.692 | 0.693 | 0.922 | 0.230 |
| ND | 696.0 | 0.0842 | 0.0057 | 0.676 | 0.225 | 0.276 | 0.672 | 0.673 | 0.909 | 0.237 |

RD has higher reliability and stronger branch/fusion performance than ND. At the same time, both RD and ND retain a large oracle gap, meaning that at least one unimodal branch is often correct even when the learned fusion does not fully exploit that information.

This is the central v6 motivation: High-D is not homogeneous. Reliability separates high-disagreement samples into states with different predictive structure, and current fusion leaves recoverable information unused.

## 5. Existing Methods Are Insufficient

We compare unconditional alignment, unconditional InfoNCE, DynamicFusion, and BalancedDirectAdd against the Concat baseline:

| Method | Overall Delta Macro-F1 | Overall 95% CI | Overall EC | High-D Delta Macro-F1 | High-D 95% CI | High-D EC | RD Delta Macro-F1 | RD 95% CI | RD EC |
|---|---:|---|---|---:|---|---|---:|---|---|
| UncondAlign | 0.0023 | [-0.0003, 0.0048] | False | 0.0014 | [-0.0031, 0.0059] | False | 0.0023 | [-0.0026, 0.0073] | False |
| UncondInfoNCE | 0.0019 | [-0.0004, 0.0042] | False | 0.0035 | [-0.0000, 0.0070] | False | 0.0041 | [-0.0002, 0.0084] | False |
| DynamicFusion | 0.0028 | [-0.0010, 0.0067] | False | -0.0004 | [-0.0039, 0.0032] | False | -0.0008 | [-0.0060, 0.0044] | False |
| BalancedDirectAdd | 0.0034 | [0.0004, 0.0065] | False | 0.0056 | [0.0006, 0.0106] | False | 0.0072 | [0.0023, 0.0121] | True |

Unconditional alignment and ordinary contrastive learning do not provide stable RD gains. DynamicFusion also fails to recover RD improvement, despite having a small overall trend.

Mechanism diagnostics explain why these controls are insufficient. UncondAlign strongly reduces hidden-state distance, including on RD, but this closeness does not translate into reliable Macro-F1 improvement. DynamicFusion assigns most weight to text on RD, which indicates modality selection rather than relation-state scheduling.

## 6. Minimal Positive Clue

BalancedDirectAdd provides the cleanest positive clue: it is the only tested control here with an error-controlled RD gain. This does not mean that BalancedDirectAdd is the final method. It is globally applied with one fixed injection strength and does not know whether a sample is RA, UA, RD, or ND.

Its role is narrower and more useful: it closes the motivation loop. If unconditional alignment, contrastive learning, and dynamic weighting are insufficient, but a simple balanced utilization operation already helps RD, then relation-conditioned balanced utilization becomes a justified next direction.

The next minimal validation is therefore RC-BalancedAdd: reuse the same balanced injection, but replace the global fixed alpha with relation-state-specific alpha. If hard relation scheduling improves RD without hurting overall performance, it supports the v6 method direction. If it does not, the negative result still clarifies that hard rules are insufficient and motivates a soft gate rather than invalidating the motivation evidence.
