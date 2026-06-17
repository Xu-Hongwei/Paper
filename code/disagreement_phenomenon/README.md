# Disagreement Phenomenon Experiment

This package is the clean motivation loop for CoPA-v6:

> multimodal disagreement is structured, so fusion should condition how it
> utilizes disagreement on relation states.

The motivation experiment is intentionally narrow. It does not try to prove the
full CoPA-v6 method. It tests whether high disagreement is actually difficult,
whether reliability disambiguates High-D samples, whether unconditional
alignment / ordinary InfoNCE / dynamic weighting are insufficient, and whether
balanced utilization is a stronger clue for reliable disagreement.

For the fuller historical Chinese walkthrough of the v5 motivation,
terminology, examples, CSV outputs, and code-level caveats, see:

```text
MOTIVATION_EXPERIMENT_GUIDE.md
```

The v6 framing in this README and `论文灵感/version6/` supersedes older guide
sections that describe `BalancedDirectAdd` as appendix-only.

Older method notes, full CoPA-v5 details, label-aware diagnostics, prototype
checks, and appendix-style code snapshots were moved to:

```text
code/disagreement_phenomenon/backup/
```

## Clean Motivation Boundary

Experiment 1 does not claim that Concat cannot learn labels. Concat learns
label-level decision boundaries through the supervised objective:

```text
concat(h_text, h_vision, h_audio) -> y
```

The narrower v6 claim is that high cross-modal disagreement is not automatically
bad. A paired sample can be reliable agreement, uncertain agreement, reliable
disagreement, or noisy disagreement. Those cases should not all be treated as
unconditional alignment positives, nor should they all receive the same fusion
interaction.

For the current v6 motivation, text remains the default semantic anchor. The
primary disagreement score therefore uses text-anchor pairs only:

```text
D_text_anchor = mean(JSD(text, audio), JSD(text, vision))
```

Audio-vision disagreement is still available as a backup/full-pair diagnostic,
but it is not the default v6 motivation grouping.

## Expected Data

The runner expects one `.npz` file per dataset:

```text
E:\Xu\data\MultiBench\mosi\mosi_aligned.npz
E:\Xu\data\MultiBench\mosei\mosei_aligned.npz
```

Each file must contain:

```text
train_text, train_vision, train_audio, train_label
valid_text, valid_vision, valid_audio, valid_label
test_text, test_vision, test_audio, test_label
```

Labels are continuous sentiment scores in `[-3, 3]`. The default
`--label_mode three_class` keeps the existing three-class protocol:

```text
label < -0.5          -> negative
-0.5 <= label <= 0.5  -> neutral
label > 0.5           -> positive
```

For appendix/robustness checks, `--label_mode binary` uses:

```text
label <= 0 -> negative
label > 0  -> positive
```

## Primary Motivation Flow

The clean v6 motivation loop has four required motivation checks, two
supplementary diagnostics, and one optional pilot. The required checks are
enough for a motivation-first paper claim; the pilot should not be written as
final method success.

1. Disagreement is not difficulty

   Train the supervised Concat baseline and report Low-D / Mid-D / High-D
   performance. On the current MOSEI multi-seed evidence, Concat is strongest
   on High-D, so the motivation should not claim that disagreement is
   inherently harmful or noisy.

2. Relation-state diagnostic

   Train the reference diagnostic model, use validation thresholds only, and
   apply the fixed thresholds to test samples:

   ```text
   L_ref = CE(C_f([h_text; h_audio; h_vision]), y)
           + eta_unimodal * mean_m CE(C_m(h_m), y)
   ```

   This is one joint model with shared training, not three separately trained
   unimodal models. The unimodal prediction heads are attached directly to
   `h_m`; they are used only to obtain `p_text`, `p_audio`, and `p_vision` for
   diagnostic grouping. The default is `eta_unimodal=0.1`.

   ```text
   D_text_anchor -> Low/Mid/High-D by validation q33/q66
   pair-graph reliability R -> High/Low-R
   ```

   In `--pair_mode text_anchor`, reliability is also text-anchored:

   ```text
   R_sample = mean(R_text * R_audio, R_text * R_vision)
   ```

   In `--pair_mode full_pair`, the audio-vision reliability edge is included
   too.

   The default `--relation_split balanced_within_d` uses validation-only
   reliability medians inside Low-D and High-D separately. This avoids letting
   a single global R threshold make RA/UA/RD/ND accidentally sparse. The old
   global median split remains available as `--relation_split global_r`.

   Relation states:

   ```text
   RA = Low-D + High-R   reliable agreement
   UA = Low-D + Low-R    uncertain agreement
   RD = High-D + High-R  reliable disagreement
   ND = High-D + Low-R   noisy disagreement
   ```

3. Old solutions are insufficient

   The main motivation table compares the supervised Concat baseline with:

   ```text
   UncondAlign
   Uncond InfoNCE
   DynamicFusion
   ```

   InfoNCE uses projection heads by default:

   ```text
   z_m = P_m(h_m)
   ```

   Classification heads still consume `h_m`; the InfoNCE loss consumes
   `z_text`, `z_audio`, and `z_vision`. This keeps the contrastive auxiliary
   loss from being interpreted as directly acting on the classifier hidden
   states. The default is `--use_nce_projection --nce_proj_dim 128`.

   All alignment/contrastive components use the same pair graph in a run. The primary v6
   setting is `--pair_mode text_anchor`, so diagnostics, UncondAlign and
   InfoNCE all use `T-A/T-V`. The appendix counterpart is
   `--pair_mode full_pair`, where all of them include `A-V` as well. The
   intended conclusion is conservative:

   ```text
   unconditional alignment gains are relation-dependent
   ```

   DirectAdd remains implemented for compatibility and text-injection
   diagnostics.
   Under `text_anchor` it is reported as `TextInject`, because it keeps text
   fixed and injects text information into audio and vision rather than acting
   as a plain alignment objective.

4. Balanced utilization clue

   `BalancedDirectAdd` LayerNorms the three hidden states, averages them, and
   adds the same averaged vector to all modalities. In v6 it is no longer only
   appendix material: it is the current minimal positive clue that reliable
   disagreement benefits more from balanced cross-modal utilization than from
   unconditional alignment, ordinary InfoNCE, or dynamic modality weighting.

5. Optional pilot: relation-conditioned BalancedAdd

   `RC-BalancedAdd` reuses the BalancedDirectAdd injection but replaces one
   global alpha with sample-level `alpha_i` from relation states:

   ```text
   rd_only: RD=1.0, all other states=0.0
   hard:    RD=1.0, RA=0.3, Mid-D=0.3, ND=0.1, UA=0.1
   ```

   Relation states come from reference-model prediction distributions and
   validation thresholds. Test labels are not used to assign alpha. In the
   current framing this is exploratory: it tests whether relation-conditioned
   scheduling is worth developing, but the core paper motivation does not depend
   on it beating fixed `BalancedDirectAdd`.

6. Selective agreement evidence

   Compare `RA` and `UA` to test whether Low-D samples should all be treated as
   alignment-positive. The expected claim is:

   ```text
   reliable agreement is a better alignment-positive signal than Low-D alone
   ```

7. Supplementary residual probe

   Use label-free diagnostic features from the reference model, not
   method-trained residual heads or true-label class means. Hidden states are
   L2-normalized before residual differences are computed:

   ```text
   |h_text - h_audio|
   |h_text - h_vision|
   ```

   Residual probe is boundary analysis, not the main claim. It can be run with
   residual modes `abs`, `signed`, `prod`, and `all`; each mode reports matched
   residual, label-shuffled residual, and sample-shuffled residual controls.
   The cautious evidence is:

   ```text
   common+residual > common-only
   or residual-only > shuffled residual
   and common+residual > common+sample-shuffled residual
   ```

   Current v6 framing treats negative or weak residual results as evidence that
   simple residual utilization is not the main answer.

## Main CLI

Single seed:

```powershell
python -B code\disagreement_phenomenon\scripts\run_phenomenon.py --preset v6_motivation --dataset mosi --data_root E:\Xu\data\MultiBench --seed 1
```

MOSEI multi-seed:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --preset v6_motivation --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5
```

Optional pilot:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --preset v6_pilot --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5
```

Appendix bundle:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --preset appendix_full --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5
```

Presets only set defaults. Any explicit flag still overrides them, for example
`--preset v6_motivation --no-run_dynamic_fusion` or
`--preset v6_motivation --pair_mode full_pair`.

The primary v6 pair graph is:

```powershell
--pair_mode text_anchor
```

Full-pair comparison remains available for appendix checks:

```powershell
--pair_mode full_pair
```

Kernel-MMD grouping also remains available as a backup diagnostic:

```powershell
--disagreement_metric kernel_mmd --pair_mode text_anchor
```

Prediction-class conditional batch MMD can be enabled as a distribution-level
appendix diagnostic:

```powershell
--run_kernel_dist_diagnostic --kernel_dist_min_group_size 10
```

This diagnostic groups hidden states by reference-model predicted class and
relation state, then computes RBF-MMD between text/audio and text/vision batch
distributions. It does not use test labels, replace `D_sample`, or enter the
training loss.

The older specific flags `--nce_pair_mode`, `--disagreement_pair_mode`,
`--kernel_pair_mode`, and `--align_pair_mode` are kept for compatibility, but
they must match the unified `--pair_mode` when provided. `--direct_add_pair_mode`
only selects the primary DirectAdd/TextInject diagnostic mode (`text_anchor` or
`full_pair`). `BalancedDirectAdd` is always trained and reported separately with
`direct_add_pair_mode=balanced`; v6 `RC-BalancedAdd` is enabled explicitly with
`--run_rc_balanced_add --rc_balanced_modes rd_only hard`.

## Primary Outputs

Single-run primary files:

```text
test_groups.csv
group_metrics.csv
delta_metrics.csv
lambda_test_delta_metrics.csv
high_d_reliability_delta.csv
relation_state_metrics.csv
relation_state_delta.csv
relation_state_distribution_calibration.csv
uncond_align_relation_delta.csv
direct_add_delta_metrics.csv
direct_add_relation_state_delta.csv
balanced_direct_add_delta_metrics.csv
balanced_direct_add_relation_state_delta.csv
concat_aware_motivation.csv
```

When `--run_residual_probe` is used:

```text
residual_discriminative_probe.csv
residual_probe_by_mode.csv
```

When `--run_kernel_dist_diagnostic` is used:

```text
kernel_distribution_relation_metrics.csv
kernel_distribution_relation_summary.csv
```

When `--run_rc_balanced_add` is used:

```text
rc_balanced_add_valid.csv
rc_balanced_add_delta_metrics.csv
rc_balanced_add_relation_state_delta.csv
```

When `--run_infonce` is used:

```text
infonce_delta_metrics.csv
infonce_lambda_sweep_valid.csv
infonce_lambda_test_delta_metrics.csv
infonce_high_d_reliability_delta.csv
infonce_relation_state_delta.csv
infonce_relation_delta.csv
```

When `--run_dynamic_fusion` is used:

```text
dynamic_fusion_delta_metrics.csv
dynamic_fusion_relation_state_delta.csv
dynamic_fusion_weight_relation_summary.csv
```

Multi-seed primary files:

```text
multi_seed_delta_summary.csv
error_control_report.csv
relation_state_delta_summary.csv
relation_state_distribution_calibration_summary.csv
uncond_align_relation_delta_summary.csv
direct_add_delta_summary.csv
direct_add_relation_state_delta_summary.csv
balanced_direct_add_delta_summary.csv
balanced_direct_add_relation_state_delta_summary.csv
concat_aware_motivation_summary.csv
experiment_one_disagreement_difficulty.json
uncond_align_delta_conclusion.json
```

When `--run_residual_probe` is used:

```text
residual_discriminative_probe_summary.csv
residual_probe_by_mode_summary.csv
```

When `--run_kernel_dist_diagnostic` is used:

```text
kernel_distribution_relation_summary.csv
```

When `--run_rc_balanced_add` is used:

```text
rc_balanced_add_delta_summary.csv
rc_balanced_add_relation_state_delta_summary.csv
```

When `--run_infonce` is used:

```text
infonce_delta_summary.csv
infonce_high_d_reliability_summary.csv
infonce_relation_state_delta_summary.csv
infonce_relation_delta_summary.csv
infonce_lambda_test_delta_summary.csv
```

When `--run_dynamic_fusion` is used:

```text
dynamic_fusion_delta_summary.csv
dynamic_fusion_relation_state_delta_summary.csv
dynamic_fusion_weight_relation_summary.csv
```

## Backup / Appendix Material

You do not need to run every appendix diagnostic for the motivation-first
version. Use this triage:

```text
Required main evidence:
- MOSEI three-class text_anchor multi-seed with --preset v6_motivation
- relation_state_distribution_calibration_summary.csv
- multi_seed_group_metrics_summary.csv
- relation-state delta summaries for UncondAlign, InfoNCE, DynamicFusion
- balanced_direct_add_relation_state_delta_summary.csv

Optional pilot:
- --run_rc_balanced_add --rc_balanced_modes rd_only hard

Appendix only when a reviewer/narrative needs it:
- MOSI robustness
- binary label robustness
- full_pair diagnostics
- kernel MMD distribution diagnostics
- residual probe boundary analysis with `--run_residual_probe`
```

The active motivation runner no longer writes label-aware, prototype, or CoPA
method outputs. Those branches are preserved in the backup snapshot:

```text
backup/MOTIVATION_EXPERIMENT_BACKUP.md
backup/code_snapshot/
```

Treat those files as backup evidence, not the main motivation table. In
particular:

```text
motivation/test grouping: entropy reliability, no labels
training gate diagnostics: label-support p_m(y), train/valid only
```

## Error Control

Multi-seed summaries report:

```text
*_sem
*_ci95_low
*_ci95_high
*_positive_rate
*_negative_rate
*_sign_consistency
*_ci95_excludes_zero
*_passes_error_control
```

A delta passes error control only when:

```text
seed count >= --error_min_seeds
same-sign seed ratio >= --error_sign_rate
95% CI does not cross zero
```

Defaults:

```text
--error_min_seeds 5
--error_sign_rate 0.8
```

## Smoke Test

```powershell
python code/disagreement_phenomenon/scripts/smoke_test.py
python code/disagreement_phenomenon/scripts/smoke_test_multi_seed.py
```
