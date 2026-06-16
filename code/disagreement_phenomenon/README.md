# Disagreement Phenomenon Experiment

This package is the clean motivation loop for CoPA-v5:

> paired multimodal samples are not always alignment-positive.

The motivation experiment is intentionally narrow. It does not try to prove the
full CoPA-v5 method. It tests whether cross-modal alignment gains depend on
the relation state of a sample, and whether reliable disagreement contains a
diagnostic residual signal.

For a fuller Chinese walkthrough of the motivation, terminology, examples,
CSV outputs, and code-level caveats, see:

```text
MOTIVATION_EXPERIMENT_GUIDE.md
```

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

The narrower claim is that Concat and unconditional alignment objectives do not
explicitly model same-sample relation states. A paired sample can be reliable
agreement, uncertain agreement, reliable disagreement, or noisy disagreement.
Those cases should not all be treated as unconditional alignment positives.

For v5, text is the semantic anchor. The primary disagreement score therefore
uses text-anchor pairs only:

```text
D_text_anchor = mean(JSD(text, audio), JSD(text, vision))
```

Audio-vision disagreement is still available as a backup/full-pair diagnostic,
but it is not the default v5 motivation grouping.

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

The clean motivation loop has four parts.

1. Relation-state diagnostic

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

2. Unconditional alignment sensitivity

   The main motivation table compares the supervised Concat baseline with:

   ```text
   UncondAlign
   Uncond InfoNCE
   ```

   InfoNCE uses projection heads by default:

   ```text
   z_m = P_m(h_m)
   ```

   Classification heads still consume `h_m`; the InfoNCE loss consumes
   `z_text`, `z_audio`, and `z_vision`. This keeps the contrastive auxiliary
   loss from being interpreted as directly acting on the classifier hidden
   states. The default is `--use_nce_projection --nce_proj_dim 128`.

   All alignment/contrastive components use the same pair graph in a run. The primary v5
   setting is `--pair_mode text_anchor`, so diagnostics, UncondAlign and
   InfoNCE all use `T-A/T-V`. The appendix counterpart is
   `--pair_mode full_pair`, where all of them include `A-V` as well. The
   intended conclusion is conservative:

   ```text
   unconditional alignment gains are relation-dependent
   ```

   DirectAdd remains implemented for compatibility and appendix diagnostics.
   Under `text_anchor` it is reported as `TextInject`, because it keeps text
   fixed and injects text information into audio and vision rather than acting
   as a plain alignment objective. A separate appendix baseline,
   `BalancedDirectAdd`, LayerNorms the three hidden states, averages them, and
   adds the same averaged vector to all modalities.

3. Selective agreement evidence

   Compare `RA` and `UA` to test whether Low-D samples should all be treated as
   alignment-positive. The expected claim is:

   ```text
   reliable agreement is a better alignment-positive signal than Low-D alone
   ```

4. Supplementary residual probe

   Use label-free diagnostic features from the reference model, not
   method-trained residual heads or true-label class means. Hidden states are
   L2-normalized before residual differences are computed:

   ```text
   |h_text - h_audio|
   |h_text - h_vision|
   ```

   Residual probe is supplementary, not the main claim. It can be run with
   residual modes `abs`, `signed`, `prod`, and `all`; each mode reports matched
   residual, label-shuffled residual, and sample-shuffled residual controls.
   The cautious evidence is:

   ```text
   common+residual > common-only
   or residual-only > shuffled residual
   and common+residual > common+sample-shuffled residual
   ```

   This supports only the cautious claim that reliable disagreement may contain
   discriminative residual information.

## Main CLI

Single seed:

```powershell
python -B code\disagreement_phenomenon\scripts\run_phenomenon.py --dataset mosi --data_root E:\Xu\data\MultiBench --seed 1 --epochs 25 --patience 6 --run_infonce --pair_mode text_anchor --relation_split balanced_within_d --deterministic
```

MOSEI multi-seed:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --pair_mode text_anchor --relation_split balanced_within_d --deterministic
```

The primary v5 pair graph is:

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
only selects the primary DirectAdd/TextInject appendix mode (`text_anchor` or
`full_pair`). `BalancedDirectAdd` is always trained and reported as a separate
appendix baseline with `direct_add_pair_mode=balanced`, so it cannot be
accidentally used as the primary DirectAdd run.

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
kernel_distribution_relation_metrics.csv
kernel_distribution_relation_summary.csv
uncond_align_relation_delta.csv
direct_add_delta_metrics.csv
direct_add_relation_state_delta.csv
balanced_direct_add_delta_metrics.csv
balanced_direct_add_relation_state_delta.csv
residual_discriminative_probe.csv
residual_probe_by_mode.csv
concat_aware_motivation.csv
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

Multi-seed primary files:

```text
multi_seed_delta_summary.csv
error_control_report.csv
relation_state_delta_summary.csv
relation_state_distribution_calibration_summary.csv
kernel_distribution_relation_summary.csv
uncond_align_relation_delta_summary.csv
direct_add_delta_summary.csv
direct_add_relation_state_delta_summary.csv
balanced_direct_add_delta_summary.csv
balanced_direct_add_relation_state_delta_summary.csv
residual_discriminative_probe_summary.csv
residual_probe_by_mode_summary.csv
concat_aware_motivation_summary.csv
```

When `--run_infonce` is used:

```text
infonce_delta_summary.csv
infonce_high_d_reliability_summary.csv
infonce_relation_state_delta_summary.csv
infonce_relation_delta_summary.csv
infonce_lambda_test_delta_summary.csv
```

## Backup / Appendix Material

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
