# Disagreement Phenomenon Experiment

This package implements the first CoPA motivation experiment and a first
label-aware CoPA prototype training baseline:

> paired multimodal samples are not always alignment-positive.

The default phenomenon experiment compares a plain concat fusion model against
an unconditional sample-level alignment model on Low/Mid/High disagreement
groups. It also exports the Experiment 1-v4 diagnostics for selective
agreement and discriminative disagreement:

```text
prediction-level relation signal
-> feature-level residual structure
-> residual discriminative evidence
```

CoPA can be enabled explicitly with `--run_copa`.

## Motivation Boundary

Experiment 1 does not claim that Concat cannot learn labels. Concat learns
label-level decision boundaries through the supervised objective:

```text
concat(h_text, h_vision, h_audio) -> y
```

The motivation question is narrower: Concat does not explicitly model relation
states inside paired multimodal samples. It may learn a useful overall
classifier while still hiding important subgroup differences such as reliable
agreement, uncertain agreement, reliable disagreement, and noisy disagreement.

In this experiment, "lossless" does not mean directly adding aligned features
back to the original representation. It means preserving the original
supervised fusion path for all samples, while applying relation-conditioned
auxiliary constraints only to reliable agreement or reliable disagreement
cases.

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

Labels should be the original continuous sentiment scores in `[-3, 3]`. The
code converts them to three classes:

```text
label < -0.5          -> negative
-0.5 <= label <= 0.5  -> neutral
label > 0.5           -> positive
```

Features can be sample-level arrays with shape `[N, D]` or sequence arrays with
shape `[N, T, D]`. Sequence arrays are mean-pooled inside the model.

## Run

Single-seed phenomenon run:

```powershell
python code/disagreement_phenomenon/scripts/run_phenomenon.py --dataset mosi --data_root E:\Xu\data\MultiBench
python code/disagreement_phenomenon/scripts/run_phenomenon.py --dataset mosei --data_root E:\Xu\data\MultiBench
```

Multi-seed phenomenon run:

```powershell
python code/disagreement_phenomenon/scripts/run_multi_seed.py --dataset mosi --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5
```

For MOSEI on the current machine, the stable large-batch setting is:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 6 7 8 9 10 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet
```

Avoid `--num_workers 2` on Windows for the large MOSEI arrays unless you have
verified it locally. It can fail with PyTorch shared-event errors.

For lower run-to-run implementation variance, add deterministic seeding:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_copa --deterministic
```

This fixes Python/NumPy/PyTorch seeds, uses a seeded DataLoader generator, and
enables deterministic PyTorch algorithms with warnings instead of hard failure.
It reduces implementation noise, but it does not remove true seed sensitivity
from model initialization and training.

## Methods

### Concat

The baseline uses three modality encoders and concatenates their hidden states:

```text
concat(h_text, h_vision, h_audio) -> classifier
```

### Unconditional Alignment

The unconditional alignment baseline adds a pairwise cosine alignment loss:

```text
L = L_task + lambda_align * L_align
```

`lambda_align` is swept with:

```text
0.001, 0.005, 0.01, 0.05, 0.1
```

The best model is selected by validation overall Macro-F1. The script also
exports the full lambda curve on test groups so the alignment strength effect
can be inspected directly.

### Error Control

Multi-seed summaries now report uncertainty columns in addition to
`mean/std/count`:

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

For delta metrics, `passes_error_control` is conservative. It is true only
when:

```text
seed count >= --error_min_seeds
same-sign seed ratio >= --error_sign_rate
95% CI does not cross zero
```

Defaults are:

```text
--error_min_seeds 5
--error_sign_rate 0.8
```

With only two seeds, most rows should remain `passes_error_control=false`; that
means the trend is preliminary rather than invalid. The compact overview is
written to:

```text
error_control_report.csv
```

### Direct Addition Alignment

DirectAdd is a simple motivation baseline for the question:

```text
If direct addition preserves original information, is relation-conditioned
soft splitting still needed?
```

It keeps the original supervised fusion path but directly adds a sample-level
aligned summary back to each modality before fusion:

```text
h_align = mean(h_text, h_vision, h_audio)
h_m_add = h_m + alpha * h_align
```

`alpha` is swept with:

```text
0.1, 0.3, 0.5, 1.0
```

DirectAdd is intentionally not relation-aware. If it is unstable on RD
(`High-D+High-R`) or sensitive to `alpha`, that supports the claim that direct
addition is not enough; relation-conditioned soft decomposition is needed.
DirectAdd is a motivation baseline, not a CoPA component.

### Unconditional InfoNCE Baseline

Enable same-sample InfoNCE with:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --nce_pair_mode text_anchor --deterministic
```

The default text-anchor mode only uses:

```text
text <-> audio
text <-> vision
```

It does not train an audio-vision alignment loss. To run the full-pair
counterpart, use:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_infonce --nce_pair_mode full_pair --deterministic
```

The sweep uses:

```text
lambda_nce = 0.01, 0.05, 0.1, 0.5
nce_temperature = 0.1
```

### Experiment 1-v4 diagnostics

Motivation grouping uses validation thresholds only. By default, the validation
set defines Low/Mid/High-D by the 1/3 and 2/3 quantiles of
prediction-distribution JSD disagreement, and defines High/Low-R by the median
entropy reliability. The test set only applies these thresholds.

The disagreement score can also be switched to a unified-hidden RBF kernel
variant:

```powershell
--disagreement_metric kernel_mmd --kernel_pair_mode text_anchor
```

In this mode, the diagnostic model first maps text/audio/vision into the same
hidden dimension. Pairwise D combines a paired RBF point distance with a
predicted-class conditional MMD estimate, using the diagnostic fusion
prediction as the class partition. `text_anchor` averages T-A and T-V for
`D_sample`; `full_pair` also includes A-V. This metric is a candidate
modality-distribution discrepancy signal, not proof that the discrepancy is
useful complementary information. The residual probe and RD subgroup results
are still the evidence for discriminative value.

The exported relation states are:

```text
RA = Low-D + High-R   reliable agreement
UA = Low-D + Low-R    uncertain agreement
RD = High-D + High-R  reliable disagreement
ND = High-D + Low-R   noisy disagreement
```

The v4 diagnostics include feature-level disagreement, residual distribution
diagnostics, residual-only/common-only/common+residual probes, shuffled
residual-label negative control, and a selective agreement prototype check.

`concat_aware_motivation.csv` combines the key diagnostic columns:

```text
Group | Concat F1 | UncondAlign F1 | DirectAdd F1 | SoftSplit Probe F1
```

This table is for motivation only. It tests whether relation states provide
meaningful diagnostic signals beyond ordinary concatenation, fixed-strength
alignment, and direct feature addition.

### CoPA-v5 Text-Anchor Baseline

Enable it with:

```powershell
python -B code\disagreement_phenomenon\scripts\run_phenomenon.py --dataset mosi --data_root E:\Xu\data\MultiBench --run_copa
```

or multi-seed:

```powershell
python -B code\disagreement_phenomenon\scripts\run_multi_seed.py --dataset mosei --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5 6 7 8 9 10 --batch_size 1024 --num_workers 0 --epochs 25 --patience 6 --quiet --run_copa
```

CoPA now follows the v5 text-anchor design. It keeps the supervised fusion path
unchanged, uses common projections for unimodal prediction and RA alignment,
and uses residual projections for RD prototype-NCE. The training loss only uses
text-anchor modality pairs:

```text
text <-> audio
text <-> vision
```

It does not train an audio-vision CoPA loss. The default CoPA gate is now the
v5-simple label-support gate. It only asks how much each modality supports the
true training label:

```text
s_m = p_m(y)
```

For each text-anchor modality pair:

```text
g_ij_agr = s_i s_j (1 - |s_i - s_j|)
g_ij_dis = max(s_i, s_j) |s_i - s_j|
```

The gates are detached. If both modalities strongly support the label and have
similar support, the pair contributes mostly to common InfoNCE. If one modality
supports the label and the other does not, the pair contributes mostly to
residual prototype-NCE. If neither modality supports the label, both gates stay
small and the sample mostly falls back to task loss.

The older reliability+agreement gate is still available as an enhanced
ablation:

```powershell
--copa_gate_type full_relation --copa_gate_metric prob_jsd
```

or with batch-local hidden-space RBF distance:

```powershell
--copa_gate_type full_relation --copa_gate_metric kernel_mmd --copa_kernel_bandwidth median
```

This uses the same common hidden space as RA alignment. It is intentionally a
gate signal rather than a direct claim that all disagreement is complementary.

The full-relation gate uses:

```text
C_m = 1 - H(p_m) / log(K)
S_m = p_m(y)
A_ij = exp(-D_ij / tau_agreement)
g_ij_agr   = C_i C_j S_i S_j A_ij
B_ij       = max(S_i, S_j)
g_ij_dis   = C_i C_j B_ij (1 - A_ij)
g_ij_noise = 1 - C_i C_j
```

The exported `g_ij_comp` columns are kept as backward-compatible aliases for
`g_ij_dis`. This gate allows a reliable contrastive modality to contribute to
disagreement learning when at least one modality strongly supports the label.

The current CoPA-v5 loss combines:

```text
RA-gated common InfoNCE: reliable agreeing text-anchor pairs -> align common semantics
RD residual prototype-NCE: reliable disagreeing text-anchor residuals -> classify residual structure
common-residual orthogonality: reduce redundancy between z^c and z^r
```

The orthogonality term is a redundancy-control regularizer only. It encourages
common and residual projections to avoid linear overlap, but it does not by
itself prove that the residual contains useful complementary evidence; that
still needs residual probe and subgroup analysis.

The CoPA sweep uses:

```text
lambda_copa = 0.01, 0.05, 0.1
copa_orth_weight = 0.01  # light redundancy control; 0.05 can over-regularize on MOSI seed 1
```

The legacy CLI names are kept for compatibility:

```text
copa_agr_weight   -> RA-gated common InfoNCE weight
copa_comp_weight  -> RD residual prototype-NCE weight
copa_proto_weight -> multiplier on the residual prototype-NCE term
copa_orth_weight  -> common-residual redundancy reduction weight
copa_gate_type    -> label_support or full_relation
copa_gate_metric  -> prob_jsd or kernel_mmd gate source
```

Important distinction:

```text
Motivation grouping on validation/test does not use labels.
CoPA-v5 supervised relation gates use train labels through p_m(y).
```

This avoids test-label leakage while still preventing confidently wrong training
modalities from contaminating prototypes.

## Prepare Data

Recommended route: use the MultiBench MOSI/MOSEI pickle files, then convert
them into this experiment's standard `.npz` format. In the current local setup,
MultiBench files are expected under:

```text
E:\Xu\data\MultiBench\mosi
E:\Xu\data\MultiBench\mosei
```

For MOSI, prefer `mosi_raw.pkl` for this experiment because it keeps the richer
feature dimensions:

```text
text=300, vision=35, audio=74
```

`mosi_data.pkl` is also valid but uses lower-dimensional features
(`vision=20, audio=5`).

For MOSEI, prefer `mosei_senti_data.pkl`. It has the same feature dimensions
needed by this experiment and avoids loading the much larger `mosei_raw.pkl`.

```powershell
python -m pip install gdown
python code/disagreement_phenomenon/scripts/prepare_multibench_data.py --dataset both --data_root E:\Xu\data\MultiBench --overwrite
```

If the Google Drive download is interrupted, re-run the same command. To convert
files that were already downloaded under `E:\Xu\data\MultiBench`, skip the
download step:

```powershell
python code/disagreement_phenomenon/scripts/prepare_multibench_data.py --dataset mosi --data_root E:\Xu\data\MultiBench --skip_download --variant auto --overwrite
python code/disagreement_phenomenon/scripts/prepare_multibench_data.py --dataset mosei --data_root E:\Xu\data\MultiBench --skip_download --variant auto --overwrite
```

These read:

```text
E:\Xu\data\MultiBench\mosi\mosi_raw.pkl
E:\Xu\data\MultiBench\mosei\mosei_senti_data.pkl
```

and write:

```text
E:\Xu\data\MultiBench\mosi\mosi_aligned.npz
E:\Xu\data\MultiBench\mosei\mosei_aligned.npz
```

Do not use `--dataset both` until the MOSEI folder contains a completed `.pkl`
file. A `*.part` file means the download is incomplete.

You can also convert a processed `.pkl` manually when it contains
`train/valid/test` splits with `text/vision/audio/labels` arrays:

```powershell
python code/disagreement_phenomenon/scripts/convert_pkl_to_npz.py --dataset mosi --input E:\Xu\data\MultiBench\mosi\mosi_raw.pkl --overwrite
```

Fallback route: rebuild from CMU Multimodal SDK `.csd` files. This depends on
the older CMU data server (`immortal.multicomp.cs.cmu.edu`), which may timeout
from some networks.

After installing the CMU Multimodal SDK:

```powershell
python -m pip install h5py
python -m pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
```

download and export the aligned `.npz` files:

```powershell
python code/disagreement_phenomenon/scripts/prepare_cmu_data.py --dataset both --data_root E:\Xu\data\MultiBench
```

This creates:

```text
E:\Xu\data\MultiBench\mosi\mosi_aligned.npz
E:\Xu\data\MultiBench\mosei\mosei_aligned.npz
```

Outputs are written to:

```text
code/disagreement_phenomenon/outputs/<dataset>/<timestamp>/
```

Key single-run outputs:

```text
test_groups.csv
group_metrics.csv
delta_metrics.csv
lambda_test_delta_metrics.csv
lambda_delta_macro_f1_curve.png
train_label_aware_relations.csv
valid_label_aware_relations.csv
label_aware_relation_summary.csv
relation_state_metrics.csv
relation_state_delta.csv
direct_add_alpha_sweep_valid.csv
direct_add_alpha_test_delta_metrics.csv
direct_add_delta_metrics.csv
direct_add_relation_state_delta.csv
concat_aware_motivation.csv
direct_add_model.pt
feature_consistency_diagnostic.csv
residual_distribution_diagnostic.csv
residual_discriminative_probe.csv
selective_agreement_prototype_check.csv
```

Additional InfoNCE outputs when `--run_infonce` is used:

```text
infonce_delta_metrics.csv
infonce_lambda_sweep_valid.csv
infonce_lambda_test_delta_metrics.csv
infonce_high_d_reliability_delta.csv
infonce_lambda_high_d_reliability_delta.csv
infonce_relation_state_delta.csv
infonce_model.pt
```

Additional CoPA outputs when `--run_copa` is used:

```text
copa_delta_metrics.csv
copa_lambda_sweep_valid.csv
copa_lambda_test_delta_metrics.csv
copa_high_d_reliability_delta.csv
copa_relation_state_delta.csv
copa_model.pt
```

Key multi-seed outputs:

```text
multi_seed_delta_summary.csv
error_control_report.csv
lambda_test_delta_summary.csv
lambda_delta_macro_f1_curve.png
multi_seed_delta_macro_f1_detailed.png
high_d_reliability_delta_detailed.png
relation_state_delta_detailed.png
direct_add_relation_state_delta_detailed.png
relation_state_method_comparison_heatmap.png
label_aware_relation_multi_seed_summary.csv
relation_state_delta_summary.csv
relation_state_metrics_summary.csv
direct_add_delta_summary.csv
direct_add_alpha_test_delta_summary.csv
direct_add_relation_state_delta_summary.csv
concat_aware_motivation_summary.csv
feature_consistency_diagnostic_summary.csv
residual_distribution_diagnostic_summary.csv
residual_discriminative_probe_summary.csv
selective_agreement_prototype_check_summary.csv
```

Additional multi-seed InfoNCE outputs when `--run_infonce` is used:

```text
infonce_delta_all.csv
infonce_delta_summary.csv
infonce_high_d_reliability_delta_all.csv
infonce_high_d_reliability_summary.csv
infonce_relation_state_delta_all.csv
infonce_relation_state_delta_summary.csv
infonce_lambda_test_delta_all.csv
infonce_lambda_test_delta_summary.csv
infonce_lambda_high_d_reliability_delta_all.csv
infonce_lambda_high_d_reliability_summary.csv
infonce_delta_macro_f1_detailed.png
infonce_high_d_reliability_delta_detailed.png
infonce_relation_state_delta_detailed.png
infonce_lambda_delta_macro_f1_curve.png
```

The multi-seed summary directory also keeps the corresponding per-seed merged
tables with `_all.csv` suffixes, including:

```text
direct_add_delta_all.csv
direct_add_alpha_test_delta_all.csv
direct_add_relation_state_delta_all.csv
concat_aware_motivation_all.csv
```

Additional multi-seed CoPA outputs when `--run_copa` is used:

```text
copa_delta_summary.csv
copa_lambda_test_delta_summary.csv
copa_lambda_delta_macro_f1_curve.png
copa_delta_macro_f1_detailed.png
copa_high_d_reliability_delta_detailed.png
copa_relation_state_delta_detailed.png
copa_relation_state_delta_summary.csv
```

Detailed plots use the same visual grammar:

```text
bar height          = multi-seed mean delta
error bar           = 95% confidence interval when available
black dots          = individual seed results
n                   = valid seed count for that group
+80% / -80%         = same-sign seed ratio
EC                  = passes the configured error-control rule
```

`relation_state_method_comparison_heatmap.png` compares UncondAlign,
DirectAdd, and CoPA across RA/UA/Mid-D/RD/ND. Each cell reports the mean
delta, 95% CI, sign consistency, and `*` when the row passes error control.

## Smoke Test

```powershell
python code/disagreement_phenomenon/scripts/smoke_test.py
python code/disagreement_phenomenon/scripts/smoke_test_multi_seed.py
```
