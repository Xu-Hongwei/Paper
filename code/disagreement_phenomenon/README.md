# Disagreement Phenomenon Experiment

This package implements the first CoPA motivation experiment:

> paired multimodal samples are not always alignment-positive.

The first version compares a plain concat fusion model against an unconditional
sample-level alignment model on Low/Mid/High disagreement groups.

## Expected Data

The runner expects one `.npz` file per dataset:

```text
E:\Xu\data\CMU_MOSI\mosi_aligned.npz
E:\Xu\data\CMU_MOSEI\mosei_aligned.npz
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

```powershell
python code/disagreement_phenomenon/scripts/run_phenomenon.py --dataset mosi --data_root E:\Xu\data
python code/disagreement_phenomenon/scripts/run_phenomenon.py --dataset mosei --data_root E:\Xu\data
```

## Prepare Data

After installing the CMU Multimodal SDK:

```powershell
python -m pip install h5py
python -m pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
```

download and export the aligned `.npz` files:

```powershell
python code/disagreement_phenomenon/scripts/prepare_cmu_data.py --dataset both --data_root E:\Xu\data
```

This creates:

```text
E:\Xu\data\CMU_MOSI\mosi_aligned.npz
E:\Xu\data\CMU_MOSEI\mosei_aligned.npz
```

Outputs are written to:

```text
code/disagreement_phenomenon/outputs/<dataset>/<timestamp>/
```

## Smoke Test

```powershell
python code/disagreement_phenomenon/scripts/smoke_test.py
```
