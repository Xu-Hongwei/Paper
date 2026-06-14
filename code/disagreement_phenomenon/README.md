# Disagreement Phenomenon Experiment

This package implements the first CoPA motivation experiment:

> paired multimodal samples are not always alignment-positive.

The first version compares a plain concat fusion model against an unconditional
sample-level alignment model on Low/Mid/High disagreement groups.

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

```powershell
python code/disagreement_phenomenon/scripts/run_phenomenon.py --dataset mosi --data_root E:\Xu\data\MultiBench
python code/disagreement_phenomenon/scripts/run_phenomenon.py --dataset mosei --data_root E:\Xu\data\MultiBench
```

For the MOSI multi-seed version of Experiment 1:

```powershell
python code/disagreement_phenomenon/scripts/run_multi_seed.py --dataset mosi --data_root E:\Xu\data\MultiBench --seeds 1 2 3 4 5
```

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

## Smoke Test

```powershell
python code/disagreement_phenomenon/scripts/smoke_test.py
```
