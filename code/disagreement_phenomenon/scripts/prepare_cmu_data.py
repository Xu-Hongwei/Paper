from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def avg(intervals: np.ndarray, features: np.ndarray) -> np.ndarray:
    del intervals
    return np.asarray(features, dtype=np.float32).mean(axis=0)


DATASET_SPECS = {
    "mosi": {
        "folder": "mosi",
        "filename": "mosi_aligned.npz",
        "sdk_name": "cmu_mosi",
        "label_key": "Opinion Segment Labels",
        "text_key": "glove_vectors",
        "vision_key": "OpenFace_2",
        "audio_key": "COVAREP",
    },
    "mosei": {
        "folder": "mosei",
        "filename": "mosei_aligned.npz",
        "sdk_name": "cmu_mosei",
        "label_key": "All Labels",
        "text_key": "glove_vectors",
        "vision_key": "OpenFace_2",
        "audio_key": "COVAREP",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CMU-MOSI/MOSEI with CMU Multimodal SDK and export the "
            ".npz files expected by run_phenomenon.py."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["mosi", "mosei", "both"],
        default="both",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"E:\Xu\data\MultiBench"),
        help="Download/export root MultiBench directory.",
    )
    parser.add_argument(
        "--label_index",
        type=int,
        default=0,
        help="Column in SDK label features to use as continuous sentiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported .npz files.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download raw .csd files even if local files already exist.",
    )
    parser.add_argument(
        "--download_retries",
        type=int,
        default=8,
        help="Retry count for each raw .csd download.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds for raw .csd downloads.",
    )
    parser.add_argument(
        "--no_proxy",
        action="store_true",
        help="Ignore system/environment proxy settings for dataset downloads.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Explicit proxy URL, e.g. http://127.0.0.1:7890.",
    )
    return parser.parse_args()


def import_sdk():
    try:
        from mmsdk import mmdatasdk
    except ImportError as exc:
        raise SystemExit(
            "Cannot import CMU Multimodal SDK.\n"
            "Install it in the base environment first:\n"
            "  python -m pip install h5py\n"
            "  python -m pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git"
        ) from exc
    return mmdatasdk


def get_fold_module(sdk_dataset):
    folds = sdk_dataset.standard_folds
    return {
        "train": folds.standard_train_fold,
        "valid": folds.standard_valid_fold,
        "test": folds.standard_test_fold,
    }


def clean_features(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array[:, None]
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def clean_labels(array: np.ndarray, label_index: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        labels = array
    else:
        squeezed = np.squeeze(array)
        if squeezed.ndim == 1:
            labels = squeezed
        else:
            labels = squeezed[:, label_index]
    return np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def recipe_for(sdk_dataset, spec: dict[str, str]) -> dict[str, str]:
    recipe = {
        spec["text_key"]: sdk_dataset.highlevel[spec["text_key"]],
        spec["vision_key"]: sdk_dataset.highlevel[spec["vision_key"]],
        spec["audio_key"]: sdk_dataset.highlevel[spec["audio_key"]],
        spec["label_key"]: sdk_dataset.labels[spec["label_key"]],
    }
    return recipe


def make_session(
    download_retries: int,
    *,
    no_proxy: bool,
    proxy: str | None,
) -> requests.Session:
    retry = Retry(
        total=download_retries,
        connect=download_retries,
        read=download_retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.trust_env = not no_proxy
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "CMU-MultimodalSDK-data-prep/1.0"
            )
        }
    )
    return session


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name.endswith(".csd"):
        raise ValueError(f"Cannot infer .csd filename from URL: {url}")
    return name


def download_csd(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: int,
    force_download: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force_download:
        print(f"[cached] {destination}")
        return destination

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {url}")
    print(f"           -> {destination}")
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or 0)
        with open(tmp_path, "wb") as handle:
            progress = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=destination.name,
                leave=False,
            )
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    progress.update(len(chunk))
            finally:
                progress.close()

    written = tmp_path.stat().st_size
    if total > 0 and written != total:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete download for {url}: got {written} bytes, expected {total} bytes."
        )

    with tmp_path.open("rb") as handle:
        magic = handle.read(len(HDF5_MAGIC))
    if magic != HDF5_MAGIC:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file from {url} is not a valid HDF5/CSD file. "
            "The server may have returned an error page or proxy response."
        )

    tmp_path.replace(destination)
    return destination


def localize_recipe(
    recipe: dict[str, str],
    raw_dir: Path,
    *,
    force_download: bool,
    download_retries: int,
    timeout: int,
    no_proxy: bool,
    proxy: str | None,
) -> dict[str, str]:
    session = make_session(download_retries, no_proxy=no_proxy, proxy=proxy)
    localized = {}
    for key, url in recipe.items():
        filename = filename_from_url(url)
        local_path = raw_dir / filename
        try:
            localized[key] = str(
                download_csd(
                    session,
                    url,
                    local_path,
                    timeout=timeout,
                    force_download=force_download,
                )
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to download {key} from {url}\n"
                "This is usually a CMU data-server/network issue. "
                "Re-run the same command later; already downloaded .csd files "
                "will be reused. For Experiment 1, the more reliable route is "
                "to use MultiBench processed pickles via prepare_multibench_data.py.\n"
                f"Original error: {exc}"
            ) from exc
    return localized


def prepare_one(
    dataset_name: str,
    data_root: Path,
    label_index: int,
    overwrite: bool,
    force_download: bool,
    download_retries: int,
    timeout: int,
    no_proxy: bool,
    proxy: str | None,
) -> Path:
    mmdatasdk = import_sdk()
    spec = DATASET_SPECS[dataset_name]
    sdk_dataset = getattr(mmdatasdk, spec["sdk_name"])
    dataset_dir = data_root / spec["folder"]
    raw_dir = dataset_dir / "raw_csd"
    out_path = dataset_dir / spec["filename"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists. Use --overwrite to rebuild it.")
        return out_path

    recipe = localize_recipe(
        recipe_for(sdk_dataset, spec),
        raw_dir,
        force_download=force_download,
        download_retries=download_retries,
        timeout=timeout,
        no_proxy=no_proxy,
        proxy=proxy,
    )
    print(f"[{dataset_name}] loading SDK computational sequences from local .csd files...")
    print(f"[{dataset_name}] raw CSD directory: {raw_dir}")
    dataset = mmdatasdk.mmdataset(recipe)

    label_key = spec["label_key"]
    print(f"[{dataset_name}] aligning all modalities to label segments: {label_key}")
    dataset.align(label_key, collapse_functions=[avg])
    dataset.impute(label_key)
    dataset.hard_unify()

    folds = get_fold_module(sdk_dataset)
    print(f"[{dataset_name}] exporting train/valid/test arrays...")
    tensors = dataset.get_tensors(
        seq_len=1,
        non_sequences=[
            spec["text_key"],
            spec["vision_key"],
            spec["audio_key"],
            spec["label_key"],
        ],
        folds=[folds["train"], folds["valid"], folds["test"]],
    )

    exported: dict[str, np.ndarray] = {}
    split_names = ["train", "valid", "test"]
    for split_name, split_data in zip(split_names, tensors):
        exported[f"{split_name}_text"] = clean_features(split_data[spec["text_key"]])
        exported[f"{split_name}_vision"] = clean_features(split_data[spec["vision_key"]])
        exported[f"{split_name}_audio"] = clean_features(split_data[spec["audio_key"]])
        exported[f"{split_name}_label"] = clean_labels(
            split_data[spec["label_key"]],
            label_index=label_index,
        )
        n = exported[f"{split_name}_label"].shape[0]
        print(
            f"  {split_name}: n={n}, "
            f"text={exported[f'{split_name}_text'].shape}, "
            f"vision={exported[f'{split_name}_vision'].shape}, "
            f"audio={exported[f'{split_name}_audio'].shape}"
        )

    np.savez_compressed(out_path, **exported)
    print(f"[{dataset_name}] saved: {out_path}")
    return out_path


def main() -> int:
    args = parse_args()
    targets = ["mosi", "mosei"] if args.dataset == "both" else [args.dataset]
    for name in targets:
        prepare_one(
            dataset_name=name,
            data_root=args.data_root,
            label_index=args.label_index,
            overwrite=args.overwrite,
            force_download=args.force_download,
            download_retries=args.download_retries,
            timeout=args.timeout,
            no_proxy=args.no_proxy,
            proxy=args.proxy,
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
