from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_pkl_to_npz import convert  # noqa: E402


MULTIBENCH_FOLDERS = {
    "mosi": {
        "drive_url": "https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing",
        "folder": "mosi",
        "filename": "mosi_aligned.npz",
    },
    "mosei": {
        "drive_url": "https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv?usp=sharing",
        "folder": "mosei",
        "filename": "mosei_aligned.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MultiBench CMU-MOSI/MOSEI pickle files and convert them "
            "to the .npz format expected by run_phenomenon.py."
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
        help=(
            "Shared MultiBench root for source files and converted .npz files. "
            "Both use data_root/<dataset>."
        ),
    )
    parser.add_argument(
        "--download_root",
        type=Path,
        default=None,
        help="Optional override for source files. Defaults to data_root.",
    )
    parser.add_argument(
        "--variant",
        choices=["auto", "raw", "data", "senti"],
        default="auto",
        help=(
            "Which MultiBench pickle to convert when multiple files exist. "
            "'auto' uses mosi_raw.pkl for MOSI and mosei_senti_data.pkl for "
            "MOSEI; 'raw' uses *_raw.pkl; 'data' uses *_data.pkl; 'senti' "
            "uses *_senti_data.pkl. Default: auto."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported .npz files.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Use already downloaded files under download_root.",
    )
    return parser.parse_args()


def import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Cannot import gdown.\n"
            "Install it first:\n"
            "  python -m pip install gdown\n"
            "Then re-run this script."
        ) from exc
    return gdown


def download_folder(dataset_name: str, destination: Path) -> None:
    gdown = import_gdown()
    destination.mkdir(parents=True, exist_ok=True)
    url = MULTIBENCH_FOLDERS[dataset_name]["drive_url"]
    print(f"[download] MultiBench {dataset_name}: {url}")
    print(f"           -> {destination}")
    gdown.download_folder(
        url=url,
        output=str(destination),
        quiet=False,
        use_cookies=False,
        resume=True,
    )


def find_pickle(dataset_name: str, download_dir: Path, variant: str) -> Path:
    candidates = sorted(download_dir.rglob("*.pkl"))
    if not candidates:
        found_files = sorted(path.name for path in download_dir.rglob("*") if path.is_file())
        hint = ""
        if any(name.endswith(".part") for name in found_files):
            hint = (
                "\nFound .part files, which usually means the Google Drive "
                "download is incomplete. Finish the download, then re-run "
                "with --skip_download."
            )
        found = "\n  ".join(found_files[:30]) if found_files else "(no files)"
        raise FileNotFoundError(
            f"No .pkl file found under {download_dir}.{hint}\n"
            f"Files seen:\n  {found}"
        )

    preferred_names = {
        "raw": [f"{dataset_name}_raw.pkl"],
        "data": [f"{dataset_name}_data.pkl", f"{dataset_name}_senti_data.pkl"],
        "senti": [f"{dataset_name}_senti_data.pkl", f"{dataset_name}_data.pkl"],
    }.get(variant)
    if preferred_names is not None:
        preferred_names = [name.lower() for name in preferred_names]
        exact = [path for path in candidates if path.name.lower() in preferred_names]
        if exact:
            return sorted(exact, key=lambda path: preferred_names.index(path.name.lower()))[0]
        found = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not find any of {', '.join(preferred_names)} under {download_dir}.\n"
            f"Available .pkl files:\n  {found}"
        )

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if dataset_name == "mosi" and name == "mosi_raw.pkl":
            return (0, name)
        if dataset_name == "mosei" and name == "mosei_senti_data.pkl":
            return (0, name)
        if name == f"{dataset_name}_raw.pkl":
            return (1, name)
        if name in {f"{dataset_name}_data.pkl", f"{dataset_name}_senti_data.pkl"}:
            return (2, name)
        if dataset_name in name and "raw" in name:
            return (3, name)
        if dataset_name in name:
            return (4, name)
        return (5, name)

    ranked = sorted(candidates, key=score)
    best = ranked[0]
    if score(best)[0] == 3 and len(ranked) > 1:
        found = "\n  ".join(str(path) for path in ranked)
        raise RuntimeError(
            f"Found multiple .pkl files under {download_dir}, but none clearly "
            f"matches dataset '{dataset_name}':\n  {found}"
        )
    return best


def output_path(dataset_name: str, data_root: Path) -> Path:
    spec = MULTIBENCH_FOLDERS[dataset_name]
    return data_root / spec["folder"] / spec["filename"]


def prepare_one(
    dataset_name: str,
    *,
    data_root: Path,
    download_root: Path,
    variant: str,
    overwrite: bool,
    skip_download: bool,
) -> Path:
    download_dir = download_root / dataset_name
    if not skip_download:
        download_folder(dataset_name, download_dir)

    input_path = find_pickle(dataset_name, download_dir, variant)
    out_path = output_path(dataset_name, data_root)
    print(f"[convert] {input_path}")
    print(f"          -> {out_path}")
    convert(input_path, out_path, overwrite=overwrite)
    return out_path


def main() -> int:
    args = parse_args()
    download_root = args.download_root or args.data_root
    targets = ["mosi", "mosei"] if args.dataset == "both" else [args.dataset]
    for dataset_name in targets:
        prepare_one(
            dataset_name,
            data_root=args.data_root,
            download_root=download_root,
            variant=args.variant,
            overwrite=args.overwrite,
            skip_download=args.skip_download,
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
