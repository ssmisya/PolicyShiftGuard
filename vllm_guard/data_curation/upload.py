import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datasets import DatasetDict, Features, load_from_disk
from huggingface_hub import HfApi

from vllm_guard.common.constants import CANONICAL_DATASET_DIR, CANONICAL_HF_DATASET_REPO


@dataclass(frozen=True)
class UploadConfig:
    dataset_dir: str = str(CANONICAL_DATASET_DIR)
    repo_id: str = CANONICAL_HF_DATASET_REPO
    token: str = ""
    private: bool = True
    splits: tuple[str, ...] = ("id_test", "ood_test", "sft", "sft_think", "rl")
    extra_files: tuple[str, ...] = ()


def load_dataset_dict(dataset_dir: Path, splits: tuple[str, ...]) -> DatasetDict:
    loaded = {}
    for split_name in splits:
        split_dir = dataset_dir / split_name
        if not (split_dir / "dataset_info.json").exists():
            continue
        loaded[split_name] = load_from_disk(str(split_dir))
    if not loaded:
        raise FileNotFoundError(f"No valid split directories found under {dataset_dir}")
    return DatasetDict(loaded)


def validate_upload_layout(dataset_dir: Path, splits: tuple[str, ...]) -> dict[str, dict[str, int | bool]]:
    summary: dict[str, dict[str, int | bool]] = {}
    for split_name in splits:
        split_dir = dataset_dir / split_name
        if not (split_dir / "dataset_info.json").exists():
            continue
        ds = load_from_disk(str(split_dir))
        summary[split_name] = {
            "rows": len(ds),
            "has_reason": "reason" in ds.column_names,
            "has_reason_source": "reason_source" in ds.column_names,
            "has_target_text": "target_text" in ds.column_names,
        }
    return summary


def align_dataset_dict_features(dataset_dict: DatasetDict) -> DatasetDict:
    """Return an upload-only DatasetDict with a union schema across splits.

    HF dataset repos require all splits to expose the same features. Locally we
    keep no-think splits without reason columns, but for upload we add missing
    columns as null values so the semantic payload remains unchanged.
    """

    union: dict[str, object] = {}
    for dataset in dataset_dict.values():
        for column_name, feature in dataset.features.items():
            if column_name not in union:
                union[column_name] = feature

    features = Features(union)
    aligned = {}
    for split_name, dataset in dataset_dict.items():
        for column_name in union:
            if column_name not in dataset.column_names:
                dataset = dataset.add_column(column_name, [None] * len(dataset))
        aligned[split_name] = dataset.select_columns(list(union)).cast(features)
    return DatasetDict(aligned)


def push_dataset(config: UploadConfig) -> dict[str, object]:
    dataset_dir = Path(config.dataset_dir)
    dataset_dict = None
    summary = {}
    upload_mode = "none"
    if config.splits:
        dataset_dict = load_dataset_dict(dataset_dir, config.splits)
        summary = validate_upload_layout(dataset_dir, config.splits)
        try:
            dataset_dict.push_to_hub(config.repo_id, token=config.token, private=config.private)
            upload_mode = "dataset_dict"
        except ValueError as exc:
            if "same features" not in str(exc):
                raise
            dataset_dict = align_dataset_dict_features(dataset_dict)
            dataset_dict.push_to_hub(config.repo_id, token=config.token, private=config.private)
            upload_mode = "aligned_dataset_dict"
    uploaded_extra_files: list[str] = []
    if config.extra_files:
        api = HfApi(token=config.token)
        for rel_path in config.extra_files:
            local_path = dataset_dir / rel_path
            if not local_path.exists():
                continue
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel_path,
                repo_id=config.repo_id,
                repo_type="dataset",
            )
            uploaded_extra_files.append(rel_path)
    return {
        "dataset_dir": str(dataset_dir),
        "repo_id": config.repo_id,
        "private": config.private,
        "upload_mode": upload_mode,
        "splits": list(dataset_dict.keys()) if dataset_dict is not None else [],
        "layout_summary": summary,
        "uploaded_extra_files": uploaded_extra_files,
    }


def parse_args(argv: Sequence[str] | None = None) -> UploadConfig:
    p = argparse.ArgumentParser(description="Canonical adaptive-policy HF upload entrypoint")
    p.add_argument("--dataset-dir", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--repo-id", default=CANONICAL_HF_DATASET_REPO)
    p.add_argument("--token", required=True)
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")
    p.add_argument("--splits", nargs="*", default=["id_test", "ood_test", "sft", "sft_think", "rl"])
    p.add_argument("--extra-files", nargs="*", default=[])
    args = p.parse_args(argv)
    return UploadConfig(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        splits=tuple(args.splits),
        extra_files=tuple(args.extra_files),
    )


def main(argv: Sequence[str] | None = None) -> None:
    summary = push_dataset(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
