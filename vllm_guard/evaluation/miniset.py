import argparse
import random
from pathlib import Path
from typing import Optional, Sequence

from datasets import Dataset, load_dataset, load_from_disk

from vllm_guard.common.constants import CANONICAL_DATASET_DIR


def _load_split(root: Path, split: str) -> Dataset:
    split_dir = root / split
    split_parquet = root / f"{split}.parquet"
    if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
        return load_from_disk(str(split_dir))
    return load_dataset("parquet", data_files=str(split_parquet), split="train")


def create_minisets(source_dir: str = str(CANONICAL_DATASET_DIR), sample_size: int = 100, seed: int = 42) -> list[str]:
    random.seed(seed)
    root = Path(source_dir)
    outputs = []
    for split in ("id_test", "ood_test"):
        output_file = root / f"{split}_mini.parquet"
        if output_file.exists():
            try:
                load_dataset("parquet", data_files=str(output_file), split="train")
                outputs.append(str(output_file))
                continue
            except Exception:
                pass
        dataset = _load_split(root, split)
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        mini = dataset.select(indices)
        tmp_file = output_file.with_suffix(".parquet.tmp")
        mini.to_parquet(str(tmp_file))
        tmp_file.replace(output_file)
        outputs.append(str(output_file))
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Create mini eval parquet files for training-time smoke evaluation")
    parser.add_argument("source_dir", nargs="?", default=str(CANONICAL_DATASET_DIR))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    outputs = create_minisets(args.source_dir, args.sample_size, args.seed)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
