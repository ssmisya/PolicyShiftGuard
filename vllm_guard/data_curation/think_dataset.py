import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

from datasets import Dataset, load_dataset, load_from_disk


def load_split(dataset_path: str, split: str):
    split_dir = Path(dataset_path) / split
    split_parquet = Path(dataset_path) / f"{split}.parquet"
    if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
        return load_from_disk(str(split_dir))
    if split_parquet.exists():
        return load_dataset("parquet", data_files=str(split_parquet), split="train")
    raise FileNotFoundError(f"Cannot find split {split} under {dataset_path}")


def build_target_text(row: dict) -> str:
    reason = (row.get("reason") or "").strip()
    if row["label"] == "block":
        violated = row.get("violated_categories") or []
        category = f"{violated[0]:02d}" if violated else f"{int(row['section_id']):02d}"
        answer = f"true | {category}"
    else:
        answer = "false"
    return f"<think>{reason}</think> {answer}"


def build_sft_think_dataset(dataset_path: str, split: str, output_dir: str, output_split: str = "sft_think") -> str:
    source = load_split(dataset_path, split)

    def iter_rows():
        for row in source:
            yield {
                "question": row["question"],
                "image": row["image"],
                "target_text": build_target_text(row),
                "reason": row.get("reason", ""),
                "answer": row.get("answer", ""),
                "label": row["label"],
                "section_id": row["section_id"],
                "section_title": row.get("section_title", ""),
                "policy_name": row.get("policy_name", ""),
                "violated_categories": row.get("violated_categories", []),
                "image_idx": row.get("image_idx", -1),
                "reason_source": row.get("reason_source", "unknown"),
            }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / output_split
    tmp_dir = out_dir / f"{output_split}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    dataset = Dataset.from_generator(iter_rows)
    dataset.save_to_disk(str(tmp_dir))
    if split_dir.exists():
        shutil.rmtree(split_dir)
    tmp_dir.rename(split_dir)
    dataset.to_parquet(str(out_dir / f"{output_split}.parquet"))
    with open(out_dir / f"{output_split}.jsonl", "w", encoding="utf-8") as handle:
        for row in dataset:
            payload = {key: value for key, value in row.items() if key != "image"}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(split_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Build sft_think split from reason-annotated sft data")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", default="sft")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-split", default="sft_think")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    build_sft_think_dataset(args.dataset_path, args.split, args.output_dir, args.output_split)


if __name__ == "__main__":
    main()
