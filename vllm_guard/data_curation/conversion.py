import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from datasets import Dataset, Features, Image, Value, load_dataset, load_from_disk
from tqdm import tqdm


DEFAULT_FEATURES = Features(
    {
        "question": Value("string"),
        "answer": Value("string"),
        "image": Image(),
        "image_idx": Value("int64"),
        "section_id": Value("int64"),
        "section_title": Value("string"),
        "policy_name": Value("string"),
        "tier": Value("string"),
        "discrimination_score": Value("float64"),
        "policy_description": Value("string"),
        "label": Value("string"),
        "split_type": Value("string"),
        "violated_categories": [Value("int64")],
        "reason": Value("string"),
        "reason_source": Value("string"),
        "target_text": Value("string"),
    }
)


def load_image_dataset(image_path: str):
    path = Path(image_path)
    if path.is_file() and path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    if (path / "dataset_info.json").exists():
        return load_from_disk(str(path))
    return load_dataset(str(path), split="train")


def attach_images_from_jsonl(jsonl_path: str, image_dataset) -> list[dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Loading jsonl"):
            item = json.loads(line)
            image_idx = int(item["image_idx"])
            if image_idx >= len(image_dataset):
                continue
            item["image"] = image_dataset[image_idx]["image"]
            rows.append(item)
    return rows


def convert_jsonl_to_hf(jsonl_path: str, image_dataset_path: str, output_path: str) -> str:
    image_dataset = load_image_dataset(image_dataset_path)
    rows = attach_images_from_jsonl(jsonl_path, image_dataset)
    dataset = Dataset.from_list(rows, features=DEFAULT_FEATURES)
    dataset.to_parquet(output_path)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Convert jsonl dataset to HF/parquet with attached images")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--image-dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    convert_jsonl_to_hf(args.jsonl, args.image_dataset, args.output)


if __name__ == "__main__":
    main()
