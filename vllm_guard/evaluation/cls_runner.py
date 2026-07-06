import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from datasets import load_from_disk
from tqdm import tqdm

from vllm_guard.evaluation.metrics import compute_all_metrics
from vllm_guard.evaluation.tables import save_tables
from vllm_guard.training.cls.inference import BinaryClassifierInference


def parse_label(label_str):
    if isinstance(label_str, str):
        return "block" if label_str.lower() in {"block", "true", "unsafe"} else "pass"
    return label_str


def run_cls_evaluation(model_path: str, dataset_path: str, split: str, output_dir: str) -> str:
    model = BinaryClassifierInference(model_path)
    dataset_root = Path(dataset_path)
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split} not found in {dataset_root}")
    dataset = load_from_disk(str(split_dir))

    predictions = []
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating CLS")):
        result = model.predict(example["image"], example["question"])
        prediction = f"true | {example.get('section_id', '01')}" if result["decision"] == "block" else "false"
        predictions.append(
            {
                "prediction": prediction,
                "label": parse_label(example["label"]),
                "policy_name": example.get("policy_name", "unknown"),
                "image_idx": example.get("image_idx", idx),
                "section_id": example.get("section_id", 1),
                "section_title": example.get("section_title", ""),
                "tier": example.get("tier", ""),
                "violated_categories": example.get("violated_categories", []),
                "predicted_categories": [int(example.get("section_id", 1))] if result["decision"] == "block" else [],
                "confidence": result["confidence"],
            }
        )

    metrics = compute_all_metrics(predictions)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_tables(metrics, str(out_dir))
    return str(out_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate the canonical CLS model on adaptive-policy splits")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", default="id_test", choices=["id_test", "ood_test", "sft", "rl"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_cls_evaluation(args.model_path, args.dataset_path, args.split, args.output_dir)


if __name__ == "__main__":
    main()
