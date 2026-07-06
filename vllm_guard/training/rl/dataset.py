import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Image as HFImage, Sequence, load_dataset, load_from_disk

from vllm_guard.training.rl.formatting import get_output_instructions, normalize_response_format
from vllm_guard.training.formatting import randomize_policy_descriptions


def load_split(dataset_path: str, split_name: str) -> Dataset:
    root = Path(dataset_path)
    split_dir = root / split_name
    split_parquet = root / f"{split_name}.parquet"
    if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
        return load_from_disk(str(split_dir))
    if split_parquet.exists():
        return load_dataset("parquet", data_files=str(split_parquet), split="train")
    raise FileNotFoundError(f"Could not find split '{split_name}' under {dataset_path}")


def normalize_label(label: str) -> str:
    if label in {"block", "unsafe", "true"}:
        return "true"
    if label in {"pass", "safe", "false"}:
        return "false"
    raise ValueError(f"Unsupported label: {label}")


def normalize_image(image_obj: Any):
    from PIL import Image

    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return {"bytes": bytes(image_obj["bytes"]), "path": None}
        if image_obj.get("path"):
            return {"bytes": None, "path": image_obj["path"]}
    if isinstance(image_obj, bytes):
        return {"bytes": bytes(image_obj), "path": None}
    raise TypeError(f"Unsupported image object type: {type(image_obj)}")


def build_ground_truth(example: dict[str, Any]) -> str:
    label = normalize_label(example["label"])
    categories = [f"{int(cat):02d}" for cat in (example.get("violated_categories", []) or [])]
    return json.dumps({"label": label, "accepted_category_ids": categories}, ensure_ascii=False, sort_keys=True)


def build_prompt(question: str, response_format: str = "think") -> list[dict[str, str]]:
    return [{"role": "user", "content": f"<image>\n{question}{get_output_instructions(response_format)}"}]


def build_row(
    example: dict[str, Any],
    *,
    split_name: str,
    index: int,
    dataset_name: str,
    response_format: str = "think",
    policy_rephrase_path: str | None = None,
    policy_rephrase_seed: int = 0,
) -> dict[str, Any]:
    label = normalize_label(example["label"])
    violated_categories = [int(x) for x in (example.get("violated_categories", []) or [])]
    answer = "false" if label == "false" else f"true | {violated_categories[0]:02d}"
    example_key = "|".join(
        str(example.get(name, ""))
        for name in ("image_idx", "section_id", "policy_name")
    )
    question = randomize_policy_descriptions(
        example["question"],
        rephrase_path=policy_rephrase_path,
        seed=policy_rephrase_seed,
        example_key=example_key,
    )
    return {
        # verl validation aggregates by data_source, so split identity must be preserved here.
        "data_source": f"adaptive_policy/{dataset_name}/{split_name}",
        "prompt": build_prompt(question, response_format=response_format),
        "images": [normalize_image(example["image"])],
        "reward_model": {"style": "rule", "ground_truth": build_ground_truth(example)},
        "extra_info": {
            "split": split_name,
            "index": index,
            "dataset_name": dataset_name,
            "response_format": response_format,
            "policy_rephrase_enabled": bool(policy_rephrase_path),
            "policy_rephrase_seed": policy_rephrase_seed if policy_rephrase_path else None,
            "image_idx": int(example["image_idx"]),
            "question": question,
            "original_question": example["question"] if policy_rephrase_path else "",
            "answer": answer,
            "label": label,
            "policy_name": example.get("policy_name", ""),
            "section_id": int(example.get("section_id", -1)),
            "violated_categories": violated_categories,
            "tier": example.get("tier", ""),
            "policy_source": example.get("policy_source", ""),
            "target_policy_label": example.get("target_policy_label", ""),
            "boundary_group_id": example.get("boundary_group_id", ""),
            "boundary_pair_role": example.get("boundary_pair_role", ""),
        },
    }


def _pair_pack_indices(source: Dataset, limit: int) -> list[int]:
    role_order = {"block": 0, "unsafe": 0, "true": 0, "pass": 1, "safe": 1, "false": 1}

    def sort_key(idx: int):
        row = source[idx]
        group_id = str(row.get("boundary_group_id") or "")
        role = str(row.get("boundary_pair_role") or row.get("label") or "").lower()
        return (group_id, role_order.get(role, 9), idx)

    indices = sorted(range(len(source)), key=sort_key)
    return indices[:limit]


def convert_split(
    dataset_path: str,
    split_name: str,
    max_samples: int = -1,
    response_format: str = "think",
    policy_rephrase_path: str | None = None,
    policy_rephrase_seed: int = 0,
    pair_pack: bool = False,
) -> Dataset:
    source = load_split(dataset_path, split_name)
    response_format = normalize_response_format(response_format)
    dataset_name = Path(dataset_path).name
    rows = []
    limit = len(source) if max_samples < 0 else min(len(source), max_samples)
    indices = _pair_pack_indices(source, limit) if pair_pack else list(range(limit))
    for output_idx, idx in enumerate(indices):
        rows.append(
            build_row(
                source[idx],
                split_name=split_name,
                index=output_idx,
                dataset_name=dataset_name,
                response_format=response_format,
                policy_rephrase_path=policy_rephrase_path,
                policy_rephrase_seed=policy_rephrase_seed,
            )
        )
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("images", Sequence(HFImage()))
