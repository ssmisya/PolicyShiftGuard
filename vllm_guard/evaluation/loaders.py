import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset, load_from_disk


REQUIRED_METADATA_FIELDS = ["image_idx", "section_id", "section_title", "policy_name", "tier"]


@dataclass
class EvalDataConfig:
    dataset_dir: Optional[str]
    dataset_repo: Optional[str]
    dataset_parquet: Optional[str]
    split: str


def load_jsonl_instances(dataset_dir: str, split: str) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / f"{split}.jsonl"
    instances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def load_hf_instances(dataset_repo: str, split: str) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_repo, split=split)
    return [dict(x) for x in ds]


def load_parquet_instances(dataset_parquet: str, split: str) -> list[dict[str, Any]]:
    p = Path(dataset_parquet)
    if p.is_dir():
        split_file = p / f"{split}.parquet"
        if split_file.exists():
            ds = load_dataset("parquet", data_files=str(split_file), split="train")
            return [dict(x) for x in ds]
        split_dir = p / split
        if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
            ds = load_from_disk(str(split_dir))
            return [dict(x) for x in ds]
        raise FileNotFoundError(f"split '{split}' not found in {p}")
    ds = load_dataset("parquet", data_files=str(p), split="train")
    return [dict(x) for x in ds]


def to_pil_image(image_obj, max_pixels=8_000_000):
    from PIL import Image

    if isinstance(image_obj, Image.Image):
        img = image_obj.convert("RGB")
    elif isinstance(image_obj, dict):
        if "bytes" in image_obj and image_obj["bytes"] is not None:
            img = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        elif "path" in image_obj and image_obj["path"]:
            img = Image.open(image_obj["path"]).convert("RGB")
        else:
            raise ValueError("dict image object missing 'bytes' or 'path'")
    elif isinstance(image_obj, bytes):
        img = Image.open(io.BytesIO(image_obj)).convert("RGB")
    else:
        raise ValueError(f"unsupported image object type: {type(image_obj)}")

    pixels = img.width * img.height
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    return img


def normalize_instance(raw: dict[str, Any], source_desc: str) -> dict[str, Any]:
    inst = dict(raw)
    if "answer" in inst and "label" not in inst:
        inst["label"] = inst["answer"]
    label_map = {"block": "unsafe", "pass": "safe", "unsafe": "unsafe", "safe": "safe"}
    raw_label = inst.get("label")
    if raw_label not in label_map:
        raise ValueError(f"invalid or missing label/answer in {source_desc}; got label={raw_label}")
    inst["label"] = label_map[raw_label]

    if "violated_categories" in inst:
        vc = inst["violated_categories"]
        inst["violated_categories"] = [int(c) for c in vc] if isinstance(vc, list) else []
    elif "answer" in inst and isinstance(inst["answer"], str) and inst["answer"].startswith("true"):
        parts = inst["answer"].split("|")
        inst["violated_categories"] = [int(c.strip()) for c in parts[1].split(",") if c.strip().isdigit()] if len(parts) >= 2 else []
    else:
        inst["violated_categories"] = []

    missing_meta = [k for k in REQUIRED_METADATA_FIELDS if k not in inst]
    if missing_meta:
        raise ValueError(f"missing required metadata fields in {source_desc}: {', '.join(missing_meta)}")

    inst["prompt"] = inst.get("question") or inst.get("policy")
    inst["discrimination_score"] = float(inst.get("discrimination_score", 0.0))
    if "image" in raw and raw["image"] is not None:
        inst["_direct_image"] = raw["image"]
    return inst


def load_instances(config: EvalDataConfig) -> list[dict[str, Any]]:
    if config.dataset_dir:
        raw_instances = load_jsonl_instances(config.dataset_dir, config.split)
        source_desc = f"dataset-dir:{config.dataset_dir}/{config.split}.jsonl"
    elif config.dataset_repo:
        raw_instances = load_hf_instances(config.dataset_repo, config.split)
        source_desc = f"dataset-repo:{config.dataset_repo}:{config.split}"
    else:
        raw_instances = load_parquet_instances(config.dataset_parquet, config.split)
        source_desc = f"dataset-parquet:{config.dataset_parquet}"
    return [normalize_instance(raw, f"{source_desc}[{i}]") for i, raw in enumerate(raw_instances)]


class ImageLoader:
    def __init__(self, source: str, source_type: str = "hf_dataset", split: str = "train"):
        self.source_type = source_type
        if source_type == "hf_dataset":
            p = Path(source)
            if p.is_file() and p.suffix == ".parquet":
                self.dataset = load_dataset("parquet", data_files=str(p), split="train")
            else:
                self.dataset = load_dataset(str(p), split=split)
        elif source_type == "directory":
            self.image_dir = Path(source)
        else:
            raise ValueError("image_source_type must be 'hf_dataset' or 'directory'")

    def get_pil_image(self, image_idx: int):
        from PIL import Image

        if self.source_type == "hf_dataset":
            item = self.dataset[int(image_idx)]
            return to_pil_image(item.get("image"))

        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            path = self.image_dir / f"{image_idx}{ext}"
            if path.exists():
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"Image {image_idx} not found in {self.image_dir}")


def get_instance_image(inst: dict[str, Any], image_loader: Optional[ImageLoader]):
    if "_direct_image" in inst:
        return to_pil_image(inst["_direct_image"])
    if image_loader is None:
        raise ValueError("instance has no direct image column and no image loader")
    return image_loader.get_pil_image(inst["image_idx"])

