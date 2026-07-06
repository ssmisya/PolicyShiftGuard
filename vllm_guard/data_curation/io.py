import io
import json
from pathlib import Path


def load_jsonl(path: str) -> dict[int, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                data[rec["idx"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return data


def load_mapping(path: str) -> dict[int, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data["shuffled_to_original"].items()}


def remap_original_to_shuffled(original_order_data: dict[int, dict], shuffled_to_original: dict[int, int]) -> dict[int, dict]:
    original_to_shuffled = {orig: shuffled for shuffled, orig in shuffled_to_original.items()}
    return {
        shuffled_idx: rec
        for original_idx, rec in original_order_data.items()
        if (shuffled_idx := original_to_shuffled.get(original_idx)) is not None
    }


def load_rules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ImageLoader:
    def __init__(self, source: str, source_type: str = "hf_dataset", split: str = "train"):
        self.source_type = source_type
        if source_type == "hf_dataset":
            from datasets import load_dataset, load_from_disk
            p = Path(source)
            if p.is_file() and p.suffix == ".parquet":
                self.dataset = load_dataset("parquet", data_files=str(p), split="train")
            elif p.is_dir() and (p / "dataset_info.json").exists():
                self.dataset = load_from_disk(str(p))
            else:
                self.dataset = load_dataset(source, split=split)
        elif source_type == "directory":
            self.image_dir = Path(source)

    def get_pil_image(self, image_idx: int):
        from PIL import Image

        if self.source_type == "hf_dataset":
            item = self.dataset[int(image_idx)]
            img = item.get("image")
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            if isinstance(img, dict):
                if "bytes" in img and img["bytes"] is not None:
                    return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                if "path" in img and img["path"]:
                    return Image.open(img["path"]).convert("RGB")
            if isinstance(img, bytes):
                return Image.open(io.BytesIO(img)).convert("RGB")
            raise ValueError(f"cannot decode image for index {image_idx}")
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            path = self.image_dir / f"{image_idx}{ext}"
            if path.exists():
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"image {image_idx} not found in {self.image_dir}")

