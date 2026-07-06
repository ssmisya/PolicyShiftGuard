import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


BASE_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "chat_template.json",
]

CKPT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
]


def merge_checkpoint_for_vllm(ckpt_path: str, base_model_path: str, output_path: Optional[str] = None, in_place: bool = False) -> str:
    ckpt = Path(ckpt_path)
    base = Path(base_model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")
    if not base.exists():
        raise FileNotFoundError(f"base model not found: {base}")

    actual_in_place = in_place or output_path is None
    output = ckpt if actual_in_place else Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    for name in BASE_FILES:
        src = base / name
        if src.exists():
            shutil.copy2(src, output / name)

    if not actual_in_place:
        for name in CKPT_FILES:
            src = ckpt / name
            if src.exists():
                shutil.copy2(src, output / name)
        for part in ckpt.glob("model-*.safetensors"):
            shutil.copy2(part, output / part.name)

    ckpt_name = ckpt.name if ckpt.name.startswith("checkpoint-") else "final"
    with open(output / "merge_info.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_ckpt": str(ckpt.resolve()),
                "ckpt_name": ckpt_name,
                "base_model": str(base.resolve()),
                "merged_at": datetime.now().isoformat(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return str(output)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Merge a training checkpoint into a vLLM-loadable HF directory")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = merge_checkpoint_for_vllm(args.ckpt, args.base, args.output, args.in_place)
    print(f"Model ready at: {out}")


if __name__ == "__main__":
    main()
