import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.training.rl.dataset import convert_split


@dataclass(frozen=True)
class RLPrepareConfig:
    dataset_path: str = str(CANONICAL_DATASET_DIR)
    train_split: str = "rl"
    val_splits: tuple[str, ...] = ("id_test", "ood_test")
    output_dir: str = ""
    response_format: str = "think"
    policy_rephrase_path: str | None = None
    policy_rephrase_seed: int = 0
    max_train_samples: int = -1
    max_val_samples: int = -1
    pair_pack_train: bool = False


def run_prepare_verl(config: RLPrepareConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RL Prepare] dataset_path={config.dataset_path}", flush=True)
    print(f"[RL Prepare] train_split={config.train_split}", flush=True)
    print(f"[RL Prepare] val_splits={list(config.val_splits)}", flush=True)
    print(f"[RL Prepare] output_dir={output_dir}", flush=True)
    print(f"[RL Prepare] response_format={config.response_format}", flush=True)
    print(f"[RL Prepare] policy_rephrase_path={config.policy_rephrase_path}", flush=True)
    print(f"[RL Prepare] policy_rephrase_seed={config.policy_rephrase_seed}", flush=True)
    print(f"[RL Prepare] pair_pack_train={config.pair_pack_train}", flush=True)

    print(f"[RL Prepare] loading train split: {config.train_split}", flush=True)
    train_ds = convert_split(
        config.dataset_path,
        config.train_split,
        max_samples=config.max_train_samples,
        response_format=config.response_format,
        policy_rephrase_path=config.policy_rephrase_path,
        policy_rephrase_seed=config.policy_rephrase_seed,
        pair_pack=config.pair_pack_train,
    )
    print(f"[RL Prepare] converted train split rows={len(train_ds)}", flush=True)

    train_path = output_dir / "train.parquet"
    print(f"[RL Prepare] writing {train_path}", flush=True)
    train_ds.to_parquet(str(train_path))
    print(f"[RL Prepare] wrote {train_path}", flush=True)

    val_files: dict[str, str] = {}
    total_val_rows = 0
    for split in config.val_splits:
        print(f"[RL Prepare] loading val split: {split}", flush=True)
        val_ds = convert_split(
            config.dataset_path,
            split,
            max_samples=config.max_val_samples,
            response_format=config.response_format,
            policy_rephrase_path=config.policy_rephrase_path,
            policy_rephrase_seed=config.policy_rephrase_seed,
            pair_pack=False,
        )
        val_path = output_dir / f"val_{split}.parquet"
        print(f"[RL Prepare] converted val split={split} rows={len(val_ds)}", flush=True)
        print(f"[RL Prepare] writing {val_path}", flush=True)
        val_ds.to_parquet(str(val_path))
        print(f"[RL Prepare] wrote {val_path}", flush=True)
        val_files[split] = str(val_path)
        total_val_rows += len(val_ds)

    stats = {
        "dataset_path": config.dataset_path,
        "train_split": config.train_split,
        "val_splits": list(config.val_splits),
        "response_format": config.response_format,
        "policy_rephrase_path": config.policy_rephrase_path,
        "policy_rephrase_seed": config.policy_rephrase_seed,
        "pair_pack_train": config.pair_pack_train,
        "train_rows": len(train_ds),
        "val_rows_total": total_val_rows,
        "val_files": val_files,
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[RL Prepare] wrote {output_dir / 'stats.json'}", flush=True)
    print(f"[RL Prepare] done train_rows={len(train_ds)} val_rows_total={total_val_rows}", flush=True)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> RLPrepareConfig:
    p = argparse.ArgumentParser(description="Canonical RL/verl data preparation entrypoint")
    p.add_argument("--dataset-path", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--train-split", default="rl")
    p.add_argument("--val-splits", default="id_test,ood_test")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--response-format", default="think", choices=("think", "nothink"))
    p.add_argument("--policy-rephrase-path", default=None)
    p.add_argument("--policy-rephrase-seed", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=-1)
    p.add_argument("--max-val-samples", type=int, default=-1)
    p.add_argument("--pair-pack-train", action="store_true")
    args = vars(p.parse_args(argv))
    args["val_splits"] = tuple(s.strip() for s in str(args["val_splits"]).split(",") if s.strip())
    return RLPrepareConfig(**args)


def main(argv: Sequence[str] | None = None) -> None:
    stats = run_prepare_verl(parse_args(argv))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
