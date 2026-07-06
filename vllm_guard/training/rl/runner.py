import argparse
from dataclasses import dataclass
from typing import Sequence

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.training.rl.pipeline import main as pipeline_main


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

    def to_cli_args(self) -> list[str]:
        args = [
            "--dataset-path", self.dataset_path,
            "--train-split", self.train_split,
            "--val-splits", ",".join(self.val_splits),
            "--output-dir", self.output_dir,
            "--response-format", self.response_format,
            "--policy-rephrase-seed", str(self.policy_rephrase_seed),
            "--max-train-samples", str(self.max_train_samples),
            "--max-val-samples", str(self.max_val_samples),
        ]
        if self.pair_pack_train:
            args.append("--pair-pack-train")
        if self.policy_rephrase_path:
            args.extend(["--policy-rephrase-path", self.policy_rephrase_path])
        return args


def run_prepare_verl(config: RLPrepareConfig) -> None:
    pipeline_main(config.to_cli_args())


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
    run_prepare_verl(parse_args(argv))


if __name__ == "__main__":
    main()
