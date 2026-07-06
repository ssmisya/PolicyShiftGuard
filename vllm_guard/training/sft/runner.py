import argparse
from dataclasses import dataclass
from typing import Sequence

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.training.sft.pipeline import main as pipeline_main


@dataclass(frozen=True)
class SFTTrainConfig:
    model_name_or_path: str
    output_dir: str
    dataset_path: str = str(CANONICAL_DATASET_DIR)
    train_split: str = "sft"
    no_reason: bool = True
    use_think_tags: bool = False
    policy_rephrase_path: str | None = None
    policy_rephrase_seed: int = 0
    extra_args: tuple[str, ...] = ()

    def to_cli_args(self) -> list[str]:
        args = [
            "--model_name_or_path", self.model_name_or_path,
            "--dataset_path", self.dataset_path,
            "--output_dir", self.output_dir,
            "--train_split", self.train_split,
            "--no_reason", str(self.no_reason),
            "--use_think_tags", str(self.use_think_tags),
            "--policy_rephrase_seed", str(self.policy_rephrase_seed),
        ]
        if self.policy_rephrase_path:
            args.extend(["--policy_rephrase_path", self.policy_rephrase_path])
        args.extend(self.extra_args)
        return args


def run_sft(config: SFTTrainConfig) -> None:
    pipeline_main(config.to_cli_args())


def parse_args(argv: Sequence[str] | None = None) -> SFTTrainConfig:
    p = argparse.ArgumentParser(description="Canonical SFT training entrypoint")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_path", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--train_split", default="sft")
    p.add_argument("--no_reason", default="True")
    p.add_argument("--use_think_tags", default="False")
    p.add_argument("--policy_rephrase_path", default=None)
    p.add_argument("--policy_rephrase_seed", type=int, default=0)
    args, extra = p.parse_known_args(argv)
    return SFTTrainConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        train_split=args.train_split,
        no_reason=str(args.no_reason).lower() == "true",
        use_think_tags=str(args.use_think_tags).lower() == "true",
        policy_rephrase_path=args.policy_rephrase_path,
        policy_rephrase_seed=args.policy_rephrase_seed,
        extra_args=tuple(extra),
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_sft(parse_args(argv))


if __name__ == "__main__":
    main()
