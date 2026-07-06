import argparse
from dataclasses import dataclass
from typing import Sequence

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.training.cls.pipeline import main as pipeline_main


@dataclass(frozen=True)
class CLSTrainConfig:
    model_name_or_path: str
    output_dir: str
    dataset_path: str = str(CANONICAL_DATASET_DIR)
    train_split: str = "sft"
    extra_args: tuple[str, ...] = ()

    def to_cli_args(self) -> list[str]:
        args = [
            "--model_name_or_path", self.model_name_or_path,
            "--dataset_path", self.dataset_path,
            "--output_dir", self.output_dir,
            "--train_split", self.train_split,
        ]
        args.extend(self.extra_args)
        return args


def run_cls(config: CLSTrainConfig) -> None:
    pipeline_main(config.to_cli_args())


def parse_args(argv: Sequence[str] | None = None) -> CLSTrainConfig:
    p = argparse.ArgumentParser(description="Canonical CLS training entrypoint")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_path", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--train_split", default="sft")
    args, extra = p.parse_known_args(argv)
    return CLSTrainConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        train_split=args.train_split,
        extra_args=tuple(extra),
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_cls(parse_args(argv))


if __name__ == "__main__":
    main()
