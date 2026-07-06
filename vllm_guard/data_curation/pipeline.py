import argparse
from dataclasses import dataclass
from typing import Sequence

from vllm_guard.common.constants import CANONICAL_DATASET_DIR, REPO_ROOT
from vllm_guard.data_curation.builder import DatasetBuildRuntimeConfig, build_dataset


@dataclass(frozen=True)
class BuildDatasetConfig:
    metadata_gpt: str
    metadata_gemini: str
    metadata_qwen: str
    mapping: str
    rules: str
    image_source: str
    image_source_type: str = "hf_dataset"
    output_dir: str = str(CANONICAL_DATASET_DIR)
    id_test_size: int = 1000
    ood_test_size: int = 1000
    sft_size: int = 2000
    seed: int = 42

def run_build_dataset(config: BuildDatasetConfig):
    runtime = DatasetBuildRuntimeConfig(**config.__dict__)
    return build_dataset(runtime)


def parse_args(argv: Sequence[str] | None = None) -> BuildDatasetConfig:
    p = argparse.ArgumentParser(description="Canonical adaptive-policy dataset build entrypoint")
    p.add_argument("--metadata-gpt", required=True)
    p.add_argument("--metadata-gemini", required=True)
    p.add_argument("--metadata-qwen", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--image-source", required=True)
    p.add_argument("--image-source-type", default="hf_dataset", choices=["hf_dataset", "directory"])
    p.add_argument("--output-dir", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--id-test-size", type=int, default=1000)
    p.add_argument("--ood-test-size", type=int, default=1000)
    p.add_argument("--sft-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    return BuildDatasetConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> None:
    run_build_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
