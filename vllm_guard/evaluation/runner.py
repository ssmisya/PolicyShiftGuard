import argparse
from dataclasses import dataclass
from typing import Sequence

from vllm_guard.evaluation.benchmarks import get_benchmark_spec
from vllm_guard.evaluation.pipeline import run_and_save


@dataclass
class EvalRunConfig:
    model_name: str
    output_dir: str
    split: str = "id_test"
    benchmark: str = "adaptive_policy_v2.7_withreason"
    model_type: str | None = None
    model_path: str | None = None
    tokenizer_path: str | None = None
    dataset_parquet: str | None = None
    batch_size: int = 1
    max_tokens: int = 64
    temperature: float = 0.0
    enable_thinking: bool = False
    response_format: str = "auto"
    extra_args: tuple[str, ...] = ()
    dataset_dir: str | None = None
    dataset_repo: str | None = None
    image_source: str | None = None
    image_source_type: str = "hf_dataset"
    resume: bool = False
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 4096
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    seed: int = 42


def run_eval(config: EvalRunConfig):
    bench = get_benchmark_spec(config.benchmark)
    if not config.dataset_dir and not config.dataset_repo and not config.dataset_parquet:
        config.dataset_parquet = str(bench.default_dataset_path)
    if config.response_format == "auto":
        config.response_format = "think" if config.enable_thinking else "reasoned"
    return run_and_save(config)


def parse_args(argv: Sequence[str] | None = None) -> EvalRunConfig:
    p = argparse.ArgumentParser(description="Canonical evaluation entrypoint")
    p.add_argument("--model-name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="id_test")
    p.add_argument("--benchmark", default="adaptive_policy_v2.7_withreason")
    p.add_argument("--model-type", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--dataset-parquet", default=None)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--dataset-repo", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true", default=False)
    p.add_argument("--response-format", choices=["auto", "reasoned", "nothink", "think"], default="auto")
    p.add_argument("--image-source", default=None)
    p.add_argument("--image-source-type", default="hf_dataset")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--vllm-tensor-parallel", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--vllm-max-model-len", type=int, default=4096)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    args, extra = p.parse_known_args(argv)
    return EvalRunConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        split=args.split,
        benchmark=args.benchmark,
        model_type=args.model_type,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_parquet=args.dataset_parquet,
        dataset_dir=args.dataset_dir,
        dataset_repo=args.dataset_repo,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        response_format=args.response_format,
        image_source=args.image_source,
        image_source_type=args.image_source_type,
        resume=args.resume,
        vllm_tensor_parallel=args.vllm_tensor_parallel,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        extra_args=tuple(extra),
    )


def main(argv: Sequence[str] | None = None) -> None:
    run_eval(parse_args(argv))


if __name__ == "__main__":
    main()
