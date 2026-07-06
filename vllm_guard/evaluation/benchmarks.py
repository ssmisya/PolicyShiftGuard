from dataclasses import dataclass
from pathlib import Path

from vllm_guard.common.constants import ADAPTIVE_POLICY_EVAL_SPLITS
from vllm_guard.datasets.registry import ADAPTIVE_POLICY_V27_WITHREASON, DatasetSpec
from vllm_guard.training.formatting import build_output_instructions


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset: DatasetSpec
    default_splits: tuple[str, ...]
    supported_splits: tuple[str, ...]
    output_instructions: str
    description: str

    @property
    def default_dataset_path(self) -> Path:
        return self.dataset.root_dir


ADAPTIVE_POLICY_V27_WITHREASON_BENCHMARK = BenchmarkSpec(
    name="adaptive_policy_v2.7_withreason",
    dataset=ADAPTIVE_POLICY_V27_WITHREASON,
    default_splits=("id_test", "ood_test"),
    supported_splits=ADAPTIVE_POLICY_EVAL_SPLITS,
    output_instructions=build_output_instructions(no_reason=False, use_think_tags=False),
    description=(
        "Canonical adaptive-policy multimodal guardrail benchmark. "
        "Evaluation uses the same question/image instance format across models."
    ),
)

BENCHMARK_REGISTRY = {
    ADAPTIVE_POLICY_V27_WITHREASON_BENCHMARK.name: ADAPTIVE_POLICY_V27_WITHREASON_BENCHMARK,
}


def get_benchmark_spec(name: str = ADAPTIVE_POLICY_V27_WITHREASON_BENCHMARK.name) -> BenchmarkSpec:
    try:
        return BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark spec: {name}") from exc

