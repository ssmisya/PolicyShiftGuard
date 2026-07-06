from dataclasses import dataclass
from pathlib import Path

from vllm_guard.common.constants import CANONICAL_DATASET_DIR, CANONICAL_HF_DATASET_REPO
from vllm_guard.datasets.schema import BASE_INSTANCE_FIELDS, DatasetSplitContract, THINK_FIELDS


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_dir: Path
    hf_repo: str
    description: str
    no_think_split: DatasetSplitContract
    think_split: DatasetSplitContract
    eval_splits: tuple[str, ...]
    rl_split: str


ADAPTIVE_POLICY_V27_WITHREASON = DatasetSpec(
    name="adaptive_policy_v2.7_withreason",
    root_dir=CANONICAL_DATASET_DIR,
    hf_repo=CANONICAL_HF_DATASET_REPO,
    description=(
        "Canonical adaptive-policy benchmark and training dataset. "
        "`sft` is no-think/no-reason, `sft_think` carries reason supervision."
    ),
    no_think_split=DatasetSplitContract(
        name="sft",
        required_fields=BASE_INSTANCE_FIELDS,
        description="No-think SFT split. Final answer only, no reason columns.",
    ),
    think_split=DatasetSplitContract(
        name="sft_think",
        required_fields=BASE_INSTANCE_FIELDS + THINK_FIELDS,
        description="Think SFT split. Includes reason and target_text.",
    ),
    eval_splits=("id_test", "ood_test"),
    rl_split="rl",
)

DATASET_REGISTRY = {
    ADAPTIVE_POLICY_V27_WITHREASON.name: ADAPTIVE_POLICY_V27_WITHREASON,
}


def get_dataset_spec(name: str = ADAPTIVE_POLICY_V27_WITHREASON.name) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset spec: {name}") from exc

