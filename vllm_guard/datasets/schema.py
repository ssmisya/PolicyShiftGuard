from dataclasses import dataclass
from typing import Iterable, Sequence


BASE_INSTANCE_FIELDS: tuple[str, ...] = (
    "question",
    "image",
    "answer",
    "image_idx",
    "section_id",
    "section_title",
    "policy_name",
    "tier",
    "discrimination_score",
    "policy_description",
    "label",
    "split_type",
    "violated_categories",
)

REASON_FIELDS: tuple[str, ...] = ("reason", "reason_source")
THINK_FIELDS: tuple[str, ...] = REASON_FIELDS + ("target_text",)


@dataclass(frozen=True)
class DatasetSplitContract:
    name: str
    required_fields: tuple[str, ...]
    description: str


def has_required_fields(columns: Sequence[str], required_fields: Iterable[str]) -> bool:
    available = set(columns)
    return all(field in available for field in required_fields)

