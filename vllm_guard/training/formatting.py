import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence


def build_output_instructions(*, no_reason: bool = False, use_think_tags: bool = False) -> str:
    if use_think_tags:
        return (
            "\n\n"
            "Based on these policies, determine if the image violates any policy.\n"
            "\n"
            "Answer in this exact format (no extra text):\n"
            "First write a brief internal reason inside <think> and </think>.\n"
            "Then append the final decision.\n"
            "If content violates a policy: <think>...</think> true | <two-digit Risk Category ID> "
            "(e.g. \"01\", \"09\")\n"
            "If content is safe: <think>...</think> false"
        )
    if no_reason:
        return (
            "\n\n"
            "Based on these policies, determine if the image violates any policy.\n"
            "\n"
            "Answer in this exact format (no extra text):\n"
            "If content violates a policy: true | <two-digit Risk Category ID> (e.g. \"01\", \"09\")\n"
            "If content is safe: false"
        )
    return (
        "\n\n"
        "Based on these policies, determine if the image violates any policy.\n"
        "\n"
        "Answer in this exact format (no extra text):\n"
        "If content violates a policy: true | <two-digit Risk Category ID> (e.g. \"01\", \"09\") "
        "| a short reason (why the content violates the policy)\n"
        "If content is safe: false | a short reason (why the content is safe)"
    )


@lru_cache(maxsize=4)
def _load_policy_rephrase_entries(rephrase_path: str) -> tuple[dict, ...]:
    path = Path(rephrase_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy rephrase file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for section in payload.get("sections", []):
        section_id = section.get("section_id")
        for item in section.get("policy_rephrases", []):
            original = str(item.get("original_description", "")).strip()
            rephrases = [str(x).strip() for x in item.get("rephrases", []) if str(x).strip()]
            if original and rephrases:
                entries.append(
                    {
                        "section_id": section_id,
                        "policy_name": item.get("policy_name", ""),
                        "original_description": original,
                        "rephrases": tuple(rephrases),
                    }
                )
    return tuple(entries)


def _stable_rephrase_index(*, seed: int, example_key: str, section_id: object, policy_name: str, n: int) -> int:
    key = f"{seed}|{example_key}|{section_id}|{policy_name}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n


def randomize_policy_descriptions(
    question: str,
    *,
    rephrase_path: Optional[str] = None,
    seed: int = 0,
    example_key: str = "",
) -> str:
    """Replace policy descriptions in a dataset question with deterministic rephrases.

    The output instructions and target format are not changed. Only exact policy
    description blocks from the rules file are replaced, so unmatched prompts are
    left untouched.
    """

    if not rephrase_path:
        return question
    randomized = question
    for entry in _load_policy_rephrase_entries(str(rephrase_path)):
        original = entry["original_description"]
        if original not in randomized:
            continue
        rephrases = entry["rephrases"]
        choice = rephrases[
            _stable_rephrase_index(
                seed=seed,
                example_key=example_key,
                section_id=entry["section_id"],
                policy_name=str(entry["policy_name"]),
                n=len(rephrases),
            )
        ]
        randomized = randomized.replace(original, choice)
    return randomized


def select_policy_rephrased_description(
    *,
    policy_description: str,
    section_id: object,
    policy_name: str,
    rephrase_path: Optional[str] = None,
    seed: int = 0,
    example_key: str = "",
) -> str:
    """Return the deterministic rephrase used for one target policy description.

    This mirrors ``randomize_policy_descriptions`` so materialized datasets can
    store the same target policy text that appears inside the randomized prompt.
    """

    if not rephrase_path:
        return policy_description
    target_section = str(section_id)
    target_policy = str(policy_name)
    for entry in _load_policy_rephrase_entries(str(rephrase_path)):
        if str(entry["section_id"]) != target_section:
            continue
        if str(entry["policy_name"]) != target_policy:
            continue
        rephrases = entry["rephrases"]
        return rephrases[
            _stable_rephrase_index(
                seed=seed,
                example_key=example_key,
                section_id=entry["section_id"],
                policy_name=str(entry["policy_name"]),
                n=len(rephrases),
            )
        ]
    return policy_description


def build_supervised_answer(
    *,
    label: str,
    violated_categories: Sequence[int],
    section_id: int,
    reason: Optional[str] = None,
    target_text: Optional[str] = None,
    no_reason: bool = False,
    use_think_tags: bool = False,
) -> str:
    if target_text and str(target_text).strip():
        return target_text

    if label == "block":
        category_id = f"{(violated_categories[0] if violated_categories else section_id):02d}"
        final_reason = reason or "Content violates the policy."
        if use_think_tags:
            return f"<think>{final_reason}</think> true | {category_id}"
        if no_reason:
            return f"true | {category_id}"
        return f"true | {category_id} | {final_reason}"

    final_reason = reason or "Content is safe under the policy."
    if use_think_tags:
        return f"<think>{final_reason}</think> false"
    if no_reason:
        return "false"
    return f"false | {final_reason}"
