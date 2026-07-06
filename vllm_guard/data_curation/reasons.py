import argparse
import asyncio
import base64
import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from datasets import Dataset, load_dataset, load_from_disk
from openai import APIError, AsyncOpenAI, RateLimitError

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.data_curation.client import PROVIDER_CONFIGS, build_openai_compatible_client
from vllm_guard.data_curation.io import load_jsonl, load_mapping, remap_original_to_shuffled
from vllm_guard.training.formatting import (
    build_supervised_answer,
    randomize_policy_descriptions,
    select_policy_rephrased_description,
)


DEFAULT_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_RULES_PATH = str(Path(__file__).resolve().parents[2] / "data_curation" / "rules" / "basic_rules_v2.json")
DEFAULT_METADATA_DIR = str(Path(__file__).resolve().parents[2] / "data_curation" / "outputs" / "metadata")
DEFAULT_MAPPING_PATH = str(Path(__file__).resolve().parents[2] / "data_curation" / "outputs" / "v2.3" / "dataset_mapping.json")


@dataclass(frozen=True)
class ReasonGenerationConfig:
    data_path: str = str(CANONICAL_DATASET_DIR)
    output_path: Optional[str] = None
    rules_path: str = DEFAULT_RULES_PATH
    metadata_gpt: Optional[str] = None
    metadata_gemini: Optional[str] = None
    metadata_qwen: Optional[str] = None
    mapping_path: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-5.4-mini"
    provider: str = "openai"
    api_key: Optional[str] = None
    split: str = "sft"
    max_concurrent: int = 24
    batch_size: int = 200
    max_retries: int = 5
    retry_delay: float = 5.0
    reason_style: str = "legacy_short"
    policy_rephrase_path: Optional[str] = None
    policy_rephrase_seed: int = 42
    materialize_policy_rephrase: bool = False
    max_tokens: int = 96
    temperature: float = 0.3


def load_rules(rules_path: str) -> dict[int, dict[str, Any]]:
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    return {int(s["section_id"]): s for s in rules["sections"]}


def get_policy_logic(section_map: dict[int, Any], section_id: int, policy_name: str) -> Optional[str]:
    section = section_map.get(section_id)
    if not section:
        return None
    for pv in section.get("policy_variants", []):
        if pv["name"] == policy_name:
            return pv.get("logic")
    return None


def get_policy_description(section_map: dict[int, Any], section_id: int, policy_name: str) -> Optional[str]:
    section = section_map.get(section_id)
    if not section:
        return None
    for pv in section.get("policy_variants", []):
        if pv["name"] == policy_name:
            return pv.get("description")
    return None


def get_fields_info(section_map: dict[int, Any], section_id: int) -> list[dict[str, str]]:
    section = section_map.get(section_id)
    if not section:
        return []
    return [{"name": f["name"], "description": f["description"], "role": f.get("role", "trigger")} for f in section.get("fields", [])]


def choose_qwen_alignment(
    qwen_raw: dict[int, dict[str, Any]],
    gpt_data: dict[int, dict[str, Any]],
    gemini_data: dict[int, dict[str, Any]],
    mapping_path: str,
) -> dict[int, dict[str, Any]]:
    common_direct = len(set(qwen_raw) & set(gpt_data) & set(gemini_data))
    best_aligned = qwen_raw
    best_common = common_direct
    if Path(mapping_path).exists():
        mapping = load_mapping(mapping_path)
        qwen_remapped = remap_original_to_shuffled(qwen_raw, mapping)
        common_remapped = len(set(qwen_remapped) & set(gpt_data) & set(gemini_data))
        if common_remapped > best_common:
            best_aligned = qwen_remapped
            best_common = common_remapped
    return best_aligned


def normalize_field_value(val: str) -> str:
    val = str(val).lower().strip()
    if val in ("missing", "no", ""):
        return "no"
    return val


def vote_field_annotations(*records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotations = [rec.get("field_annotations", {}) for rec in records]
    merged = {}
    section_keys = set()
    for annot in annotations:
        section_keys |= set(annot.keys())
    for sec_key in section_keys:
        secs = [annot.get(sec_key, {}) for annot in annotations]
        section_id = None
        section_title = None
        for sec in secs:
            if sec:
                section_id = sec.get("section_id")
                section_title = sec.get("section_title")
                break
        if section_id is None:
            continue
        all_fields = set()
        for sec in secs:
            all_fields |= set(sec.get("fields", {}).keys())
        voted_fields = {}
        for fname in all_fields:
            vals = []
            for sec in secs:
                raw = sec.get("fields", {}).get(fname, {}).get("value", "missing")
                vals.append(normalize_field_value(raw))
            non_missing = [v for v in vals if v != "missing"]
            if non_missing:
                vals = non_missing
            counts = {}
            for val in vals:
                counts[val] = counts.get(val, 0) + 1
            top_val = max(counts.items(), key=lambda x: x[1])[0]
            voted_fields[fname] = {"value": top_val}
        merged[sec_key] = {
            "section_id": section_id,
            "section_title": section_title,
            "fields": voted_fields,
        }
    return merged


def build_voted_annotation_map(
    image_indices: list[int],
    metadata_gpt: str,
    metadata_gemini: str,
    metadata_qwen: Optional[str],
    mapping_path: str,
) -> dict[int, dict[str, dict[str, Any]]]:
    needed = {int(i) for i in image_indices}
    gpt_data = load_jsonl(metadata_gpt)
    gemini_data = load_jsonl(metadata_gemini)
    qwen_data_aligned = {}
    if metadata_qwen and Path(metadata_qwen).exists():
        qwen_raw = load_jsonl(metadata_qwen)
        qwen_data_aligned = choose_qwen_alignment(qwen_raw, gpt_data, gemini_data, mapping_path)

    voted_map = {}
    for idx in sorted(needed):
        if idx not in gpt_data or idx not in gemini_data:
            continue
        recs = [gpt_data[idx], gemini_data[idx]]
        if qwen_data_aligned and idx in qwen_data_aligned:
            recs.append(qwen_data_aligned[idx])
        voted_map[idx] = vote_field_annotations(*recs)
    return voted_map


def get_section_annotation(voted_annotations: Optional[dict[str, dict[str, Any]]], section_id: int) -> dict[str, Any]:
    if not voted_annotations:
        return {}
    for sec_data in voted_annotations.values():
        if int(sec_data.get("section_id", -1)) == int(section_id):
            return sec_data
    return {}


def describe_target_fields(section_map: dict[int, Any], section_id: int, voted_annotations: Optional[dict[str, dict[str, Any]]]) -> str:
    fields = get_fields_info(section_map, section_id)
    sec_data = get_section_annotation(voted_annotations, section_id)
    voted_fields = sec_data.get("fields", {}) if sec_data else {}
    lines = []
    for field in fields:
        value = voted_fields.get(field["name"], {}).get("value", "unknown")
        label_text = {"yes": "yes", "no": "no", "unknown": "unclear"}.get(str(value).lower(), str(value).lower())
        role_text = "Trigger field" if field["role"] == "trigger" else "Supporting field"
        lines.append(f"- {role_text}: {field['description']} | label: {label_text}")
    return "\n".join(lines) if lines else "- No field labels available"


def build_example_key(example: dict[str, Any], *, idx: int, split: str = "sft") -> str:
    return (
        f"{split}:{example.get('image_idx')}:{example.get('section_id')}:"
        f"{example.get('policy_name')}:{idx}"
    )


def get_effective_policy_description(
    example: dict[str, Any],
    section_map: dict[int, Any],
    *,
    idx: int,
    split: str = "sft",
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> str:
    section_id = int(example["section_id"])
    policy_name = example.get("policy_name", "")
    policy_description = (
        example.get("policy_description", "")
        or get_policy_description(section_map, section_id, policy_name)
        or ""
    )
    if not policy_rephrase_path:
        return policy_description
    return select_policy_rephrased_description(
        policy_description=policy_description,
        section_id=section_id,
        policy_name=str(policy_name),
        rephrase_path=policy_rephrase_path,
        seed=policy_rephrase_seed,
        example_key=build_example_key(example, idx=idx, split=split),
    )


def build_materialized_row(
    row: dict[str, Any],
    section_map: dict[int, Any],
    *,
    idx: int,
    split: str = "sft",
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> dict[str, Any]:
    payload = dict(row)
    if not policy_rephrase_path:
        return payload
    example_key = build_example_key(row, idx=idx, split=split)
    payload["original_question"] = row.get("question", "")
    payload["original_policy_description"] = row.get("policy_description", "")
    payload["question"] = randomize_policy_descriptions(
        row.get("question", ""),
        rephrase_path=policy_rephrase_path,
        seed=policy_rephrase_seed,
        example_key=example_key,
    )
    payload["policy_description"] = get_effective_policy_description(
        row,
        section_map,
        idx=idx,
        split=split,
        policy_rephrase_path=policy_rephrase_path,
        policy_rephrase_seed=policy_rephrase_seed,
    )
    payload["policy_rephrase_enabled"] = True
    payload["policy_rephrase_seed"] = policy_rephrase_seed
    return payload


def extract_image_bytes(item: dict[str, Any], max_size: int = 512, quality: int = 80) -> Optional[bytes]:
    img_data = item.get("image")
    if img_data is None:
        return None
    try:
        from PIL import Image

        if isinstance(img_data, dict) and "bytes" in img_data:
            img = Image.open(io.BytesIO(img_data["bytes"]))
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, Image.Image):
            img = img_data
        else:
            return None
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception:
        return None


def image_bytes_to_content(raw_bytes: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    header = raw_bytes[:4]
    if header[:3] == b"\xff\xd8\xff":
        mime = "jpeg"
    elif header[:4] == b"\x89PNG":
        mime = "png"
    elif header[:4] == b"GIF8":
        mime = "gif"
    elif header[:4] == b"RIFF":
        mime = "webp"
    else:
        mime = "jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}


def build_prompt(
    example: dict[str, Any],
    section_map: dict[int, Any],
    voted_annotations: Optional[dict[str, dict[str, Any]]] = None,
    *,
    idx: int = 0,
    split: str = "sft",
    reason_style: str = "legacy_short",
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> str:
    label = example["label"]
    section_id = int(example["section_id"])
    violated_cats = example.get("violated_categories", [])
    section_title = example.get("section_title", "Unknown")
    policy_name = example.get("policy_name", "Unknown")
    policy_description = get_effective_policy_description(
        example,
        section_map,
        idx=idx,
        split=split,
        policy_rephrase_path=policy_rephrase_path,
        policy_rephrase_seed=policy_rephrase_seed,
    )
    logic_str = get_policy_logic(section_map, section_id, policy_name) or "N/A"
    fields_block = describe_target_fields(section_map, section_id, voted_annotations)

    answer = "true" if label == "block" else "false"
    violated_str = ", ".join(f"{c:02d}" for c in violated_cats) if label == "block" and violated_cats else "none"

    if reason_style == "v29_policy_trace":
        final_answer = example.get("answer") or (f"true | {violated_str}" if answer == "true" else "false")
        return f"""You are generating the <think> reasoning text for one supervised vision-language safety example.

The dataset label is fixed. Do not change the answer. Your job is to write a faithful reasoning trace that matches the image, the target policy text, and the ground-truth answer.

Target Category: {section_title} (Category ID: {section_id:02d})
Target Policy: {policy_name}
Target Policy Text (use this exact policy wording; it may be a randomized rephrase):
{policy_description}

Blocking Logic Reference (for your understanding only; do not quote code-like logic):
{logic_str}

Observed Target-Category Cues (semantic labels from metadata; do not expose field names):
{fields_block}

Ground-Truth Final Answer: {final_answer}
Ground-Truth Label: {answer}
Accepted Violated Categories: {violated_str}

Write a concise reason as exactly three short English sentences:
1. Start with "Image:" and briefly name only the visual cue that matters.
2. Start with "Policy:" and briefly state the relevant policy rule.
3. Start with "Decision:" and briefly connect the cue to the rule; explicitly say "the answer is true" or "the answer is false" to match the ground truth.

Requirements:
- Output only the three-sentence reason text, with no markdown, no JSON, no bullet list, and no final answer line.
- Keep the whole reason compact, ideally under 45 words total.
- Do not mention schema keys, underscores, field names, or code-like expressions.
- Do not say the policy is randomized.
- Do not contradict the ground-truth final answer.
- Prefer clear, minimal wording over detailed explanation."""

    return f"""You are writing one short supervision reason for a Trust & Safety training example.

Look at the image and use ONLY the target policy information below.

Target Section: {section_title} (ID: {section_id:02d})
Target Policy: {policy_name}
Target Policy Description:
{policy_description}

Blocking Logic:
{logic_str}

Observed Field Labels for the target section:
{fields_block}

Ground-Truth Final Answer: {answer}
Violated Categories: {violated_str}

Write exactly one complete English sentence, about 15-20 words.

Requirements:
- Explain the answer as a short reasoning process:
  what the image shows (or does not show),
  what the policy / field logic implies,
  and therefore why the answer is true or false.
- Do not mention schema keys, JSON, underscores, or code-like names.
- Do not output fragments.
- Make it one complete sentence, not just a conclusion.

Good example for true:
"The image shows a real knife, and the policy blocks weapons, so the answer is true."

Good example for false:
"The image shows no weapon or violent act, and the policy only blocks those cues, so the answer is false."

Output only the sentence."""


def is_reason_usable(reason: str) -> bool:
    if not isinstance(reason, str):
        return False
    text = reason.strip()
    if not text or text.startswith("[ERROR"):
        return False
    lower = text.lower()
    if len(text.split()) < 8:
        return False
    if text[-1] not in ".!?":
        return False
    banned_exact = {
        "the image is",
        "the image is safe",
        "the image is unsafe",
        "this image is safe",
        "this image is unsafe",
        "the image violates the policy",
        "the image does not violate the policy",
        "safe",
        "unsafe",
    }
    if lower in banned_exact:
        return False
    if not (
        "violates the policy" in lower
        or "does not violate the policy" in lower
        or "answer is true" in lower
        or "answer is false" in lower
    ):
        return False
    if any(marker in lower for marker in ("has_", "is_", "_yes", "_no", "block if", "logic", "json")):
        return False
    if lower.endswith((" a.", " an.", " no.", " real.", " visible.", " clearly.", " fictional.", " sexually.", " exposed.")):
        return False
    return True


def _naturalize_field_description(desc: str) -> str:
    text = (desc or "").strip().rstrip(".")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    prefixes = (
        r"^Whether the image contains\s+",
        r"^Whether the image is\s+",
        r"^Whether the subject is\s+",
        r"^Whether any person in the image has\s+",
        r"^Presence of\s+",
        r"^Text contains\s+",
        r"^Images with\s+",
        r"^Image with\s+",
    )
    for pat in prefixes:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = text.strip(" ,;:")
    return text or "relevant risk cues"


def _compact_field_description(desc: str, max_words: int = 10) -> str:
    words = _naturalize_field_description(desc).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",")


def _join_phrases(phrases: list[str]) -> str:
    phrases = [p.strip() for p in phrases if p.strip()]
    if not phrases:
        return "the relevant cues"
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} or {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, or {phrases[-1]}"


def build_fallback_reason(
    example: dict[str, Any],
    section_map: dict[int, Any],
    voted_annotations: Optional[dict[str, dict[str, Any]]] = None,
    *,
    idx: int = 0,
    split: str = "sft",
    reason_style: str = "legacy_short",
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> str:
    section_id = int(example["section_id"])
    label = example["label"]
    fields = get_fields_info(section_map, section_id)
    sec_data = get_section_annotation(voted_annotations, section_id)
    voted_fields = sec_data.get("fields", {}) if sec_data else {}

    yes_fields = []
    no_trigger_fields = []
    for field in fields:
        value = str(voted_fields.get(field["name"], {}).get("value", "unknown")).lower()
        phrase = _compact_field_description(field["description"])
        if value == "yes":
            yes_fields.append(phrase)
        elif value == "no" and field.get("role") == "trigger":
            no_trigger_fields.append(phrase)

    policy_description = get_effective_policy_description(
        example,
        section_map,
        idx=idx,
        split=split,
        policy_rephrase_path=policy_rephrase_path,
        policy_rephrase_seed=policy_rephrase_seed,
    ) or "the target policy"
    if label == "block":
        trigger_text = _join_phrases(yes_fields[:2])
        if reason_style == "v29_policy_trace":
            return (
                f"Image: It shows {trigger_text}. "
                f"Policy: This policy disallows that content. "
                "Decision: The cue matches the policy, so the answer is true."
            )
        return f"The image shows {trigger_text}, and this conflicts with {policy_description.lower()}, so the answer is true."
    safe_text = _join_phrases(no_trigger_fields[:2])
    if reason_style == "v29_policy_trace":
        return (
            f"Image: It does not show {safe_text}. "
            "Policy: This policy blocks only matching disallowed content. "
            "Decision: No blocked cue is present, so the answer is false."
        )
    return f"The image does not show {safe_text}, and it stays within {policy_description.lower()}, so the answer is false."


async def generate_reason_single(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    example: dict[str, Any],
    idx: int,
    section_map: dict[int, Any],
    voted_annotations: Optional[dict[str, dict[str, Any]]],
    model: str,
    max_retries: int,
    retry_delay: float,
    reason_style: str,
    policy_rephrase_path: Optional[str],
    policy_rephrase_seed: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt = build_prompt(
        example,
        section_map,
        voted_annotations=voted_annotations,
        idx=idx,
        reason_style=reason_style,
        policy_rephrase_path=policy_rephrase_path,
        policy_rephrase_seed=policy_rephrase_seed,
    )
    img_bytes = extract_image_bytes(example)
    fallback_reason = build_fallback_reason(
        example,
        section_map,
        voted_annotations=voted_annotations,
        idx=idx,
        reason_style=reason_style,
        policy_rephrase_path=policy_rephrase_path,
        policy_rephrase_seed=policy_rephrase_seed,
    )
    if img_bytes is None:
        return {"idx": idx, "reason": fallback_reason, "reason_source": "fallback_template"}

    image_content = image_bytes_to_content(img_bytes)
    for attempt in range(max_retries):
        try:
            attempt_prompt = prompt
            if attempt > 0:
                if reason_style == "v29_policy_trace":
                    attempt_prompt += (
                        "\n\nIMPORTANT: Your previous answer was malformed. "
                        "Return exactly three concise sentences starting with Image:, Policy:, and Decision:. "
                        "Keep the whole reason under 45 words if possible. "
                        "The Decision sentence must explicitly say the answer is true or the answer is false."
                    )
                else:
                    attempt_prompt += (
                        "\n\nIMPORTANT: Your previous answer was too vague or malformed. "
                        "Write one complete English sentence that gives a concrete reason and a clear conclusion. "
                        "Do not answer with fragments like 'The image is safe' or 'The image violates the policy'."
                    )
            messages = [{"role": "user", "content": [{"type": "text", "text": attempt_prompt}, image_content]}]
            async with semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                    ),
                    timeout=90.0,
                )
            reason = response.choices[0].message.content.strip()
            if is_reason_usable(reason):
                return {"idx": idx, "reason": reason, "reason_source": "model"}
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
                continue
            return {"idx": idx, "reason": fallback_reason, "reason_source": "fallback_template"}
        except (RateLimitError, APIError, asyncio.TimeoutError, Exception):
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
                continue
            return {"idx": idx, "reason": fallback_reason, "reason_source": "fallback_template"}


def load_checkpoint(checkpoint_path: Path) -> dict[int, dict[str, str]]:
    done = {}
    if not checkpoint_path.exists():
        return done
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in rec and "reason" in rec and is_reason_usable(rec["reason"]):
                done[int(rec["idx"])] = {
                    "reason": rec["reason"],
                    "reason_source": rec.get("reason_source", "model"),
                }
    return done


async def process_batch(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    ds,
    pending_indices: list[int],
    checkpoint_file,
    section_map: dict[int, Any],
    voted_annotation_map: dict[int, dict[str, dict[str, Any]]],
    model: str,
    max_retries: int,
    retry_delay: float,
    reason_style: str,
    policy_rephrase_path: Optional[str],
    policy_rephrase_seed: int,
    max_tokens: int,
    temperature: float,
) -> dict[int, dict[str, str]]:
    tasks = [
        asyncio.ensure_future(
            generate_reason_single(
                client=client,
                semaphore=semaphore,
                example=ds[i],
                idx=i,
                section_map=section_map,
                voted_annotations=voted_annotation_map.get(int(ds[i]["image_idx"])),
                model=model,
                max_retries=max_retries,
                retry_delay=retry_delay,
                reason_style=reason_style,
                policy_rephrase_path=policy_rephrase_path,
                policy_rephrase_seed=policy_rephrase_seed,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        for i in pending_indices
    ]
    batch_reasons = {}
    for coro in asyncio.as_completed(tasks):
        res = await coro
        checkpoint_file.write(json.dumps(res, ensure_ascii=False) + "\n")
        checkpoint_file.flush()
        batch_reasons[int(res["idx"])] = {
            "reason": res["reason"],
            "reason_source": res.get("reason_source", "model"),
        }
    return batch_reasons


def load_split(dataset_path: Path, split: str):
    split_dir = dataset_path / split
    split_parquet = dataset_path / f"{split}.parquet"
    if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
        return load_from_disk(str(split_dir))
    if split_parquet.exists():
        return load_dataset("parquet", data_files=str(split_parquet), split="train")
    raise FileNotFoundError(f"Cannot find split {split} under {dataset_path}")


def build_materialized_dataset(
    ds,
    section_map: dict[int, Any],
    *,
    split: str = "sft",
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> Dataset:
    def iter_rows():
        for i, row in enumerate(ds):
            yield build_materialized_row(
                row,
                section_map,
                idx=i,
                split=split,
                policy_rephrase_path=policy_rephrase_path,
                policy_rephrase_seed=policy_rephrase_seed,
            )

    return Dataset.from_generator(iter_rows)


def build_sft_dataset(
    ds,
    section_map: dict[int, Any],
    *,
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> Dataset:
    return build_materialized_dataset(
        ds,
        section_map,
        split="sft",
        policy_rephrase_path=policy_rephrase_path,
        policy_rephrase_seed=policy_rephrase_seed,
    )


def build_sft_think_dataset(
    ds,
    done: dict[int, dict[str, str]],
    section_map: dict[int, Any],
    *,
    policy_rephrase_path: Optional[str] = None,
    policy_rephrase_seed: int = 42,
) -> Dataset:
    def iter_rows():
        for i, row in enumerate(ds):
            effective_row = build_materialized_row(
                row,
                section_map,
                idx=i,
                policy_rephrase_path=policy_rephrase_path,
                policy_rephrase_seed=policy_rephrase_seed,
            )
            reason = done.get(i, {}).get("reason", "")
            reason_source = done.get(i, {}).get("reason_source", "missing")
            target_text = build_supervised_answer(
                label=effective_row["label"],
                violated_categories=effective_row.get("violated_categories", []),
                section_id=int(effective_row["section_id"]),
                reason=reason,
                use_think_tags=True,
            )
            payload = {
                "question": effective_row["question"],
                "image": effective_row["image"],
                "target_text": target_text,
                "reason": reason,
                "answer": effective_row.get("answer", ""),
                "label": effective_row["label"],
                "section_id": effective_row["section_id"],
                "section_title": effective_row.get("section_title", ""),
                "policy_name": effective_row.get("policy_name", ""),
                "violated_categories": effective_row.get("violated_categories", []),
                "image_idx": effective_row.get("image_idx", -1),
                "reason_source": reason_source,
                "tier": effective_row.get("tier", ""),
                "policy_description": effective_row.get("policy_description", ""),
                "split_type": "sft_think",
            }
            for extra_key in (
                "discrimination_score",
                "original_question",
                "original_policy_description",
                "policy_rephrase_enabled",
                "policy_rephrase_seed",
            ):
                if extra_key in effective_row:
                    payload[extra_key] = effective_row[extra_key]
            yield payload
    return Dataset.from_generator(iter_rows)


def write_dataset_artifacts(dataset, output_dir: Path, split_name: str) -> None:
    split_dir = output_dir / split_name
    tmp_split_dir = output_dir / f"{split_name}_tmp"
    if tmp_split_dir.exists():
        shutil.rmtree(tmp_split_dir)
    dataset.save_to_disk(str(tmp_split_dir))
    if split_dir.exists():
        shutil.rmtree(split_dir)
    tmp_split_dir.rename(split_dir)
    dataset.to_parquet(str(output_dir / f"{split_name}.parquet"))
    with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
        for row in dataset:
            dump = {k: v for k, v in row.items() if k != "image"}
            f.write(json.dumps(dump, ensure_ascii=False) + "\n")


def mirror_split_if_needed(src_root: Path, dst_root: Path, split_name: str) -> None:
    if src_root.resolve() == dst_root.resolve():
        return
    src_dir = src_root / split_name
    dst_dir = dst_root / split_name
    if src_dir.exists() and not dst_dir.exists():
        shutil.copytree(src_dir, dst_dir)
    src_parquet = src_root / f"{split_name}.parquet"
    dst_parquet = dst_root / f"{split_name}.parquet"
    if src_parquet.exists() and not dst_parquet.exists():
        shutil.copy2(src_parquet, dst_parquet)
    src_jsonl = src_root / f"{split_name}.jsonl"
    dst_jsonl = dst_root / f"{split_name}.jsonl"
    if src_jsonl.exists() and not dst_jsonl.exists():
        shutil.copy2(src_jsonl, dst_jsonl)


def mirror_static_root_artifacts(src_root: Path, dst_root: Path) -> None:
    if src_root.resolve() == dst_root.resolve():
        return
    split_or_generated_names = {
        "id_test",
        "ood_test",
        "sft",
        "sft_think",
        "rl",
        "eval_results",
        "validation",
        "visualizations",
        "inspection",
        "backups",
    }
    skip_prefixes = (
        "sft_checkpoint",
        "sft_think",
        "sft.",
    )
    for item in src_root.iterdir():
        if item.name in split_or_generated_names:
            continue
        if any(item.name.startswith(prefix) for prefix in skip_prefixes):
            continue
        dst = dst_root / item.name
        if dst.exists():
            continue
        if item.is_file():
            shutil.copy2(item, dst)


async def run_reason_generation_async(config: ReasonGenerationConfig) -> dict[str, Any]:
    if config.provider == "gemini-native":
        raise ValueError("gemini-native is not supported for reason generation; use an OpenAI-compatible route")

    api_key = config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Use --api-key or export OPENAI_API_KEY / GEMINI_API_KEY")

    data_path = Path(config.data_path)
    output_path = Path(config.output_path) if config.output_path else data_path
    output_path.mkdir(parents=True, exist_ok=True)

    section_map = load_rules(config.rules_path)
    metadata_gpt = config.metadata_gpt or str(Path(DEFAULT_METADATA_DIR) / "metadata_gpt51.jsonl")
    metadata_gemini = config.metadata_gemini or str(Path(DEFAULT_METADATA_DIR) / "metadata_gemini_flash.jsonl")
    metadata_qwen = config.metadata_qwen or str(Path(DEFAULT_METADATA_DIR) / "metadata_qwen25vl72b.jsonl")
    mapping_path = config.mapping_path or DEFAULT_MAPPING_PATH

    ds = load_split(data_path, config.split)
    all_image_indices = [int(ds[i]["image_idx"]) for i in range(len(ds))]
    voted_annotation_map = build_voted_annotation_map(
        all_image_indices,
        metadata_gpt=metadata_gpt,
        metadata_gemini=metadata_gemini,
        metadata_qwen=metadata_qwen,
        mapping_path=mapping_path,
    )

    client = build_openai_compatible_client(
        provider=config.provider,
        api_key=api_key,
        base_url=config.base_url or DEFAULT_BASE_URL,
    )
    semaphore = asyncio.Semaphore(config.max_concurrent)

    ckpt_path = output_path / f"{config.split}_checkpoint.jsonl"
    done = load_checkpoint(ckpt_path)
    pending = [i for i in range(len(ds)) if i not in done]
    if pending:
        with ckpt_path.open("a", encoding="utf-8") as fout:
            for batch_start in range(0, len(pending), config.batch_size):
                batch_indices = pending[batch_start:batch_start + config.batch_size]
                batch_reasons = await process_batch(
                    client=client,
                    semaphore=semaphore,
                    ds=ds,
                    pending_indices=batch_indices,
                    checkpoint_file=fout,
                    section_map=section_map,
                    voted_annotation_map=voted_annotation_map,
                    model=config.model,
                    max_retries=config.max_retries,
                    retry_delay=config.retry_delay,
                    reason_style=config.reason_style,
                    policy_rephrase_path=config.policy_rephrase_path,
                    policy_rephrase_seed=config.policy_rephrase_seed,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                done.update(batch_reasons)

    mirror_static_root_artifacts(data_path, output_path)

    for split_name in ("id_test", "ood_test", "rl", "eval_results"):
        if split_name == "rl" and config.materialize_policy_rephrase:
            continue
        if split_name == "eval_results":
            src = data_path / split_name
            dst = output_path / split_name
            if src.exists() and not dst.exists() and src.is_dir() and data_path.resolve() != output_path.resolve():
                shutil.copytree(src, dst)
            continue
        mirror_split_if_needed(data_path, output_path, split_name)

    if config.materialize_policy_rephrase:
        sft_ds = build_sft_dataset(
            ds,
            section_map,
            policy_rephrase_path=config.policy_rephrase_path,
            policy_rephrase_seed=config.policy_rephrase_seed,
        )
        write_dataset_artifacts(sft_ds, output_path, "sft")
        try:
            rl_ds = load_split(data_path, "rl")
        except FileNotFoundError:
            rl_ds = None
        if rl_ds is not None:
            randomized_rl_ds = build_materialized_dataset(
                rl_ds,
                section_map,
                split="rl",
                policy_rephrase_path=config.policy_rephrase_path,
                policy_rephrase_seed=config.policy_rephrase_seed,
            )
            write_dataset_artifacts(randomized_rl_ds, output_path, "rl")
    else:
        mirror_split_if_needed(data_path, output_path, "sft")

    think_ds = build_sft_think_dataset(
        ds,
        done,
        section_map,
        policy_rephrase_path=config.policy_rephrase_path if config.materialize_policy_rephrase else None,
        policy_rephrase_seed=config.policy_rephrase_seed,
    )
    write_dataset_artifacts(think_ds, output_path, "sft_think")

    summary = {
        "data_path": str(data_path),
        "output_path": str(output_path),
        "model": config.model,
        "provider": config.provider,
        "split": config.split,
        "rows": len(ds),
        "reason_style": config.reason_style,
        "policy_rephrase_path": config.policy_rephrase_path,
        "policy_rephrase_seed": config.policy_rephrase_seed,
        "materialize_policy_rephrase": config.materialize_policy_rephrase,
        "reason_source_counts": {},
    }
    for info in done.values():
        source = info.get("reason_source", "unknown")
        summary["reason_source_counts"][source] = summary["reason_source_counts"].get(source, 0) + 1
    return summary


def run_reason_generation(config: ReasonGenerationConfig) -> dict[str, Any]:
    return asyncio.run(run_reason_generation_async(config))


def parse_args(argv: Sequence[str] | None = None) -> ReasonGenerationConfig:
    p = argparse.ArgumentParser(description="Canonical adaptive-policy reason generation entrypoint")
    p.add_argument("--data-path", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--output-path", default=None)
    p.add_argument("--rules", dest="rules_path", default=DEFAULT_RULES_PATH)
    p.add_argument("--metadata-gpt", default=None)
    p.add_argument("--metadata-gemini", default=None)
    p.add_argument("--metadata-qwen", default=None)
    p.add_argument("--mapping", dest="mapping_path", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--provider", default="openai", choices=sorted(PROVIDER_CONFIGS))
    p.add_argument("--api-key", default=None)
    p.add_argument("--split", default="sft", choices=["sft"])
    p.add_argument("--max-concurrent", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-delay", type=float, default=5.0)
    p.add_argument("--reason-style", default="legacy_short", choices=["legacy_short", "v29_policy_trace"])
    p.add_argument("--policy-rephrase-path", default=None)
    p.add_argument("--policy-rephrase-seed", type=int, default=42)
    p.add_argument("--materialize-policy-rephrase", action="store_true")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args(argv)
    return ReasonGenerationConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_reason_generation(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
