import argparse
import asyncio
import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from openai import APIError, AsyncOpenAI, RateLimitError

from vllm_guard.data_curation.client import PROVIDER_CONFIGS, build_openai_compatible_client, resolve_api_key


DEFAULT_RULES_PATH = str(Path(__file__).resolve().parents[2] / "data_curation" / "rules" / "basic_rules_v2.json")

SYSTEM_PROMPT = (
    "You are a professional image content safety analyst. "
    "You follow instructions precisely and always respond in the exact JSON format requested. "
    "Do not include any explanation, markdown fencing, or extra text outside the JSON object."
)

SECTION_EVAL_TEMPLATE = """Analyze the provided image for the following risk category.

## Category: {section_title}

{section_description}

## Fields to Evaluate

For each field below, determine whether it applies to this image based ONLY on what you can see. Answer "yes" or "no" with a confidence level.

{fields_block}

## Output Format (strict JSON, no extra text)

{{
{output_example}
}}"""


@dataclass(frozen=True)
class MetadataCurationConfig:
    input_path: str
    output_path: str
    rules_path: str = DEFAULT_RULES_PATH
    model: str = "gpt-4.1-mini"
    provider: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    concurrency: int = 16
    split: str = "train"
    start_idx: int = 0
    end_idx: Optional[int] = None
    batch_size: int = 100
    verbose: bool = False


def load_rules(rules_path: str) -> dict[str, Any]:
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_section_map(rules: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {s["section_id"]: s for s in rules["sections"]}


def load_hf_dataset(input_path: str, split: str = "train"):
    from datasets import load_dataset, load_from_disk

    p = Path(input_path)
    if p.is_file() and p.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(p), split="train")
    if p.is_dir() and (p / "dataset_info.json").exists():
        return load_from_disk(str(p))
    if p.is_dir():
        return load_dataset(str(p), split=split)
    return load_dataset(str(p), split=split)


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


def build_image_content(source) -> dict[str, Any]:
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": source}}
        with open(source, "rb") as f:
            raw = f.read()
        return image_bytes_to_content(raw)
    if isinstance(source, bytes):
        return image_bytes_to_content(source)
    if isinstance(source, dict) and "bytes" in source:
        return image_bytes_to_content(source["bytes"])
    try:
        from PIL import Image

        if isinstance(source, Image.Image):
            buf = io.BytesIO()
            fmt = source.format or "JPEG"
            source.save(buf, format=fmt)
            return image_bytes_to_content(buf.getvalue())
    except Exception:
        pass
    raise ValueError(f"Unsupported image source type: {type(source)}")


def build_section_prompt(section: dict[str, Any]) -> str:
    fields_lines = []
    output_lines = []
    for i, field in enumerate(section["fields"]):
        fields_lines.append(
            f'- **{field["name"]}**: {field["description"]}\n'
            f'  Question: {field["question"]}'
        )
        comma = "," if i < len(section["fields"]) - 1 else ""
        output_lines.append(
            f'  "{field["name"]}": {{"value": "yes" or "no", "confidence": "high" or "medium" or "low"}}{comma}'
        )
    return SECTION_EVAL_TEMPLATE.format(
        section_title=section["title"],
        section_description=section.get("section_description", ""),
        fields_block="\n".join(fields_lines),
        output_example="\n".join(output_lines),
    )


def _openai_messages_to_gemini(messages: list[dict[str, Any]]):
    from google.genai import types

    system_text = None
    contents = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_text = msg["content"]
            continue
        gemini_role = "user" if role == "user" else "model"
        parts = []
        content = msg["content"]
        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for block in content:
                if block["type"] == "text":
                    parts.append(types.Part.from_text(text=block["text"]))
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        header, b64data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        raw_bytes = base64.b64decode(b64data)
                        parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime_type))
        contents.append(types.Content(role=gemini_role, parts=parts))
    return system_text, contents


async def _call_gemini_native(client, model: str, semaphore: asyncio.Semaphore, messages: list[dict[str, Any]], temperature: float):
    from google.genai import types

    system_text, contents = _openai_messages_to_gemini(messages)
    config = types.GenerateContentConfig(temperature=temperature)
    if system_text:
        config.system_instruction = system_text
    async with semaphore:
        response = await client.aio.models.generate_content(model=model, contents=contents, config=config)
    return response.text.strip()


async def call_model(
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    provider: str,
) -> str:
    if provider == "gemini-native":
        return await _call_gemini_native(client, model, semaphore, messages, temperature)
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    return resp.choices[0].message.content.strip()


def _clean_json_response(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


async def evaluate_section_fields(
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    image_content: dict[str, Any],
    section: dict[str, Any],
    provider: str,
) -> dict[str, dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": build_section_prompt(section)}, image_content]},
    ]
    raw = await call_model(
        client=client,
        model=model,
        semaphore=semaphore,
        messages=messages,
        provider=provider,
    )
    try:
        data = json.loads(_clean_json_response(raw))
    except json.JSONDecodeError:
        data = {}

    results = {}
    for field in section["fields"]:
        fname = field["name"]
        if fname in data and isinstance(data[fname], dict):
            value = str(data[fname].get("value", "no")).strip().lower()
            confidence = str(data[fname].get("confidence", "medium")).strip().lower()
        else:
            value = "no"
            confidence = "low"
        if value not in ("yes", "no"):
            value = "no"
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        results[fname] = {"value": value, "confidence": confidence}
    return results


def evaluate_policy_logic(logic_str: str, field_values: dict[str, bool]) -> bool:
    text = re.sub(r"BLOCK\s+IF\s*:", "", logic_str)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"#.*", "", line).strip()
        if line:
            lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+(AND|OR)\s*$", "", text.strip())
    text = re.sub(r"\bAND\b", "and", text)
    text = re.sub(r"\bOR\b", "or", text)
    text = re.sub(r"\bNOT\b", "not", text)

    def _replace_field(match):
        return str(field_values.get(match.group(0), False))

    text = re.sub(r"\b[A-Z][A-Za-z0-9_]*\b", _replace_field, text)
    try:
        return bool(eval(text))
    except Exception:
        return False


def compute_policy_labels(fields_by_section: dict[str, dict[str, Any]], section_map: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    policy_labels = {}
    for key, section_data in fields_by_section.items():
        sid = section_data["section_id"]
        section_def = section_map.get(sid)
        if not section_def or "policy_variants" not in section_def:
            continue
        field_bools = {fname: fdata["value"] == "yes" for fname, fdata in section_data["fields"].items()}
        policies = {}
        for pv in section_def["policy_variants"]:
            block = evaluate_policy_logic(pv["logic"], field_bools)
            policies[pv["name"]] = "block" if block else "pass"
        policy_labels[key] = {
            "section_id": sid,
            "section_title": section_data["section_title"],
            "policies": policies,
        }
    return policy_labels


async def classify_single_item(
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    item_idx: int,
    section_map: dict[int, dict[str, Any]],
    provider: str,
) -> dict[str, Any]:
    image_content = build_image_content(item["image"])
    sids = sorted(section_map.keys())
    section_tasks = [
        evaluate_section_fields(
            client=client,
            model=model,
            semaphore=semaphore,
            image_content=image_content,
            section=section_map[sid],
            provider=provider,
        )
        for sid in sids
    ]
    gathered = await asyncio.gather(*section_tasks)
    section_results = {sid: result for sid, result in zip(sids, gathered)}

    risk_sections = []
    for sid, fields in section_results.items():
        section = section_map[sid]
        trigger_names = {f["name"] for f in section["fields"] if f.get("role") == "trigger"}
        has_trigger = any(fields.get(fname, {}).get("value") == "yes" for fname in trigger_names)
        if has_trigger:
            risk_sections.append(sid)

    fields_by_section = {}
    for sid in sids:
        section = section_map[sid]
        key = f"{sid}_{section['title']}"
        fields_by_section[key] = {
            "section_id": sid,
            "section_title": section["title"],
            "fields": section_results[sid],
        }

    policy_labels = compute_policy_labels(fields_by_section, section_map)
    safety = "unsafe" if risk_sections else "safe"
    return {
        "idx": item_idx,
        "safety": safety,
        "risk_categories": [{"section_id": sid, "section_title": section_map[sid]["title"]} for sid in sorted(risk_sections)],
        "field_annotations": fields_by_section,
        "policy_labels": policy_labels,
    }


async def classify_single_item_with_retry(
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    item_idx: int,
    section_map: dict[int, dict[str, Any]],
    provider: str,
    max_retries: int = 5,
    retry_delay: float = 5.0,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    if verbose:
        print(f"    [idx={item_idx}] Starting classification...")
    for attempt in range(max_retries):
        try:
            result = await classify_single_item(
                client=client,
                model=model,
                semaphore=semaphore,
                item=item,
                item_idx=item_idx,
                section_map=section_map,
                provider=provider,
            )
            if verbose:
                print(f"    [idx={item_idx}] ✓ Success (attempt {attempt + 1}) - safety: {result.get('safety', 'unknown')}")
            return result
        except RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return None
        except APIError:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return None
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return None


async def classify_and_write_single_item(
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    item_idx: int,
    section_map: dict[int, dict[str, Any]],
    output_file,
    provider: str,
    verbose: bool = False,
) -> Optional[dict[str, Any]]:
    result = await classify_single_item_with_retry(
        client=client,
        model=model,
        semaphore=semaphore,
        item=item,
        item_idx=item_idx,
        section_map=section_map,
        provider=provider,
        verbose=verbose,
    )
    if result is None:
        return None
    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    output_file.flush()
    return result


async def classify_batch(
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    items: list[dict[str, Any]],
    indices: list[int],
    section_map: dict[int, dict[str, Any]],
    output_file,
    provider: str,
    verbose: bool = False,
) -> list[Optional[dict[str, Any]]]:
    tasks = [
        classify_and_write_single_item(
            client=client,
            model=model,
            semaphore=semaphore,
            item=item,
            item_idx=idx,
            section_map=section_map,
            output_file=output_file,
            provider=provider,
            verbose=verbose,
        )
        for item, idx in zip(items, indices)
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)


def load_completed_indices(output_path: Path) -> set[int]:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in rec:
                done.add(rec["idx"])
    return done


def build_client(config: MetadataCurationConfig):
    if config.provider == "gemini-native":
        from google import genai

        api_key = resolve_api_key(config.api_key, "GEMINI_API_KEY")
        return genai.Client(api_key=api_key)
    return build_openai_compatible_client(
        provider=config.provider,
        api_key=config.api_key,
        base_url=config.base_url,
    )


async def run_metadata_curation_async(config: MetadataCurationConfig) -> dict[str, Any]:
    rules = load_rules(config.rules_path)
    section_map = build_section_map(rules)
    dataset = load_hf_dataset(config.input_path, split=config.split)
    client = build_client(config)
    semaphore = asyncio.Semaphore(config.concurrency)

    total = len(dataset)
    start = config.start_idx
    end = min(config.end_idx, total) if config.end_idx is not None else total
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_indices = load_completed_indices(output_path)

    safe_count = 0
    unsafe_count = 0
    skipped_count = 0

    with output_path.open("a", encoding="utf-8") as fout:
        for batch_start in range(start, end, config.batch_size):
            batch_end = min(batch_start + config.batch_size, end)
            pending = [i for i in range(batch_start, batch_end) if i not in done_indices]
            if not pending:
                continue
            batch_items = [dataset[i] for i in pending]
            results = await classify_batch(
                client=client,
                model=config.model,
                semaphore=semaphore,
                items=batch_items,
                indices=pending,
                section_map=section_map,
                output_file=fout,
                provider=config.provider,
                verbose=config.verbose,
            )
            for res in results:
                if res is None:
                    skipped_count += 1
                elif res.get("safety") == "safe":
                    safe_count += 1
                else:
                    unsafe_count += 1

    return {
        "input_path": config.input_path,
        "output_path": str(output_path),
        "model": config.model,
        "provider": config.provider,
        "processed_range": [start, end],
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "skipped_count": skipped_count,
        "resumed_count": len(done_indices),
        "sections": len(rules["sections"]),
    }


def run_metadata_curation(config: MetadataCurationConfig) -> dict[str, Any]:
    return asyncio.run(run_metadata_curation_async(config))


def parse_args(argv: Sequence[str] | None = None) -> MetadataCurationConfig:
    p = argparse.ArgumentParser(description="Canonical metadata curation entrypoint")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rules", default=DEFAULT_RULES_PATH)
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--provider", default="openai", choices=[*sorted(PROVIDER_CONFIGS), "gemini-native"])
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--split", default="train")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    return MetadataCurationConfig(
        input_path=args.input,
        output_path=args.output,
        rules_path=args.rules,
        model=args.model,
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        split=args.split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_metadata_curation(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
