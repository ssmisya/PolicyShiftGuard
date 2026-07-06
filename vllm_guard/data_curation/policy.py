import math
import re
from collections import Counter


def build_section_map(rules):
    return {s["section_id"]: s for s in rules["sections"]}


def normalize_field_value(val: str) -> str:
    val = val.lower().strip()
    if val in ("missing", "no", ""):
        return "no"
    return val


def vote_field_annotations(*records):
    annotations = [rec.get("field_annotations", {}) for rec in records]
    all_section_keys = set().union(*(set(a.keys()) for a in annotations))
    vote_stats = {"total": 0, "unanimous": 0, "majority": 0, "unresolved": 0}
    merged = {}
    for sec_key in all_section_keys:
        secs = [annot.get(sec_key, {}) for annot in annotations]
        section_id = next((s.get("section_id") for s in secs if s.get("section_id") is not None), None)
        section_title = next((s.get("section_title") for s in secs if s.get("section_title")), None)
        if section_id is None:
            continue
        all_fields = set().union(*(set(s.get("fields", {}).keys()) for s in secs))
        voted_fields = {}
        for fname in all_fields:
            vals = [normalize_field_value(s.get("fields", {}).get(fname, {}).get("value", "missing")) for s in secs]
            vote_stats["total"] += 1
            counts = Counter(vals)
            top_val, top_count = counts.most_common(1)[0]
            if top_count == len(vals):
                voted_fields[fname] = {"value": top_val, "confidence": "vote_unanimous"}
                vote_stats["unanimous"] += 1
            elif top_count > len(vals) / 2:
                voted_fields[fname] = {"value": top_val, "confidence": "vote_majority"}
                vote_stats["majority"] += 1
            else:
                voted_fields[fname] = {"value": "unknown", "confidence": "vote_unresolved"}
                vote_stats["unresolved"] += 1
        merged[sec_key] = {"section_id": section_id, "section_title": section_title, "fields": voted_fields}
    return merged, vote_stats


def evaluate_policy_logic(logic_str: str, field_values: dict[str, bool]) -> bool:
    text = re.sub(r"BLOCK\s+IF\s*:", "", logic_str)
    text = " ".join(re.sub(r"#.*", "", line).strip() for line in text.split("\n") if re.sub(r"#.*", "", line).strip())
    text = re.sub(r"\s+(AND|OR)\s*$", "", text.strip())
    text = re.sub(r"\bAND\b", "and", text)
    text = re.sub(r"\bOR\b", "or", text)
    text = re.sub(r"\bNOT\b", "not", text)
    text = re.sub(r"\b[A-Z][A-Za-z0-9_]*\b", lambda m: str(field_values.get(m.group(0), False)), text)
    try:
        return bool(eval(text))  # noqa: S307
    except Exception:
        return False


def compute_all_policy_labels(voted_annotations, section_map):
    all_labels = {}
    for _, section_data in voted_annotations.items():
        sid = section_data["section_id"]
        section_def = section_map.get(sid)
        if not section_def or "policy_variants" not in section_def:
            continue
        field_bools = {fname: fdata["value"] == "yes" for fname, fdata in section_data["fields"].items()}
        all_labels[sid] = {
            pv["name"]: ("block" if evaluate_policy_logic(pv["logic"], field_bools) else "pass")
            for pv in section_def["policy_variants"]
        }
    return all_labels


def discrimination_score(policies):
    decisions = list(policies.values())
    n = len(decisions)
    if n == 0:
        return 0.0
    n_block = sum(1 for d in decisions if d == "block")
    if n_block == 0 or n_block == n:
        return 0.0
    p = n_block / n
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def is_policy_discriminative(policies):
    unique_labels = set(policies.values()) if policies else set()
    return len(unique_labels) > 1 and "block" in unique_labels and "pass" in unique_labels


def compute_final_label(policy_combination, all_policy_labels):
    for sid, policy_name in policy_combination.items():
        if all_policy_labels.get(sid, {}).get(policy_name, "pass") == "block":
            return "block"
    return "pass"


def compute_violated_categories(policy_combination, all_policy_labels):
    return sorted([
        sid for sid, policy_name in policy_combination.items()
        if all_policy_labels.get(sid, {}).get(policy_name, "pass") == "block"
    ])

