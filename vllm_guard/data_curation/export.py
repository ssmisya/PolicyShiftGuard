from collections import defaultdict


def build_prompt_for_instance(policy_combination, section_map):
    policy_lines = []
    for sid in sorted(section_map.keys()):
        section = section_map[sid]
        policy_name = policy_combination.get(sid)
        if not policy_name:
            continue
        policy_desc = next((pv["description"] for pv in section.get("policy_variants", []) if pv["name"] == policy_name), None)
        if not policy_desc:
            continue
        policy_lines.append(
            f"[Category ID]: {sid:02d}    [Category Name]: {section['title']}\n"
            f"-- Policy: {policy_name}\n{policy_desc}"
        )
    all_policies_text = "\n" + ("-" * 80) + "\n".join(["\n" + p + "\n" + ("-" * 80) for p in policy_lines])
    return (
        "You are an image safety moderator. "
        "Analyze the provided image according to the following policies.\n\n"
        "**POLICIES:**"
        "{all_policies}"
    ).format(all_policies=all_policies_text)


def format_answer_v26(label, violated_categories):
    if label == "block" and violated_categories:
        return f"true | {','.join(f'{c:02d}' for c in violated_categories)}"
    return "false"


def build_policy_catalog(section_map):
    return {
        str(sid): {
            "title": sec["title"],
            "section_description": sec.get("section_description", ""),
            "fields": [{"name": f["name"], "description": f["description"], "role": f.get("role", "trigger")} for f in sec.get("fields", [])],
            "policy_variants": [{"name": pv["name"], "description": pv["description"]} for pv in sec.get("policy_variants", [])],
        }
        for sid, sec in section_map.items()
    }


def compute_stats(instances):
    stats = {
        "total": len(instances),
        "by_tier": defaultdict(int),
        "by_section": defaultdict(int),
        "by_label": defaultdict(int),
        "by_section_policy": defaultdict(int),
        "violated_categories_distribution": defaultdict(int),
        "num_violated_categories": defaultdict(int),
    }
    for inst in instances:
        stats["by_tier"][inst["tier"]] += 1
        stats["by_section"][inst["section_id"]] += 1
        stats["by_label"][inst["label"]] += 1
        stats["by_section_policy"][f"{inst['section_id']}_{inst['policy_name']}"] += 1
        violated = inst.get("violated_categories", [])
        stats["num_violated_categories"][str(len(violated))] += 1
        for cat_id in violated:
            stats["violated_categories_distribution"][str(cat_id)] += 1
    return {k: dict(v) if isinstance(v, defaultdict) else v for k, v in stats.items()}
