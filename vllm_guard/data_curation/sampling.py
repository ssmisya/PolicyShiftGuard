import random
from collections import Counter, defaultdict

from vllm_guard.data_curation.export import build_prompt_for_instance
from vllm_guard.data_curation.policy import (
    compute_violated_categories,
    discrimination_score,
    is_policy_discriminative,
)


ID_POLICIES = {
    1: ["Policy A (General Safe / Social Media Standard)", "Policy B (Strict Puritan / Family Safe)", "Policy E (Looser Regulations / Creative Freedom)", "Policy D (E-commerce & Advertising / Lingerie Mode)"],
    2: ["Policy A (Real-world Safety / Social Media Standard)", "Policy B (Zero Tolerance / School & Kids)", "Policy E (Maximum Freedom / Legal Compliance)"],
    3: ["Policy A (Global Family Friendly / Mainstream Social)", "Policy D (Strict Wellness / Minor Protection)"],
    4: ["Policy B (Commercial Clean / Stock Photo Mode)"],
    5: ["Policy A (Western Standard / Liberal)", "Policy E (Global Universal Safety / Maximum Neutrality)"],
    6: ["Policy A (Social Media Sharing)", "Policy B (Street View / Anonymity First)"],
    7: ["Policy A (Hate Speech Filter / Basic Safety)", "Policy C (Neutral Environment / Non-Political)"],
}

OOD_POLICIES = {
    1: ["Policy C (Medical & Educational / Scientific Reference)"],
    2: ["Policy C (Journalism & Archive / Anti-Entertainment)", "Policy D (Gaming & Creative Platform / Fiction Only)"],
    3: ["Policy B (Regional Permissive - e.g., Canada/Thailand/California)", "Policy C (Retail & Pharmacy Marketplace)"],
    4: ["Policy A (Creative Assistant / Parody Friendly)", "Policy C (Brand Protection / Official Manufacturing)"],
    5: ["Policy B (Halal/Kosher Compliance)", "Policy C (Japanese Professional / Traditional)", "Policy D (India Friendly / Hindu Context)"],
    6: ["Policy C (Secure Data Entry / OCR)"],
    7: ["Policy B (Anti-Spam & Commercial Cleanliness)"],
}


def find_valid_policy_combination(target_section_id, target_policy_name, all_policy_labels, section_map, target_final_label, policy_set, rng):
    target_section_labels = all_policy_labels.get(target_section_id, {})
    target_label = target_section_labels.get(target_policy_name, "pass")
    combo = {target_section_id: target_policy_name}
    if target_final_label == "block":
        if target_label == "block":
            for sid in section_map.keys():
                if sid == target_section_id:
                    continue
                section_labels = all_policy_labels.get(sid, {})
                available = policy_set.get(sid, [])
                if not available:
                    continue
                if section_labels:
                    pass_policies = [p for p in available if p in section_labels and section_labels[p] == "pass"]
                    combo[sid] = rng.choice(pass_policies) if pass_policies else next((p for p in available if p in section_labels), available[0])
                else:
                    combo[sid] = available[0]
            return combo
        blocking_sections = []
        for sid, section_labels in all_policy_labels.items():
            if sid == target_section_id:
                continue
            available = policy_set.get(sid, [])
            blocking = [p for p in available if p in section_labels and section_labels[p] == "block"]
            if blocking:
                blocking_sections.append((sid, blocking))
        if not blocking_sections:
            return None
        chosen_sid, blocking = rng.choice(blocking_sections)
        combo[chosen_sid] = rng.choice(blocking)
        for sid in section_map.keys():
            if sid in combo:
                continue
            section_labels = all_policy_labels.get(sid, {})
            available = policy_set.get(sid, [])
            if not available:
                continue
            if section_labels:
                pass_policies = [p for p in available if p in section_labels and section_labels[p] == "pass"]
                combo[sid] = rng.choice(pass_policies) if pass_policies else next((p for p in available if p in section_labels), available[0])
            else:
                combo[sid] = available[0]
        return combo
    for sid in section_map.keys():
        if sid == target_section_id:
            if target_label != "pass":
                return None
            continue
        section_labels = all_policy_labels.get(sid, {})
        available = policy_set.get(sid, [])
        if not available:
            continue
        if not section_labels:
            combo[sid] = available[0]
            continue
        passing = [p for p in available if p in section_labels and section_labels[p] == "pass"]
        if not passing:
            return None
        combo[sid] = rng.choice(passing)
    return combo


def find_ood_policy_combination(target_section_id, target_ood_policy_name, all_policy_labels, section_map, desired_final_label, rng):
    target_section_labels = all_policy_labels.get(target_section_id, {})
    target_label = target_section_labels.get(target_ood_policy_name, "pass")
    if desired_final_label == "block" and target_label != "block":
        return None
    if desired_final_label == "pass" and target_label != "pass":
        return None
    combo = {target_section_id: target_ood_policy_name}
    for sid in sorted(section_map.keys()):
        if sid == target_section_id:
            continue
        section_labels = all_policy_labels.get(sid, {})
        all_available = list(set(ID_POLICIES.get(sid, []) + OOD_POLICIES.get(sid, [])))
        if not all_available:
            continue
        if not section_labels:
            combo[sid] = rng.choice(all_available)
            continue
        if desired_final_label == "block":
            pass_policies = [p for p in all_available if p in section_labels and section_labels[p] == "pass"]
            combo[sid] = rng.choice(pass_policies) if pass_policies else rng.choice([p for p in all_available if p in section_labels] or all_available)
        else:
            passing = [p for p in all_available if p in section_labels and section_labels[p] == "pass"]
            if not passing:
                return None
            combo[sid] = rng.choice(passing)
    return combo


def generate_discriminative_instances(image_indices, record_map, section_map, policy_set, split_name, seed=42):
    rng = random.Random(seed + hash(split_name))
    instances, seen = [], set()
    for idx in image_indices:
        rec = record_map[idx]
        all_policy_labels = rec.get("all_policy_labels", {})
        for sid, available_policies in policy_set.items():
            section_labels = all_policy_labels.get(sid, {})
            section = section_map.get(sid)
            if not section_labels or not section:
                continue
            for policy_name in available_policies:
                if policy_name not in section_labels:
                    continue
                dedup = (idx, sid, policy_name)
                if dedup in seen:
                    continue
                seen.add(dedup)
                target_label = section_labels[policy_name]
                valid_combo = find_valid_policy_combination(sid, policy_name, all_policy_labels, section_map, target_label, policy_set, rng)
                if not valid_combo:
                    continue
                policy_desc = next((pv["description"] for pv in section.get("policy_variants", []) if pv["name"] == policy_name), None)
                if not policy_desc:
                    continue
                inst = {
                    "image_idx": idx,
                    "split": split_name,
                    "tier": "policy_discriminative",
                    "section_id": sid,
                    "section_title": section["title"],
                    "policy_name": policy_name,
                    "policy_description": policy_desc,
                    "label": target_label,
                    "discrimination_score": round(discrimination_score(section_labels), 4),
                    "policy_combination": valid_combo,
                    "violated_categories": compute_violated_categories(valid_combo, all_policy_labels),
                }
                inst["question"] = build_prompt_for_instance(valid_combo, section_map)
                instances.append(inst)
    rng.shuffle(instances)
    return instances


def generate_ood_instances(image_indices, record_map, section_map, seed=42):
    rng = random.Random(seed + hash("ood_v26"))
    instances, seen = [], set()
    for idx in image_indices:
        rec = record_map[idx]
        all_policy_labels = rec.get("all_policy_labels", {})
        for sid, ood_policies in OOD_POLICIES.items():
            section_labels = all_policy_labels.get(sid, {})
            section = section_map.get(sid)
            if not section_labels or not section:
                continue
            for policy_name in ood_policies:
                if policy_name not in section_labels:
                    continue
                dedup = (idx, sid, policy_name)
                if dedup in seen:
                    continue
                seen.add(dedup)
                valid_combo = find_ood_policy_combination(sid, policy_name, all_policy_labels, section_map, section_labels[policy_name], rng)
                if not valid_combo:
                    continue
                violated_categories = compute_violated_categories(valid_combo, all_policy_labels)
                policy_desc = next((pv["description"] for pv in section.get("policy_variants", []) if pv["name"] == policy_name), None)
                if not policy_desc:
                    continue
                inst = {
                    "image_idx": idx,
                    "split": "ood_test",
                    "tier": "policy_discriminative",
                    "section_id": sid,
                    "section_title": section["title"],
                    "policy_name": policy_name,
                    "policy_description": policy_desc,
                    "label": "block" if violated_categories else "pass",
                    "discrimination_score": round(discrimination_score(section_labels), 4),
                    "policy_combination": valid_combo,
                    "violated_categories": violated_categories,
                }
                inst["question"] = build_prompt_for_instance(valid_combo, section_map)
                instances.append(inst)
    rng.shuffle(instances)
    return instances


def generate_rl_instances(all_indices, record_map, section_map, exclude_image_ids, seed=42):
    rng = random.Random(seed + hash("rl_broad"))
    instances, seen = [], set()
    for idx in all_indices:
        if idx in exclude_image_ids:
            continue
        rec = record_map[idx]
        all_policy_labels = rec.get("all_policy_labels", {})
        if not all_policy_labels:
            continue
        is_truly_safe = rec.get("safety") == "safe"
        for sid, available_policies in ID_POLICIES.items():
            section_labels = all_policy_labels.get(sid, {})
            section = section_map.get(sid)
            if not section_labels or not section:
                continue
            for policy_name in available_policies:
                if policy_name not in section_labels:
                    continue
                dedup = (idx, sid, policy_name)
                if dedup in seen:
                    continue
                seen.add(dedup)
                valid_combo = find_valid_policy_combination(sid, policy_name, all_policy_labels, section_map, section_labels[policy_name], ID_POLICIES, rng)
                if not valid_combo:
                    continue
                policy_desc = next((pv["description"] for pv in section.get("policy_variants", []) if pv["name"] == policy_name), None)
                if not policy_desc:
                    continue
                inst = {
                    "image_idx": idx,
                    "split": "rl",
                    "tier": "rl_truly_safe" if is_truly_safe else "rl_broad",
                    "section_id": sid,
                    "section_title": section["title"],
                    "policy_name": policy_name,
                    "policy_description": policy_desc,
                    "label": section_labels[policy_name],
                    "discrimination_score": round(discrimination_score(section_labels), 4),
                    "policy_combination": valid_combo,
                    "violated_categories": compute_violated_categories(valid_combo, all_policy_labels),
                }
                inst["question"] = build_prompt_for_instance(valid_combo, section_map)
                instances.append(inst)
    rng.shuffle(instances)
    return instances


def group_instances_by_image(instances):
    grouped = defaultdict(list)
    for inst in instances:
        grouped[inst["image_idx"]].append(inst)
    return grouped


def balance_instances(instances, target_size=None, seed=42):
    block_inst = [inst for inst in instances if inst["label"] == "block"]
    pass_inst = [inst for inst in instances if inst["label"] == "pass"]
    rng = random.Random(seed)
    rng.shuffle(block_inst)
    rng.shuffle(pass_inst)
    if target_size is None:
        n = min(len(block_inst), len(pass_inst))
        balanced = block_inst[:n] + pass_inst[:n]
        rng.shuffle(balanced)
        return balanced, n, n
    half = target_size // 2
    n_block = min(half, len(block_inst))
    n_pass = min(half, len(pass_inst))
    if n_block < half:
        n_pass = min(target_size - n_block, len(pass_inst))
    if n_pass < half:
        n_block = min(target_size - n_pass, len(block_inst))
    balanced = block_inst[:n_block] + pass_inst[:n_pass]
    rng.shuffle(balanced)
    return balanced, n_block, n_pass


def select_image_subset_for_balanced_target(grouped_instances, target_size, seed=42):
    target_half = target_size // 2
    rng = random.Random(seed)
    candidates = []
    for image_idx, items in grouped_instances.items():
        counts = Counter(inst["label"] for inst in items)
        candidates.append({"image_idx": image_idx, "block": counts.get("block", 0), "pass": counts.get("pass", 0), "total": len(items)})
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: (min(x["block"], x["pass"]), x["total"], max(x["block"], x["pass"])), reverse=True)
    selected_images, total_block, total_pass = [], 0, 0
    for cand in candidates:
        if total_block >= target_half and total_pass >= target_half:
            break
        if cand["block"] == 0 and total_block >= target_half:
            continue
        if cand["pass"] == 0 and total_pass >= target_half:
            continue
        selected_images.append(cand["image_idx"])
        total_block += cand["block"]
        total_pass += cand["pass"]
    if total_block < target_half or total_pass < target_half:
        raise ValueError(f"cannot find enough image-disjoint capacity for target_size={target_size}")
    return set(selected_images), total_block, total_pass

