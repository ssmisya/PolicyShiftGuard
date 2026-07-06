import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from vllm_guard.data_curation.export import build_policy_catalog, build_prompt_for_instance, compute_stats, format_answer_v26
from vllm_guard.data_curation.io import ImageLoader, load_jsonl, load_mapping, load_rules, remap_original_to_shuffled
from vllm_guard.data_curation.policy import build_section_map, compute_all_policy_labels, is_policy_discriminative, vote_field_annotations
from vllm_guard.data_curation.sampling import (
    ID_POLICIES,
    OOD_POLICIES,
    balance_instances,
    find_valid_policy_combination,
    generate_discriminative_instances,
    generate_ood_instances,
    generate_rl_instances,
    group_instances_by_image,
    select_image_subset_for_balanced_target,
)


@dataclass
class DatasetBuildRuntimeConfig:
    metadata_gpt: str
    metadata_gemini: str
    metadata_qwen: str
    mapping: str
    rules: str
    image_source: str
    image_source_type: str
    output_dir: str
    id_test_size: int
    ood_test_size: int
    sft_size: int
    seed: int


@dataclass
class AugmentSFTDatasetRuntimeConfig:
    base_dataset_dir: str
    metadata_gpt: str
    metadata_gemini: str
    metadata_qwen: str
    rules: str
    image_source: str
    image_source_type: str
    output_dir: str
    add_truly_safe: int
    add_truly_unsafe: int
    seed: int


def generate_hf_dataset(instances, image_loader, split_name):
    from datasets import Dataset, Image as HFImage

    def _row_generator():
        for inst in instances:
            pil_img = image_loader.get_pil_image(inst["image_idx"])
            violated = inst.get("violated_categories", [])
            yield {
                "question": inst["question"],
                "image": pil_img,
                "answer": format_answer_v26(inst["label"], violated),
                "image_idx": inst["image_idx"],
                "section_id": inst["section_id"],
                "section_title": inst["section_title"],
                "policy_name": inst["policy_name"],
                "tier": inst["tier"],
                "discrimination_score": inst.get("discrimination_score", 0.0),
                "policy_description": inst.get("policy_description", ""),
                "label": inst["label"],
                "split_type": inst.get("split_type", ""),
                "violated_categories": violated,
            }

    dataset = Dataset.from_generator(_row_generator)
    return dataset.cast_column("image", HFImage())


def _policy_labels_by_section_id(record):
    out = {}
    for sec_data in record.get("policy_labels", {}).values():
        sid = sec_data.get("section_id")
        if sid is None:
            continue
        out[int(sid)] = dict(sec_data.get("policies", {}))
    return out


def _has_any_trigger(voted_annotations, section_map):
    for sec_data in voted_annotations.values():
        section_def = section_map.get(sec_data["section_id"], {})
        for field in section_def.get("fields", []):
            if field.get("role") != "trigger":
                continue
            if sec_data["fields"].get(field["name"], {}).get("value", "no") == "yes":
                return True
    return False


def _build_record_map(indices, gpt_data, gemini_data, qwen_data, section_map):
    record_map = {}
    for idx in indices:
        voted_annotations, _ = vote_field_annotations(gpt_data[idx], gemini_data[idx], qwen_data[idx])
        record_map[idx] = {
            "idx": idx,
            "safety": "unsafe" if _has_any_trigger(voted_annotations, section_map) else "safe",
            "field_annotations": voted_annotations,
            "all_policy_labels": compute_all_policy_labels(voted_annotations, section_map),
        }
    return record_map


def _build_truly_sft_candidates(indices, gpt_data, gemini_data, qwen_data, section_map, seed):
    rng = random.Random(seed)
    record_map = _build_record_map(indices, gpt_data, gemini_data, qwen_data, section_map)
    safe_candidates, unsafe_candidates = [], []

    for idx in indices:
        records = [gpt_data[idx], gemini_data[idx], qwen_data[idx]]
        safety_votes = [rec.get("safety") for rec in records]
        strict_safe = all(v == "safe" for v in safety_votes)
        strict_unsafe = all(v == "unsafe" for v in safety_votes)
        if not strict_safe and not strict_unsafe:
            continue

        policy_maps = [_policy_labels_by_section_id(rec) for rec in records]
        merged_rec = record_map[idx]
        all_policy_labels = merged_rec["all_policy_labels"]

        for sid, available_policies in ID_POLICIES.items():
            section = section_map.get(sid)
            section_labels = all_policy_labels.get(sid, {})
            if not section or not section_labels:
                continue

            for policy_name in available_policies:
                votes = []
                for policy_map in policy_maps:
                    per_section = policy_map.get(sid, {})
                    if policy_name not in per_section:
                        votes = []
                        break
                    votes.append(per_section[policy_name])
                if not votes:
                    continue

                target_label = None
                tier = None
                if strict_safe and all(v == "pass" for v in votes):
                    target_label = "pass"
                    tier = "sft_truly_safe"
                elif strict_unsafe and all(v == "block" for v in votes):
                    target_label = "block"
                    tier = "sft_truly_unsafe"
                if target_label is None:
                    continue

                valid_combo = find_valid_policy_combination(
                    sid,
                    policy_name,
                    all_policy_labels,
                    section_map,
                    target_label,
                    ID_POLICIES,
                    rng,
                )
                if not valid_combo:
                    continue

                policy_desc = next(
                    (pv["description"] for pv in section.get("policy_variants", []) if pv["name"] == policy_name),
                    None,
                )
                if not policy_desc:
                    continue

                inst = {
                    "image_idx": idx,
                    "split": "sft",
                    "tier": tier,
                    "section_id": sid,
                    "section_title": section["title"],
                    "policy_name": policy_name,
                    "policy_description": policy_desc,
                    "label": target_label,
                    "discrimination_score": 0.0,
                    "policy_combination": valid_combo,
                    "violated_categories": [
                        cat_sid
                        for cat_sid, chosen_policy in valid_combo.items()
                        if all_policy_labels.get(cat_sid, {}).get(chosen_policy, "pass") == "block"
                    ],
                    "question": build_prompt_for_instance(valid_combo, section_map),
                    "split_type": "sft",
                }
                if target_label == "pass":
                    safe_candidates.append(inst)
                else:
                    unsafe_candidates.append(inst)

    return record_map, safe_candidates, unsafe_candidates


def _select_diverse_one_per_image(candidates, target_size, seed):
    grouped = defaultdict(list)
    for inst in candidates:
        grouped[inst["image_idx"]].append(inst)
    if len(grouped) < target_size:
        raise ValueError(
            f"Not enough distinct images for target_size={target_size}; only {len(grouped)} image groups available."
        )

    rng = random.Random(seed)
    image_ids = list(grouped.keys())
    rng.shuffle(image_ids)
    section_counts = Counter()
    selected = []
    for image_idx in image_ids:
        options = list(grouped[image_idx])
        rng.shuffle(options)
        options.sort(key=lambda inst: (section_counts[inst["section_id"]], inst["section_id"], inst["policy_name"]))
        chosen = options[0]
        selected.append(chosen)
        section_counts[chosen["section_id"]] += 1
        if len(selected) == target_size:
            break
    if len(selected) != target_size:
        raise ValueError(f"Could only select {len(selected)} candidates for target_size={target_size}")
    rng.shuffle(selected)
    return selected


def build_augmented_sft_dataset(config: AugmentSFTDatasetRuntimeConfig):
    base_dataset_dir = Path(config.base_dataset_dir)
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Refusing to write to non-directory output path: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output_dir: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    gpt_data = load_jsonl(config.metadata_gpt)
    gemini_data = load_jsonl(config.metadata_gemini)
    qwen_data = load_jsonl(config.metadata_qwen)
    rules = load_rules(config.rules)
    section_map = build_section_map(rules)

    common_indices = sorted(set(gpt_data) & set(gemini_data) & set(qwen_data))
    used_images = set()
    for split_name in ("sft", "sft_think", "id_test", "ood_test", "rl"):
        split_path = base_dataset_dir / f"{split_name}.parquet"
        import pandas as pd  # local import to keep base import surface small

        df = pd.read_parquet(split_path, columns=["image_idx"])
        used_images.update(int(x) for x in df["image_idx"].tolist())
    candidate_indices = [idx for idx in common_indices if idx not in used_images]

    record_map, safe_candidates, unsafe_candidates = _build_truly_sft_candidates(
        candidate_indices,
        gpt_data,
        gemini_data,
        qwen_data,
        section_map,
        seed=config.seed,
    )
    selected_safe = _select_diverse_one_per_image(safe_candidates, config.add_truly_safe, seed=config.seed + 11)
    selected_unsafe = _select_diverse_one_per_image(unsafe_candidates, config.add_truly_unsafe, seed=config.seed + 23)

    with (base_dataset_dir / "sft.jsonl").open("r", encoding="utf-8") as f:
        base_sft_instances = [json.loads(line) for line in f if line.strip()]
    augmented_sft_instances = base_sft_instances + selected_safe + selected_unsafe

    for item in base_dataset_dir.iterdir():
        if item.name in {
            "sft",
            "sft_think",
            "sft.parquet",
            "sft.jsonl",
            "sft_think.parquet",
            "sft_think.jsonl",
            "sft_checkpoint.jsonl",
        }:
            continue
        dst = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    image_loader = ImageLoader(config.image_source, config.image_source_type)
    sft_dataset = generate_hf_dataset(augmented_sft_instances, image_loader, "sft")
    split_dir = output_dir / "sft"
    parquet_path = output_dir / "sft.parquet"
    jsonl_path = output_dir / "sft.jsonl"
    sft_dataset.save_to_disk(str(split_dir))
    sft_dataset.to_parquet(str(parquet_path))
    with jsonl_path.open("w", encoding="utf-8") as f:
        for inst in augmented_sft_instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    base_stats_path = base_dataset_dir / "stats.json"
    if base_stats_path.exists():
        with base_stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {}
    stats["sft"] = compute_stats(augmented_sft_instances)
    stats["sft_append"] = {
        "added_truly_safe": len(selected_safe),
        "added_truly_unsafe": len(selected_unsafe),
        "selected_safe_images": len({inst["image_idx"] for inst in selected_safe}),
        "selected_unsafe_images": len({inst["image_idx"] for inst in selected_unsafe}),
        "candidate_common_unused_images": len(candidate_indices),
        "candidate_safe_cases": len(safe_candidates),
        "candidate_unsafe_cases": len(unsafe_candidates),
        "selected_safe_by_section": dict(Counter(inst["section_id"] for inst in selected_safe)),
        "selected_unsafe_by_section": dict(Counter(inst["section_id"] for inst in selected_unsafe)),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return {
        "base_sft_rows": len(base_sft_instances),
        "augmented_sft_rows": len(augmented_sft_instances),
        "added_truly_safe": len(selected_safe),
        "added_truly_unsafe": len(selected_unsafe),
        "selected_safe_by_section": dict(Counter(inst["section_id"] for inst in selected_safe)),
        "selected_unsafe_by_section": dict(Counter(inst["section_id"] for inst in selected_unsafe)),
        "candidate_common_unused_images": len(candidate_indices),
        "candidate_safe_cases": len(safe_candidates),
        "candidate_unsafe_cases": len(unsafe_candidates),
        "record_map_size": len(record_map),
    }


def build_dataset(config: DatasetBuildRuntimeConfig):
    random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt_data = load_jsonl(config.metadata_gpt)
    gemini_data = load_jsonl(config.metadata_gemini)
    qwen_data_raw = load_jsonl(config.metadata_qwen)
    mapping = load_mapping(config.mapping)
    qwen_data_remapped = remap_original_to_shuffled(qwen_data_raw, mapping)
    direct_common = len(set(gpt_data) & set(gemini_data) & set(qwen_data_raw))
    remapped_common = len(set(gpt_data) & set(gemini_data) & set(qwen_data_remapped))
    qwen_data_aligned = qwen_data_raw if direct_common >= remapped_common else qwen_data_remapped

    rules = load_rules(config.rules)
    section_map = build_section_map(rules)
    common_indices = sorted(set(gpt_data) & set(gemini_data) & set(qwen_data_aligned))
    record_map = {}
    id_discriminative_pool, ood_discriminative_pool, safe_indices, all_unsafe_indices = [], [], [], []
    global_vote_stats = {"total": 0, "unanimous": 0, "majority": 0, "unresolved": 0}

    for idx in common_indices:
        voted_annotations, vs = vote_field_annotations(gpt_data[idx], gemini_data[idx], qwen_data_aligned[idx])
        for k in global_vote_stats:
            global_vote_stats[k] += vs[k]
        all_policy_labels = compute_all_policy_labels(voted_annotations, section_map)
        has_any_trigger = any(
            sec_data["fields"].get(f["name"], {}).get("value", "no") == "yes"
            for sec_data in voted_annotations.values()
            for f in section_map.get(sec_data["section_id"], {}).get("fields", [])
            if f.get("role") == "trigger"
        )
        merged_rec = {"idx": idx, "safety": "unsafe" if has_any_trigger else "safe", "field_annotations": voted_annotations, "all_policy_labels": all_policy_labels}
        record_map[idx] = merged_rec
        if not has_any_trigger:
            safe_indices.append(idx)
            continue
        all_unsafe_indices.append(idx)
        if any(is_policy_discriminative({p: labels.get(p) for p in ID_POLICIES.get(sid, []) if p in labels}) for sid, labels in all_policy_labels.items()):
            id_discriminative_pool.append(idx)
        for sid, labels in all_policy_labels.items():
            ood_labels = {p: labels.get(p) for p in OOD_POLICIES.get(sid, []) if p in labels}
            id_labels = {p: labels.get(p) for p in ID_POLICIES.get(sid, []) if p in labels}
            if ood_labels and "block" in (set(ood_labels.values()) | set(id_labels.values())) and "pass" in (set(ood_labels.values()) | set(id_labels.values())):
                ood_discriminative_pool.append(idx)
                break

    all_id_instances = generate_discriminative_instances(id_discriminative_pool, record_map, section_map, ID_POLICIES, "id_all", seed=config.seed)
    ood_test_instances = generate_ood_instances(ood_discriminative_pool, record_map, section_map, seed=config.seed + 1)

    id_by_image = group_instances_by_image(all_id_instances)
    ood_by_image = group_instances_by_image(ood_test_instances)
    id_test_image_ids, _, _ = select_image_subset_for_balanced_target(id_by_image, config.id_test_size, seed=config.seed + 100)
    id_test_balanced, _, _ = balance_instances([inst for img in id_test_image_ids for inst in id_by_image[img]], target_size=config.id_test_size, seed=config.seed + 101)
    ood_test_image_ids, _, _ = select_image_subset_for_balanced_target(ood_by_image, config.ood_test_size, seed=config.seed + 300)
    ood_test_balanced, _, _ = balance_instances([inst for img in ood_test_image_ids for inst in ood_by_image[img]], target_size=config.ood_test_size, seed=config.seed + 301)
    eval_reserved_images = set(id_test_image_ids) | set(ood_test_image_ids)
    remaining_id_by_image = {img: items for img, items in id_by_image.items() if img not in eval_reserved_images}
    sft_image_ids, _, _ = select_image_subset_for_balanced_target(remaining_id_by_image, config.sft_size, seed=config.seed + 150)
    sft_balanced, _, _ = balance_instances([inst for img in sft_image_ids for inst in remaining_id_by_image[img]], target_size=config.sft_size, seed=config.seed + 151)
    rl_reserved_images = eval_reserved_images | set(sft_image_ids)
    rl_pool = generate_rl_instances(common_indices, record_map, section_map, rl_reserved_images, seed=config.seed + 2)
    rl_balanced, _, _ = balance_instances(rl_pool, target_size=None, seed=config.seed + 200)

    for inst in id_test_balanced:
        inst["split_type"] = "id_test"
    for inst in ood_test_balanced:
        inst["split_type"] = "ood_test"
    for inst in sft_balanced:
        inst["split_type"] = "sft"
    for inst in rl_balanced:
        inst["split_type"] = "rl"

    splits = {"id_test": id_test_balanced, "ood_test": ood_test_balanced, "sft": sft_balanced, "rl": rl_balanced}
    image_loader = ImageLoader(config.image_source, config.image_source_type)
    hf_datasets = {name: generate_hf_dataset(instances, image_loader, name) for name, instances in splits.items()}

    for split_name, ds in hf_datasets.items():
        split_dir = output_dir / split_name
        parquet_path = output_dir / f"{split_name}.parquet"
        jsonl_path = output_dir / f"{split_name}.jsonl"
        if split_dir.exists():
            shutil.rmtree(split_dir)
        if parquet_path.exists():
            parquet_path.unlink()
        if jsonl_path.exists():
            jsonl_path.unlink()
        ds.save_to_disk(str(split_dir))
        ds.to_parquet(str(parquet_path))
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for inst in splits[split_name]:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    with open(output_dir / "policy_catalog.json", "w", encoding="utf-8") as f:
        json.dump(build_policy_catalog(section_map), f, indent=2, ensure_ascii=False)
    with open(output_dir / "policy_split.json", "w", encoding="utf-8") as f:
        json.dump({"id_policies": {str(k): v for k, v in ID_POLICIES.items()}, "ood_policies": {str(k): v for k, v in OOD_POLICIES.items()}}, f, indent=2, ensure_ascii=False)
    stats = {split_name: compute_stats(instances) for split_name, instances in splits.items()}
    stats["pools"] = {"common_indices": len(common_indices), "id_discriminative": len(id_discriminative_pool), "ood_discriminative": len(ood_discriminative_pool), "safe": len(safe_indices), "unsafe": len(all_unsafe_indices)}
    stats["vote_stats"] = global_vote_stats
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return splits, hf_datasets, stats
