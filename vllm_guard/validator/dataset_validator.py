import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from datasets import load_dataset, load_from_disk

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.datasets.registry import get_dataset_spec
from vllm_guard.datasets.schema import BASE_INSTANCE_FIELDS, THINK_FIELDS, has_required_fields
from vllm_guard.evaluation.miniset import create_minisets
from vllm_guard.validator.common import ValidationIssue, ValidationReport
from vllm_guard.validator.visualization import save_dataset_examples_visualization


def _load_split(root: Path, split: str):
    split_dir = root / split
    split_parquet = root / f"{split}.parquet"
    if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
        return load_from_disk(str(split_dir))
    if split_parquet.exists():
        return load_dataset("parquet", data_files=str(split_parquet), split="train")
    raise FileNotFoundError(f"Could not find split '{split}' under {root}")


def validate_dataset(dataset_path: str | Path = CANONICAL_DATASET_DIR) -> ValidationReport:
    root = Path(dataset_path)
    spec = get_dataset_spec()
    issues: list[ValidationIssue] = []
    summary = {}
    examples = {}

    split_names = [*spec.eval_splits, spec.no_think_split.name, spec.think_split.name, spec.rl_split]
    unique_images: dict[str, set[int]] = {}
    unique_instances: dict[str, set[tuple[int, int, str]]] = {}

    for split in split_names:
        ds = _load_split(root, split)
        cols = ds.column_names
        unique_images[split] = {int(x) for x in ds["image_idx"]}
        unique_instances[split] = {
            (int(image_idx), int(section_id), str(policy_name))
            for image_idx, section_id, policy_name in zip(
                ds["image_idx"], ds["section_id"], ds["policy_name"]
            )
        }
        summary[f"{split}_rows"] = len(ds)
        summary[f"{split}_unique_images"] = len(unique_images[split])
        label_counter = Counter(ds["label"])
        summary[f"{split}_labels"] = dict(label_counter)
        if split == spec.no_think_split.name:
            if not has_required_fields(cols, BASE_INSTANCE_FIELDS):
                issues.append(ValidationIssue("error", "sft is missing required no-think fields", split))
            if any(field in cols for field in THINK_FIELDS):
                issues.append(ValidationIssue("error", "sft should not contain reason/think-only fields", split))
            examples["sft_example"] = {k: ds[0][k] for k in ("answer", "label", "policy_name", "violated_categories")}
        elif split == spec.think_split.name:
            if not has_required_fields(cols, BASE_INSTANCE_FIELDS + THINK_FIELDS):
                issues.append(ValidationIssue("error", "sft_think is missing reason/target_text fields", split))
            examples["sft_think_example"] = {k: ds[0][k] for k in ("answer", "reason", "target_text")}

    # Split hygiene
    for a, b in [("id_test", "sft"), ("id_test", "rl"), ("ood_test", "sft"), ("ood_test", "rl"), ("sft", "rl")]:
        overlap = unique_images[a] & unique_images[b]
        summary[f"image_overlap_{a}_{b}"] = len(overlap)
        if overlap:
            issues.append(ValidationIssue("error", "Image-level leakage detected", f"{a} vs {b}"))

    id_ood_overlap = unique_images["id_test"] & unique_images["ood_test"]
    summary["image_overlap_id_ood"] = len(id_ood_overlap)

    # Instance-level uniqueness
    all_keys = defaultdict(list)
    for split, keys in unique_instances.items():
        for key in keys:
            all_keys[key].append(split)
    dup_instances = {
        k: v
        for k, v in all_keys.items()
        if len(v) > 1 and set(v) != {"sft", "sft_think"}
    }
    summary["cross_split_instance_duplicates"] = len(dup_instances)
    if dup_instances:
        issues.append(ValidationIssue("error", "Instance-level duplication detected across splits"))
    summary["allowed_sft_sft_think_instance_overlap"] = len(unique_instances["sft"] & unique_instances["sft_think"])

    # Balance
    for split, target in [("id_test", (500, 500)), ("ood_test", (500, 500)), ("sft", (1000, 1000))]:
        ds = _load_split(root, split)
        c = Counter(ds["label"])
        if (c.get("block", 0), c.get("pass", 0)) != target:
            issues.append(ValidationIssue("error", f"{split} label balance mismatch", split))

    # Samples for inspection
    examples["id_test_example"] = {
        k: _load_split(root, "id_test")[0][k] for k in ("answer", "policy_name", "section_id", "label")
    }
    examples["rl_example"] = {
        k: _load_split(root, "rl")[0][k] for k in ("answer", "tier", "policy_name", "label")
    }

    mini_outputs = create_minisets(str(root), sample_size=100, seed=42)
    summary["miniset_outputs"] = mini_outputs
    for split in ("id_test", "ood_test"):
        mini_path = root / f"{split}_mini.parquet"
        mini_ds = _load_split(root, f"{split}_mini")
        summary[f"{split}_mini_rows"] = len(mini_ds)
        if len(mini_ds) != 100:
            issues.append(ValidationIssue("error", f"{split}_mini should contain exactly 100 instances", str(mini_path)))
        examples[f"{split}_mini_example"] = {
            k: mini_ds[0][k] for k in ("answer", "policy_name", "section_id", "label")
        }

    return ValidationReport(
        name="Dataset Validator",
        ok=not any(issue.severity == "error" for issue in issues),
        summary=summary,
        issues=issues,
        examples=examples,
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate canonical adaptive-policy dataset outputs")
    p.add_argument("--dataset-path", default=str(CANONICAL_DATASET_DIR))
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_dataset(dataset_path)
    vis_manifest = save_dataset_examples_visualization(dataset_path, output_dir / "dataset_examples.html")
    report.summary["examples_visualization"] = vis_manifest["output_html"]
    report.summary["examples_visualization_counts"] = vis_manifest["splits"]
    report.write_json(output_dir / "dataset_validation_report.json")
    report.write_md(output_dir / "dataset_validation_report.md")
    print(report.to_dict())


if __name__ == "__main__":
    main()
