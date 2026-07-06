import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from vllm_guard.validator.common import ValidationIssue, ValidationReport


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def validate_rl_dataset(rl_dir: str | Path) -> ValidationReport:
    root = Path(rl_dir)
    issues: list[ValidationIssue] = []
    summary = {"rl_dir": str(root), "exists": root.exists()}
    if not root.exists():
        issues.append(ValidationIssue("error", "RL dataset directory does not exist", str(root)))
        return ValidationReport("RL Validator", False, summary, issues)

    train_file = root / "train.parquet"
    val_files = sorted(root.glob("val*.parquet"))
    for file in [train_file, *val_files]:
        if not file.exists():
            issues.append(ValidationIssue("error", "missing verl parquet", str(file)))
    if not val_files:
        issues.append(ValidationIssue("error", "missing verl validation parquet(s)", str(root / "val*.parquet")))
    if issues:
        return ValidationReport("RL Validator", False, summary, issues)

    train_df = pd.read_parquet(train_file)
    summary["train_rows"] = len(train_df)
    summary["columns"] = list(train_df.columns)
    summary["val_files"] = [str(p) for p in val_files]
    required_cols = {"data_source", "prompt", "images", "reward_model", "extra_info"}
    missing = sorted(required_cols - set(train_df.columns))
    summary["missing_columns"] = missing
    if missing:
        issues.append(ValidationIssue("error", "verl parquet missing required columns", str(train_file)))

    if len(train_df):
        raw_sample = train_df.iloc[0].to_dict()
        sample = {
            "prompt": _json_safe(raw_sample.get("prompt")),
            "data_source": _json_safe(raw_sample.get("data_source")),
            "reward_model": _json_safe(raw_sample.get("reward_model")),
            "extra_info": _json_safe(raw_sample.get("extra_info")),
            "n_images": len(raw_sample.get("images", []) or []),
        }
    else:
        sample = {}

    stats_file = root / "stats.json"
    configured_val_splits: list[str] = []
    dataset_name = "v2.7_withreason"
    if stats_file.exists():
        summary["stats"] = json.loads(stats_file.read_text())
        configured_val_splits = list(summary["stats"].get("val_splits", []) or [])
        dataset_path = str(summary["stats"].get("dataset_path", "") or "")
        if dataset_path:
            dataset_name = Path(dataset_path).name
    else:
        issues.append(ValidationIssue("warning", "stats.json missing", str(stats_file)))
        if "data_source" in train_df.columns and len(train_df):
            parts = str(train_df.iloc[0]["data_source"]).split("/")
            if len(parts) >= 3:
                dataset_name = parts[1]
    summary["dataset_name"] = dataset_name

    expected_val_names = (
        {f"val_{split}.parquet" for split in configured_val_splits}
        if configured_val_splits
        else {"val_id_test_mini.parquet", "val_ood_test_mini.parquet"}
    )
    actual_val_names = {p.name for p in val_files}
    summary["expected_val_files"] = sorted(expected_val_names)
    summary["actual_val_files"] = sorted(actual_val_names)
    missing_val_names = sorted(expected_val_names - actual_val_names)
    if missing_val_names:
        issues.append(
            ValidationIssue(
                "error",
                "missing separated RL validation parquet(s)",
                ", ".join(missing_val_names),
            )
        )

    for val_file in val_files:
        val_df = pd.read_parquet(val_file)
        if not len(val_df):
            issues.append(ValidationIssue("error", "empty RL validation parquet", str(val_file)))
            continue
        data_sources = sorted({str(x) for x in val_df["data_source"].tolist()}) if "data_source" in val_df.columns else []
        summary.setdefault("val_data_sources", {})[val_file.name] = data_sources
        expected_source = f"adaptive_policy/{dataset_name}/{val_file.stem.removeprefix('val_')}"
        if data_sources != [expected_source]:
            issues.append(
                ValidationIssue(
                    "error",
                    "RL validation parquet data_source does not match split-specific expectation",
                    f"{val_file}: expected {expected_source}, got {data_sources}",
                )
            )

    if (summary.get("stats") or {}).get("pair_pack_train"):
        pair_issues = []
        if len(train_df) % 2:
            pair_issues.append(f"odd train row count: {len(train_df)}")
        for idx in range(0, len(train_df) - 1, 2):
            left = train_df.iloc[idx].get("extra_info") or {}
            right = train_df.iloc[idx + 1].get("extra_info") or {}
            left_group = left.get("boundary_group_id")
            right_group = right.get("boundary_group_id")
            roles = {left.get("boundary_pair_role"), right.get("boundary_pair_role")}
            if left_group != right_group or roles != {"block", "pass"}:
                pair_issues.append(
                    f"rows {idx},{idx + 1}: {left_group}/{left.get('boundary_pair_role')} "
                    f"vs {right_group}/{right.get('boundary_pair_role')}"
                )
                if len(pair_issues) >= 5:
                    break
        summary["pair_pack_issues"] = pair_issues
        if pair_issues:
            issues.append(
                ValidationIssue(
                    "error",
                    "pair_pack_train is enabled but train.parquet is not adjacent block/pass pairs",
                    "; ".join(pair_issues),
                )
            )

    return ValidationReport("RL Validator", not any(i.severity == "error" for i in issues), summary, issues, {"sample": sample})


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate prepared RL/verl dataset artifacts")
    p.add_argument("--rl-dir", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    report = validate_rl_dataset(args.rl_dir)
    out = Path(args.output) if args.output else Path(args.rl_dir) / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(out)
    report.write_md(out.with_suffix(".md"))
    print(report.to_dict())


if __name__ == "__main__":
    main()
