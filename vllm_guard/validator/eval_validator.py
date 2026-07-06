import argparse
import json
from pathlib import Path
from typing import Sequence

from vllm_guard.common.constants import CANONICAL_DATASET_DIR
from vllm_guard.common.constants import REPO_ROOT
from vllm_guard.validator.common import ValidationIssue, ValidationReport
from vllm_guard.validator.visualization import save_eval_examples_visualization


REQUIRED_SPLIT_FILES = [
    "results.jsonl",
    "metrics.json",
    "table_main.tex",
    "table_summary.tex",
    "table_confusion.tex",
]

REQUIRED_TOP_LEVEL_FILES = [
    "case_visualizations.html",
    "case_visualizations.json",
]


def validate_eval_code() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    runner_file = REPO_ROOT / "vllm_guard" / "evaluation" / "runner.py"
    pipeline_file = REPO_ROOT / "vllm_guard" / "evaluation" / "pipeline.py"
    reporting_file = REPO_ROOT / "vllm_guard" / "evaluation" / "reporting.py"
    adapters_file = REPO_ROOT / "vllm_guard" / "evaluation" / "adapters.py"
    runner_text = runner_file.read_text(encoding="utf-8")
    pipeline_text = pipeline_file.read_text(encoding="utf-8")
    reporting_text = reporting_file.read_text(encoding="utf-8")
    adapters_text = adapters_file.read_text(encoding="utf-8")
    if "adaptive_policy_v2.7_withreason" not in runner_text and "get_benchmark_spec" not in runner_text:
        issues.append(ValidationIssue("error", "canonical eval runner is not defaulting to the benchmark registry", str(runner_file)))
    if "create_adapter" not in pipeline_text or "save_results_bundle" not in pipeline_text:
        issues.append(ValidationIssue("error", "canonical eval pipeline is incomplete", str(pipeline_file)))
    if "compute_all_metrics" not in reporting_text or "save_tables" not in reporting_text:
        issues.append(ValidationIssue("error", "canonical reporting is not self-contained", str(reporting_file)))
    if "vllm_guard.evaluation.model_registry" not in adapters_text:
        issues.append(ValidationIssue("error", "canonical adapters still rely on old model registry", str(adapters_file)))
    return issues


def validate_eval_output(eval_root: str | Path) -> ValidationReport:
    root = Path(eval_root)
    issues = validate_eval_code()
    summary = {"eval_root": str(root), "exists": root.exists()}
    if not root.exists():
        issues.append(ValidationIssue("error", "eval root does not exist", str(root)))
        return ValidationReport("Eval Validator", False, summary, issues)

    split_dirs = list(root.glob("*/id_test")) + list(root.glob("*/ood_test"))
    summary["evaluated_splits"] = len(split_dirs)
    for split_dir in split_dirs:
        for name in REQUIRED_SPLIT_FILES:
            if not (split_dir / name).exists():
                issues.append(ValidationIssue("error", f"missing {name}", str(split_dir)))
        metrics_path = split_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if "overall" not in metrics or "per_section" not in metrics:
                issues.append(ValidationIssue("error", "metrics.json missing required sections", str(metrics_path)))

    tables_dir = root / "tables"
    summary["tables_dir_exists"] = tables_dir.exists()
    if not tables_dir.exists():
        issues.append(ValidationIssue("error", "summary tables directory missing", str(tables_dir)))
    else:
        expected = [f"table{i}_{name}.{ext}" for i, name in [
            (1, "overall_id"), (2, "overall_ood"), (3, "id_ood_comparison"),
            (4, "thinking_comparison"), (5, "per_category_id"), (6, "per_category_ood"),
        ] for ext in ("md", "tex")]
        missing = [name for name in expected if not (tables_dir / name).exists()]
        summary["missing_tables"] = missing
        for name in missing:
            issues.append(ValidationIssue("error", "missing summary table", str(tables_dir / name)))

    vis_manifest = save_eval_examples_visualization(root, CANONICAL_DATASET_DIR, root / "case_visualizations.html")
    summary["case_visualizations"] = vis_manifest["output_html"]
    summary["case_visualization_counts"] = vis_manifest["splits"]
    for name in REQUIRED_TOP_LEVEL_FILES:
        if not (root / name).exists():
            issues.append(ValidationIssue("error", "missing eval visualization artifact", str(root / name)))

    return ValidationReport("Eval Validator", not issues, summary, issues)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate evaluation outputs and summary tables")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    report = validate_eval_output(args.eval_root)
    out = Path(args.output) if args.output else Path(args.eval_root) / "validation_report.json"
    report.write_json(out)
    report.write_md(out.with_suffix(".md"))
    print(report.to_dict())


if __name__ == "__main__":
    main()
