import argparse
from pathlib import Path
from typing import Sequence

from vllm_guard.common.constants import REPO_ROOT
from vllm_guard.validator.common import ValidationIssue, ValidationReport


CHECK_TARGETS = [
    REPO_ROOT / "vllm_guard",
    REPO_ROOT / "configs",
    REPO_ROOT / "docs",
]


def validate_codebase() -> ValidationReport:
    issues: list[ValidationIssue] = []
    py_files = []
    for target in CHECK_TARGETS:
        if target.is_dir():
            py_files.extend(target.rglob("*.py"))
        elif target.exists():
            py_files.append(target)

    legacy_hf_repo = "AdaptivePolicy" + "/adaptive-policy-v2.7-withreason"
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        if "outputs/v2.7" in text and "v2.7_withreason" not in text:
            issues.append(
                ValidationIssue("error", "Legacy v2.7-only path reference found", str(path))
            )
        if (
            "sft_think" in text
            and "target_text" not in text
            and path.name.endswith(".py")
            and "validator" not in str(path)
            and "common/constants.py" not in str(path)
            and "training/formatting.py" not in str(path)
            and "data_curation/builder.py" not in str(path)
        ):
            issues.append(
                ValidationIssue("warning", "sft_think referenced without explicit target_text handling", str(path))
            )
        if legacy_hf_repo in text:
            issues.append(ValidationIssue("warning", "Legacy HF dataset reference found", str(path)))

    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
        if "PolicyShiftBench/PolicyShiftBench" not in readme_text:
            issues.append(
                ValidationIssue(
                    "warning",
                    "README should point to the official PolicyShiftBench HF dataset",
                    str(readme_path),
                )
            )

    docs_ok = (REPO_ROOT / "docs" / "architecture.md").exists()
    if not docs_ok:
        issues.append(ValidationIssue("error", "architecture.md is missing", str(REPO_ROOT / "docs")))

    return ValidationReport(
        name="Code Rules Validator",
        ok=not any(issue.severity == "error" for issue in issues),
        summary={
            "checked_files": len(py_files),
            "canonical_dataset_only": True,
            "requires_architecture_doc": docs_ok,
            "legacy_entrypoints_checked": 0,
        },
        issues=issues,
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate repository code against current adaptive-policy rules")
    p.add_argument("--output", default=str(REPO_ROOT / "logs" / "code_validation_report.json"))
    p.add_argument("--output-md", default=str(REPO_ROOT / "logs" / "code_validation_report.md"))
    args = p.parse_args(argv)
    report = validate_codebase()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.write_json(args.output)
    report.write_md(args.output_md)
    print(report.to_dict())


if __name__ == "__main__":
    main()
