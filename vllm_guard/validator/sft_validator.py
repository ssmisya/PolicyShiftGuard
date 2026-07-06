import argparse
from pathlib import Path
from typing import Sequence

from vllm_guard.common.constants import REPO_ROOT
from vllm_guard.validator.common import ValidationIssue, ValidationReport


REQUIRED_MODEL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def validate_sft_code() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    runner_file = REPO_ROOT / "vllm_guard" / "training" / "sft" / "runner.py"
    pipeline_file = REPO_ROOT / "vllm_guard" / "training" / "sft" / "pipeline.py"
    runner_text = runner_file.read_text(encoding="utf-8")
    pipeline_text = pipeline_file.read_text(encoding="utf-8")

    if "build_output_instructions" not in pipeline_text or "build_supervised_answer" not in pipeline_text:
        issues.append(ValidationIssue("error", "canonical SFT pipeline is not wired to shared SFT formatter", str(pipeline_file)))
    if "vllm_guard.training.eval_callback" not in pipeline_text:
        issues.append(ValidationIssue("error", "canonical SFT pipeline is not wired to canonical miniset callback", str(pipeline_file)))
    if "eval_steps_callback: int = field(default=50" not in pipeline_text:
        issues.append(ValidationIssue("error", "canonical SFT pipeline default miniset eval_steps should be 50", str(pipeline_file)))
    if "CANONICAL_DATASET_DIR" not in runner_text and "v2.7_withreason" not in runner_text:
        issues.append(ValidationIssue("error", "canonical SFT runner is not defaulting to v2.7_withreason", str(runner_file)))
    if "\"sft_think\"" not in runner_text and "sft_think" not in pipeline_text and "train_split" not in runner_text:
        issues.append(ValidationIssue("warning", "SFT code path does not mention think split explicitly"))
    return issues


def validate_sft_output(model_dir: str | Path) -> ValidationReport:
    path = Path(model_dir)
    issues = validate_sft_code()
    summary = {
        "model_dir": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        issues.append(ValidationIssue("error", "model output directory does not exist", str(path)))
        return ValidationReport("SFT Output Validator", False, summary, issues)

    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).exists()]
    summary["missing_files"] = missing
    if missing:
        issues.append(ValidationIssue("error", "missing required model/tokenizer artifacts", str(path)))

    ckpts = sorted(p.name for p in path.glob("checkpoint-*") if p.is_dir())
    summary["checkpoint_count"] = len(ckpts)
    summary["checkpoints"] = ckpts[:10]
    return ValidationReport("SFT Output Validator", not issues, summary, issues)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate SFT training outputs")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    report = validate_sft_output(args.model_dir)
    out = Path(args.output) if args.output else Path(args.model_dir) / "validation_report.json"
    report.write_json(out)
    report.write_md(out.with_suffix(".md"))
    print(report.to_dict())


if __name__ == "__main__":
    main()
