import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationIssue:
    severity: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    name: str
    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    examples: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "summary": self.summary,
            "issues": [asdict(x) for x in self.issues],
            "examples": self.examples,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    def write_md(self, path: str | Path) -> None:
        lines = [f"# {self.name}", "", f"- ok: `{self.ok}`"]
        for key, value in self.summary.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
        lines.append("## Issues")
        if not self.issues:
            lines.append("- none")
        else:
            for issue in self.issues:
                suffix = f" [{issue.path}]" if issue.path else ""
                lines.append(f"- `{issue.severity}` {issue.message}{suffix}")
        if self.examples:
            lines.append("")
            lines.append("## Examples")
            for key, value in self.examples.items():
                lines.append(f"### {key}")
                lines.append("```json")
                lines.append(json.dumps(value, indent=2, ensure_ascii=False))
                lines.append("```")
        Path(path).write_text("\n".join(lines) + "\n")

