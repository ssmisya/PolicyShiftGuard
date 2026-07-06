from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParsedPrediction:
    valid: bool
    decision: Optional[str]
    predicted_categories: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    raw_text: str = ""
    latency: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

