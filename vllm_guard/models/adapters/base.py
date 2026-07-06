from typing import Any, Protocol


class ModelAdapter(Protocol):
    """Minimal adapter contract for benchmark inference."""

    def generate(self, prompts_and_images: list[dict[str, Any]]) -> list[str]:
        ...

