"""The shape every provider returns.

A frozen dataclass — constructed once per call, never mutated. Free
.to_dict() via dataclasses.asdict; __str__ formats the human-friendly
display that runner.py and call_logger.py rely on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True, slots=True)
class QueryResult:
    content: Any  # str | parsed BaseModel for structured output
    msg: str
    system_msg: str
    new_msg_history: List[Dict[str, Any]]
    model_name: str
    kwargs: Dict[str, Any]
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    thought: str = ""
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model_posteriors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = ["=" * 80, f"Model: {self.model_name}", f"Total Cost: ${self.cost:.4f}"]
        lines.append(f"  Input: ${self.input_cost:.4f} ({self.input_tokens} tokens)")
        lines.append(f"  Output: ${self.output_cost:.4f} ({self.output_tokens} tokens)")
        if self.output_tokens > 0:
            ratio = self.thinking_tokens / self.output_tokens
            lines.append(f"  --> Thinking tokens: {self.thinking_tokens} ({ratio:.2f})")
        if self.cache_read_tokens or self.cache_creation_tokens:
            lines.append(
                f"  Cache read: {self.cache_read_tokens} tokens, "
                f"created: {self.cache_creation_tokens} tokens"
            )
        lines.append("-" * 80)
        if self.thought:
            lines.append("Thought:")
            lines.append(self.thought)
            lines.append("-" * 80)
        lines.append("Content:")
        lines.append(str(self.content))
        if self.model_posteriors:
            lines.append("-" * 80)
            lines.append("Model Posteriors:")
            for name, prob in self.model_posteriors.items():
                lines.append(f"  {name}: {prob:.4f}")
        lines.append("=" * 80)
        return "\n".join(lines)
