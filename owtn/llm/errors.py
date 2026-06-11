"""LLM call error types that carry partial telemetry.

When a structured-output call returns a 200 but fails Pydantic validation,
the provider has already paid for the response and has the raw payload,
token counts, and cost in scope. Without a way to thread that data out,
api.py:_log never runs and the call is logged as if it never happened.

LLMValidationError is the wrapper providers raise on validation failure.
api.py catches it, writes a diagnostic yaml with the raw payload + the
validation error, then re-raises so existing retry loops continue to work.
"""

from __future__ import annotations

from typing import Any


class LLMValidationError(Exception):
    """Provider call returned a 200 but the body failed structured-output
    validation. Carries the raw payload and partial token/cost accounting
    so the api.py logger can record diagnostics before propagating.

    Existing exception-handlers (e.g. `try/except Exception` in retry loops)
    treat this exactly like the underlying ValidationError — the original
    error message is preserved in `__str__`. Inspect `.cause` for the
    original error and `.raw_output` for the model's actual emission.
    """

    def __init__(
        self,
        *,
        cause: Exception,
        raw_output: str,
        model_name: str,
        msg: str,
        system_msg: str,
        thought: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        cost: float = 0.0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.raw_output = raw_output
        self.model_name = model_name
        self.msg = msg
        self.system_msg = system_msg
        self.thought = thought
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens
        self.cost = cost
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.kwargs = kwargs or {}
