"""Per-call LLM logging for debugging.

Configure with a log directory via context var. All LLM calls through
owtn.llm.query get logged to <log_dir>/llm/<model>/NNNN.yaml.

Usage:
    from owtn.llm.call_logger import llm_log_dir, llm_context
    llm_log_dir.set("/path/to/results/run_xxx/stage_1")
    llm_context.set({"role": "judge", "judge_id": "mira-okonkwo"})
    result = await query_async(...)  # automatically logged
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

llm_log_dir: ContextVar[str | None] = ContextVar("llm_log_dir", default=None)
llm_context: ContextVar[dict] = ContextVar("llm_context", default={})

_counter = itertools.count(1)
_counter_lock = threading.Lock()
_pid = os.getpid()


def _next_id() -> str:
    """Return a process-unique call ID (pid_seq) to avoid collisions across subprocesses."""
    with _counter_lock:
        seq = next(_counter)
    return f"{_pid}_{seq:04d}"


class _LiteralStr(str):
    """String subclass that yaml.dump renders with literal block style (|)."""


def _literal_representer(dumper, data):
    # PyYAML falls back to escaped double-quoted style if any line has
    # trailing whitespace (space_break rule). LLM output frequently has
    # invisible trailing spaces, which produces unreadable logs. Strip them
    # so the readable |-block style applies.
    cleaned = "\n".join(line.rstrip() for line in data.splitlines())
    return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")


_dumper = yaml.SafeDumper
_dumper.add_representer(_LiteralStr, _literal_representer)


def log_call(
    *,
    model: str,
    provider: str,
    system_msg: str,
    user_msg: str,
    content: str | None,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost: float,
    duration_s: float,
    thought: str = "",
    kwargs: dict | None = None,
    raw_output: str | None = None,
    error: str | None = None,
    system_prefix: str = "",
) -> None:
    """Log a single LLM call to disk. No-op if llm_log_dir is unset.

    `raw_output` is the model's literal emission for structured-output calls
    (the JSON it produced before Pydantic parsing). When set, it replaces
    `content` in the yaml's `output:` field — this preserves what the model
    actually said, not the parsed-then-stringified repr.

    `error` marks the call as a validation failure: the yaml gets
    `status: validation_failed` and an `error:` field. The diagnostic yaml
    still contains the raw payload, tokens, and cost.

    `system_prefix` is the cache_control block sent to the API (e.g. judge
    persona prepended to keep prompt-caching engaged). Logged separately
    from `system_msg` so the saved yaml shows the full system surface the
    model actually saw — without this field, judge yamls had `system_msg:""`
    even though the persona was being threaded correctly to the API.
    """
    log_dir = llm_log_dir.get(None)
    if log_dir is None:
        return

    call_id = _next_id()
    ctx = llm_context.get({})

    model_dir_name = model.replace("/", "_").replace(":", "_")
    out_dir = Path(log_dir) / "llm" / model_dir_name
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    record = {
        "call_id": call_id,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "model": model,
        "provider": provider,
        "tokens": {
            "input": int(input_tokens),
            "output": int(output_tokens),
            "thinking": int(thinking_tokens),
            "cache_read": int(cache_read_tokens),
            "cache_creation": int(cache_creation_tokens),
        },
        "cost": round(float(cost), 6),
        "duration_s": round(float(duration_s), 3),
    }
    if error is not None:
        record["status"] = "validation_failed"

    # Merge context vars (role, judge_id, operator, generation, etc.)
    # Cast numpy types to native Python types for YAML serialization.
    for k, v in ctx.items():
        if v is not None:
            record[k] = v.item() if hasattr(v, "item") else v

    if kwargs:
        filtered = {}
        for k, v in kwargs.items():
            if k in ("temperature", "max_tokens", "reasoning_effort"):
                filtered[k] = float(v) if hasattr(v, "item") else v
        if filtered:
            record["kwargs"] = filtered

    # Use literal block style for multiline text fields. raw_output (when
    # available) preserves the model's literal emission; falls back to the
    # parsed-and-stringified content for non-structured calls.
    if system_prefix:
        record["system_prefix"] = _LiteralStr(system_prefix)
    record["system_msg"] = _LiteralStr(system_msg)
    record["user_msg"] = _LiteralStr(user_msg)
    output_text = raw_output if raw_output else (content or "")
    record["output"] = _LiteralStr(output_text)
    if thought:
        record["thought"] = _LiteralStr(thought)
    if error is not None:
        record["error"] = _LiteralStr(error)

    path = out_dir / f"{call_id}.yaml"
    try:
        path.write_text(
            yaml.dump(
                record,
                Dumper=_dumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        )
    except Exception as e:
        logger.warning("Failed to write LLM call log %s: %s", path, e)
