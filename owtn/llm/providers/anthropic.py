"""Anthropic + Bedrock providers.

Native structured output via forced tool use. Each `output_model` is
declared as a tool whose `input_schema` is the Pydantic JSON schema;
`tool_choice` forces the model to emit a tool call in that exact shape.
No instructor injection — Anthropic's tool-use prompt internals are the
only "schema-related text" the model sees, and that's the API contract,
not our prompt.

system_prefix is sent as a separate cache_control content block so
prompt caching activates (saves ~80% input cost on repeated prefixes).

Both AnthropicProvider and BedrockProvider share the call shape; only
the SDK client construction differs (one uses ANTHROPIC_API_KEY, the
other uses AWS credentials).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Optional, Type

import anthropic
import backoff
from pydantic import BaseModel, ValidationError

from ..errors import LLMValidationError
from ..result import QueryResult
from .base import THINKING_TOKENS, TIMEOUT, resolve_effort, resolve_temperature
from .pricing import calculate_cost, is_reasoning_model, requires_adaptive_thinking

logger = logging.getLogger(__name__)

MAX_TRIES = 20
MAX_VALUE = 20
MAX_TIME = 600
ANTHROPIC_DEFAULT_MAX_TOKENS = 16384
ANTHROPIC_MAX_TOKENS_CEILING = 64000

# Force `tool_choice: tool` for output models that empirically skip the
# tool call (or emit empty tool input) under `auto` — typically when the
# user prompt's "respond with JSON" framing competes with the tool channel.
# See lab/issues/2026-04-30-stage-2-lineage-brief-tool-use-miss.md.
_FORCE_TOOL_CHOICE_MODELS = {
    "LineageBrief", "PopulationBrief",
    # Stage 3 casting argue — empirically returns empty `arguments={}` on
    # sonnet-4-6 under `auto`.
    "CastingArgueOutput",
}

_RETRY_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APIStatusError,
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
)


# Anthropic adaptive-thinking effort enum: low / medium / high / xhigh / max.
# `xhigh` is Opus 4.7 only per the API docs (legacy adaptive models like
# Sonnet 4.6 / Opus 4.6 only have low/medium/high/max). Today the adaptive
# path only fires for Opus 4.7, so the full 5-level passthrough is safe; if
# we migrate Sonnet 4.6 to adaptive in the future, xhigh will need a per-
# model downgrade. owtn `min` has no Anthropic counterpart and folds to low.
# Fine-grained budgets aren't expressible under adaptive — Anthropic doesn't
# expose them. Callers that pass `thinking_tokens` explicitly to an adaptive
# model are silently ignored on the budget axis (the effort enum still drives).
_ADAPTIVE_EFFORT = {
    "min": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def _resolve_thinking_blocks(
    *,
    api_model: str,
    effort: str,
    explicit_tokens: Optional[int],
    ceiling: int,
) -> tuple[Optional[dict], Optional[dict]]:
    """Decide the ``thinking`` and ``output_config`` blocks for an Anthropic call.

    Returns ``(thinking, output_config)``. Both ``None`` means thinking off.
    Three concrete shapes:

    - ``(None, None)`` — non-reasoning model, or ``effort="disabled"`` with no
      explicit budget override.
    - ``({"type": "enabled", "budget_tokens": N}, None)`` — legacy path
      (Sonnet 4.6, Haiku 4.5, Opus 4.6). Budget clamped below ``ceiling``;
      Anthropic rejects budget ≥ max_tokens.
    - ``({"type": "adaptive"}, {"effort": <low|medium|high>})`` — adaptive
      path (Opus 4.7+). ``explicit_tokens`` ignored; the effort enum drives.
    """
    if not is_reasoning_model(api_model):
        return None, None

    if requires_adaptive_thinking(api_model):
        if effort == "disabled":
            return None, None
        # ``display: "summarized"`` makes adaptive thinking visible as a
        # ``thinking`` content block on the response. Without it Opus 4.7
        # omits thinking from response.content entirely (thinking still
        # happens internally and is billed inside output_tokens, but our
        # _extract_text gets thought=""). Legacy adaptive models (Sonnet 4.6,
        # Opus 4.6) emit full thinking via the legacy budget path and don't
        # need this knob.
        return (
            {"type": "adaptive", "display": "summarized"},
            {"effort": _ADAPTIVE_EFFORT[effort]},
        )

    # Legacy budget-based path.
    if explicit_tokens is not None:
        budget = explicit_tokens if explicit_tokens < ceiling else 1024
        return {"type": "enabled", "budget_tokens": budget}, None
    if effort != "disabled":
        t = THINKING_TOKENS[effort]
        budget = t if t < ceiling else 1024
        return {"type": "enabled", "budget_tokens": budget}, None
    return None, None


def _add_user_cache_marker(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` with `cache_control: ephemeral` set on the
    last block of the most recent user-role message, stripped from any earlier
    user-role blocks so the request stays under Anthropic's 4-breakpoint
    limit as history grows across tool-use iterations.

    Pairing this with `_build_system`'s system-block marker creates two cache
    breakpoints: the system entry helps calls that share system but have
    different user msgs (judges with shared rubric); the user-msg entry
    catches calls that share both (picker retries, voice-loop iterations).
    Anthropic uses the largest matching prefix on read, so both engage
    correctly.
    """
    last_user = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user = i
    if last_user == -1:
        return list(messages)
    out: list[dict] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            out.append(m)
            continue
        content = m.get("content")
        # Normalize string content into a single text block.
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            out.append(m)
            continue
        new_blocks = []
        for j, b in enumerate(content):
            nb = {k: v for k, v in b.items() if k != "cache_control"}
            if i == last_user and j == len(content) - 1:
                nb["cache_control"] = {"type": "ephemeral"}
            new_blocks.append(nb)
        out.append({**m, "content": new_blocks})
    return out


def _build_system(system_msg: str, system_prefix: Optional[str]):
    """Build the `system` parameter with `cache_control: ephemeral` on the
    largest stable block.

    With a prefix, the prefix is the cached block (system_msg is the per-call
    suffix). Without a prefix, system_msg itself is wrapped with cache_control
    so prompt caching engages on repeat calls. Anthropic silently no-ops
    cache_control on prefixes below the per-model minimum (1024 tokens for
    Sonnet+, 2048 for Haiku), so the 1.25x cache_creation surcharge only
    applies when caching actually engages.

    Empty system_msg with no prefix returns the empty string — skip the
    content-block wrapper for truly empty calls.
    """
    if system_prefix:
        blocks = [
            {"type": "text", "text": system_prefix, "cache_control": {"type": "ephemeral"}}
        ]
        if system_msg:
            blocks.append({"type": "text", "text": system_msg})
        return blocks
    if system_msg:
        return [
            {"type": "text", "text": system_msg, "cache_control": {"type": "ephemeral"}}
        ]
    return system_msg


def get_anthropic_costs(response, model: str) -> dict:
    """Token counts and dollar costs from a messages.create response."""
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    input_cost, output_cost = calculate_cost(model, input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # Anthropic folds thinking into output_tokens (no separate count in
        # response.usage). Reporting 0 here is a logging convention — the
        # actual thinking tokens are billed and included in output_tokens.
        "thinking_tokens": 0,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cost": input_cost + output_cost,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
    }


def _backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _structured_output_tool(output_model: Type[BaseModel]) -> dict:
    """Build the forced-tool-use payload for a Pydantic output model.

    The tool name is the class name; the schema is the model's JSON schema
    (Anthropic accepts Draft-07 with $defs/$ref/oneOf, so no inlining needed).
    `description` falls back to a generic message if the model lacks a docstring.
    """
    return {
        "name": output_model.__name__,
        "description": (output_model.__doc__ or f"Return a {output_model.__name__}").strip(),
        "input_schema": output_model.model_json_schema(),
    }


def _find_tool_use_payload(response, output_model: Type[BaseModel]) -> Optional[dict]:
    """Locate the tool_use block matching `output_model.__name__` and return
    its raw input dict. Returns None if no matching block is found.

    Defensively unwraps a one-key `{<wrapper>: {...}}` shell when the
    wrapper key is not a field on `output_model`. Haiku-4.5 occasionally
    emits the tool input nested under `parameter` / `parameters` / `input`
    despite the schema being top-level; recovering the inner payload there
    is cheaper than retrying the call.
    """
    for block in response.content:
        if block.type == "tool_use" and block.name == output_model.__name__:
            payload = block.input
            if (
                isinstance(payload, dict)
                and len(payload) == 1
                and isinstance(next(iter(payload.values())), dict)
            ):
                only_key = next(iter(payload))
                if only_key not in output_model.model_fields:
                    payload = payload[only_key]
            return payload
    return None


def _extract_text(response) -> tuple[str, str]:
    """Return (content, thought). Anthropic puts thinking blocks before
    text blocks when extended thinking is enabled."""
    thought = ""
    content = ""
    for block in response.content:
        if block.type == "thinking":
            thought = block.thinking
        elif block.type == "text":
            content = block.text
    return content, thought


class AnthropicProvider:
    """Anthropic provider. Singleton clients per (sync/async)."""

    name = "anthropic"

    def __init__(self) -> None:
        self._sync_client: Optional[anthropic.Anthropic] = None
        self._async_client: Optional[anthropic.AsyncAnthropic] = None

    def _make_sync_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(timeout=TIMEOUT)

    def _make_async_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(timeout=TIMEOUT)

    def _sync(self) -> anthropic.Anthropic:
        if self._sync_client is None:
            self._sync_client = self._make_sync_client()
        return self._sync_client

    def _async(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = self._make_async_client()
        return self._async_client

    def build_call_kwargs(self, *, api_model: str, requested: Mapping[str, Any]) -> dict:
        """Anthropic shape: max_tokens (capped at 64k), ``thinking`` and
        (for Opus 4.7+) ``output_config`` blocks for extended-thinking models,
        top_p/top_k forbidden under thinking.

        Thinking-block resolution:
          - explicit `thinking_tokens` int wins on legacy models (fine-grained)
          - else falls back to THINKING_TOKENS[reasoning_effort] for back-compat
          - on adaptive-thinking models the effort enum drives ``output_config``
            (``thinking_tokens`` is silently ignored — adaptive has no budget axis)
          - else thinking is off
        """
        effort = resolve_effort(api_model, requested.get("reasoning_effort", "disabled"))
        out: dict = {}
        if (v := requested.get("max_tokens")) is not None:
            out["max_tokens"] = min(v, ANTHROPIC_MAX_TOKENS_CEILING)

        temp = resolve_temperature(api_model, requested.get("temperature"), effort)
        if temp is not None:
            out["temperature"] = temp

        thinking, output_config = _resolve_thinking_blocks(
            api_model=api_model,
            effort=effort,
            explicit_tokens=requested.get("thinking_tokens"),
            ceiling=out.get("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS),
        )
        if thinking is not None:
            out["thinking"] = thinking
        if output_config is not None:
            out["output_config"] = output_config
        if thinking is None:
            # top_p/top_k are forbidden when thinking is enabled.
            if (v := requested.get("top_p")) is not None:
                out["top_p"] = v
            if (v := requested.get("top_k")) is not None:
                out["top_k"] = v
        return out

    @backoff.on_exception(
        backoff.expo, _RETRY_EXCEPTIONS,
        max_tries=MAX_TRIES, max_value=MAX_VALUE, max_time=MAX_TIME,
        on_backoff=_backoff_handler,
    )
    def query(
        self,
        *,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        system_prefix: Optional[str],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
        client: Optional[anthropic.Anthropic] = None,
    ) -> QueryResult:
        client = client or self._sync()
        new_msg_history = msg_history + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]
        call_kwargs, extra_headers = self._prepare_call_kwargs(model, kwargs, output_model)
        response = client.messages.create(
            model=model,
            system=_build_system(system_msg, system_prefix),
            messages=_add_user_cache_marker(new_msg_history),
            extra_headers=extra_headers,
            **call_kwargs,
        )
        return self._build_result(
            response=response, model=model, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, output_model=output_model, kwargs=kwargs,
        )

    @backoff.on_exception(
        backoff.expo, _RETRY_EXCEPTIONS,
        max_tries=MAX_TRIES, max_value=MAX_VALUE, max_time=MAX_TIME,
        on_backoff=_backoff_handler,
    )
    async def query_async(
        self,
        *,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        system_prefix: Optional[str],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
        client: Optional[anthropic.AsyncAnthropic] = None,
    ) -> QueryResult:
        client = client or self._async()
        new_msg_history = msg_history + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]
        call_kwargs, extra_headers = self._prepare_call_kwargs(model, kwargs, output_model)
        response = await client.messages.create(
            model=model,
            system=_build_system(system_msg, system_prefix),
            messages=_add_user_cache_marker(new_msg_history),
            extra_headers=extra_headers,
            **call_kwargs,
        )
        return self._build_result(
            response=response, model=model, msg=msg, system_msg=system_msg,
            new_msg_history=new_msg_history, output_model=output_model, kwargs=kwargs,
        )

    def _prepare_call_kwargs(
        self,
        api_model: str,
        kwargs: dict,
        output_model: Optional[Type[BaseModel]],
    ) -> tuple[dict, Optional[dict]]:
        """Translate generic kwargs into Anthropic shape and attach the tool
        payload when output_model is set. tool_choice defaults to `auto`,
        forced only for models in `_FORCE_TOOL_CHOICE_MODELS`.

        Returns (call_kwargs, extra_headers); extra_headers carries the
        interleaved-thinking beta when thinking is enabled.
        """
        out = self.build_call_kwargs(api_model=api_model, requested=kwargs)
        out.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
        if output_model is not None:
            tool = _structured_output_tool(output_model)
            tool["cache_control"] = {"type": "ephemeral"}
            out["tools"] = [tool]
            if output_model.__name__ in _FORCE_TOOL_CHOICE_MODELS:
                out["tool_choice"] = {
                    "type": "tool", "name": output_model.__name__,
                }
                # Anthropic rejects `thinking` + forced tool_choice (`Thinking
                # may not be enabled when tool_choice forces tool use.`). The
                # forced tool_choice is the load-bearing fix for these models
                # (they skip the call under `auto`); thinking is the optional
                # CoT layer. Drop thinking on these calls so the structured
                # output succeeds.
                out.pop("thinking", None)
                out.pop("output_config", None)
            else:
                out["tool_choice"] = {"type": "auto"}
        extra_headers: Optional[dict] = None
        # The interleaved-thinking beta was designed for the legacy
        # `thinking: {type: "enabled"}` shape. Adaptive thinking on Opus 4.7+
        # has its own native interleaving and the beta header is unnecessary
        # — sending both has been observed correlating with the model
        # emitting legacy-XML tool-call patterns (e.g. `<function_calls>
        # <invoke name="write_file">`) as literal text inside its thinking
        # block instead of via the real tool API.
        thinking_block = out.get("thinking")
        if thinking_block and thinking_block.get("type") == "enabled":
            extra_headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
        return out, extra_headers

    def _build_result(
        self,
        *,
        response,
        model: str,
        msg: str,
        system_msg: str,
        new_msg_history: list[dict],
        output_model: Optional[Type[BaseModel]],
        kwargs: dict,
    ) -> QueryResult:
        # Cost is computed before parsing — the call already happened, the
        # tokens are billed, and we want the LLMValidationError path below
        # to carry accurate telemetry even when validation fails.
        cost_results = get_anthropic_costs(response, model)
        raw_output = ""
        if output_model is not None:
            payload = _find_tool_use_payload(response, output_model)
            if payload is None:
                raise LLMValidationError(
                    cause=ValueError(
                        f"Anthropic structured-output response missing tool_use "
                        f"for {output_model.__name__}"
                    ),
                    raw_output="",
                    model_name=model, msg=msg, system_msg=system_msg,
                    kwargs=kwargs, **cost_results,
                )
            raw_output = json.dumps(payload, ensure_ascii=False, indent=2)
            try:
                content = output_model.model_validate(payload)
            except ValidationError as e:
                # Preserve the thinking block if present — useful for diagnosing
                # whether the model reasoned its way into a bad shape.
                thought = ""
                for block in response.content:
                    if block.type == "thinking":
                        thought = block.thinking
                        break
                raise LLMValidationError(
                    cause=e,
                    raw_output=raw_output,
                    model_name=model, msg=msg, system_msg=system_msg,
                    thought=thought, kwargs=kwargs, **cost_results,
                ) from e
            # Pull the thinking block when extended thinking is enabled — the
            # response carries it alongside the tool_use under tool_choice=auto.
            thought = ""
            for block in response.content:
                if block.type == "thinking":
                    thought = block.thinking
                    break
            new_msg_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": str(content)}]}
            )
        else:
            content, thought = _extract_text(response)
            new_msg_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": content}]}
            )
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=new_msg_history,
            model_name=model,
            kwargs=kwargs,
            **cost_results,
            thought=thought,
            raw_output=raw_output,
        )


    @backoff.on_exception(
        backoff.expo, _RETRY_EXCEPTIONS,
        max_tries=MAX_TRIES, max_value=MAX_VALUE, max_time=MAX_TIME,
        on_backoff=_backoff_handler,
    )
    async def query_async_with_tools(
        self,
        *,
        model: str,
        msg: str,
        system_msg: str,
        msg_history: list[dict],
        system_prefix: Optional[str],
        tools: list[dict],
        dispatch: Any,  # async (name, params) -> str
        max_iters: int = 10,
        kwargs: dict,
        client: Optional[anthropic.AsyncAnthropic] = None,
    ) -> QueryResult:
        """Tool-use loop. Sends tool schemas; on tool_use blocks, runs the
        dispatcher for each, appends tool_result blocks, re-calls until the
        model responds without tool_use or `max_iters` is hit.

        Returns a QueryResult whose `content` is the final assistant text,
        whose token/cost fields sum across all loop iterations, and whose
        `new_msg_history` is the full multi-turn transcript including all
        tool_use/tool_result rounds.
        """
        client = client or self._async()
        call_kwargs = self.build_call_kwargs(api_model=model, requested=kwargs)
        call_kwargs.setdefault("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
        # Translate neutral {name, description, parameters} → Anthropic's
        # {name, description, input_schema}. Allows the orchestrator to stay
        # provider-agnostic.
        call_kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema") or t.get("parameters") or {"type": "object"},
            }
            for t in tools
        ]
        # Mark cache_control on the last tool definition so the tools array
        # joins the cached prefix. Adds a 3rd breakpoint (system + tools +
        # latest-user); cache scopes nest, so calls that share system+tools
        # but vary user_msg still hit the tools cache. Anthropic auto-skips
        # below per-model minimum, so small toolsets cost nothing.
        if call_kwargs["tools"]:
            call_kwargs["tools"][-1] = {
                **call_kwargs["tools"][-1],
                "cache_control": {"type": "ephemeral"},
            }
        extra_headers: Optional[dict] = None
        # See `_prepare_call_kwargs` for the rationale: the interleaved-thinking
        # beta is for the legacy `thinking: {type: "enabled"}` shape only;
        # adaptive thinking has native interleaving and adding the beta on
        # top correlates with legacy-XML tool-call leakage.
        thinking_block = call_kwargs.get("thinking")
        if thinking_block and thinking_block.get("type") == "enabled":
            extra_headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}

        history: list[dict] = list(msg_history) + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]

        totals = {
            "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
            "input_cost": 0.0, "output_cost": 0.0, "cost": 0.0,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
        }
        last_text = ""
        last_thought = ""

        for _ in range(max_iters):
            response = await client.messages.create(
                model=model,
                system=_build_system(system_msg, system_prefix),
                messages=_add_user_cache_marker(history),
                extra_headers=extra_headers,
                **call_kwargs,
            )
            costs = get_anthropic_costs(response, model)
            for k in totals:
                totals[k] += costs[k]

            assistant_blocks = [
                _block_to_dict(b) for b in response.content
            ]
            history.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                last_text, last_thought = _extract_text(response)
                break

            tool_results = []
            for use in tool_uses:
                try:
                    result_str = await dispatch(use.name, dict(use.input))
                except Exception as e:
                    logger.warning("tool %s dispatch failed: %s", use.name, e)
                    result_str = f"tool error: {e}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": use.id,
                    "content": result_str,
                })
            history.append({"role": "user", "content": tool_results})
        else:
            logger.warning(
                "tool-use loop hit max_iters=%d without final text response",
                max_iters,
            )
            last_text, last_thought = _extract_text(response)

        return QueryResult(
            content=last_text,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=history,
            model_name=model,
            kwargs=kwargs,
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            thinking_tokens=totals["thinking_tokens"],
            cost=totals["cost"],
            input_cost=totals["input_cost"],
            output_cost=totals["output_cost"],
            cache_read_tokens=totals["cache_read_tokens"],
            cache_creation_tokens=totals["cache_creation_tokens"],
            thought=last_thought,
        )


def _block_to_dict(block: Any) -> dict:
    """Serialize an Anthropic content block back to its API-shape dict so
    it can be replayed in the next call's message history."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": getattr(block, "signature", "")}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input),
        }
    return {"type": block.type}


class BedrockProvider(AnthropicProvider):
    """Bedrock-hosted Claude. Same call shape, different SDK client."""

    name = "bedrock"

    def _make_sync_client(self) -> anthropic.AnthropicBedrock:
        return anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )

    def _make_async_client(self) -> anthropic.AsyncAnthropicBedrock:
        return anthropic.AsyncAnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
            timeout=TIMEOUT,
        )


# Module-level singletons.
ANTHROPIC = AnthropicProvider()
BEDROCK = BedrockProvider()


# Back-compat shim: existing tests import these names from this module.
# Keep them as thin wrappers around the singleton until the test refactor.
def query_anthropic(client, model, msg, system_msg, msg_history, output_model,
                    model_posteriors=None, **kwargs) -> QueryResult:
    """Legacy free-function form. Routes to AnthropicProvider.query, passing
    the caller-supplied client through. Kept for backwards compatibility."""
    system_prefix = kwargs.pop("system_prefix", None)
    result = ANTHROPIC.query(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        # The old shape included model_posteriors; reconstruct the result with it.
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result


async def query_anthropic_async(client, model, msg, system_msg, msg_history, output_model,
                                model_posteriors=None, **kwargs) -> QueryResult:
    system_prefix = kwargs.pop("system_prefix", None)
    result = await ANTHROPIC.query_async(
        model=model, msg=msg, system_msg=system_msg, msg_history=msg_history,
        system_prefix=system_prefix, output_model=output_model, kwargs=kwargs,
        client=client,
    )
    if model_posteriors is not None:
        from dataclasses import replace
        return replace(result, model_posteriors=model_posteriors)
    return result
