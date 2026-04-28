"""Tool-use loop — mocked Anthropic client returning tool_use → tool_result → text.

Validates: dispatch is called per tool_use block; results are appended as
tool_result content blocks; loop terminates when the model returns text
without tool_use; cost sums across iterations; max_iters bounds the loop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from owtn.llm.providers.anthropic import AnthropicProvider


def _block(type_, **fields):
    b = MagicMock()
    b.type = type_
    for k, v in fields.items():
        setattr(b, k, v)
    return b


def _response(content_blocks, *, input_tokens=10, output_tokens=20):
    r = MagicMock()
    r.content = content_blocks
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    r.usage.cache_read_input_tokens = 0
    r.usage.cache_creation_input_tokens = 0
    return r


@pytest.mark.asyncio
async def test_loop_dispatches_tool_then_returns_final_text():
    """Two-iteration loop: round 1 tool_use → dispatch → round 2 final text."""
    tool_block = _block("tool_use", id="toolu_1", name="echo", input={"x": "hi"})
    text_block = _block("text", text="done")
    final_text_block = _block("text", text="final answer")

    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=[
        _response([tool_block, text_block], input_tokens=100, output_tokens=50),
        _response([final_text_block], input_tokens=120, output_tokens=10),
    ])

    dispatched: list[tuple[str, dict]] = []

    async def dispatch(name, params):
        dispatched.append((name, dict(params)))
        return f"result-of-{name}"

    result = await AnthropicProvider().query_async_with_tools(
        model="claude-sonnet-4-6", msg="please", system_msg="sys",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "echo", "parameters": {"type": "object"}}],
        dispatch=dispatch, max_iters=5, kwargs={"max_tokens": 1024},
        client=client,
    )

    assert dispatched == [("echo", {"x": "hi"})]
    assert result.content == "final answer"
    assert result.input_tokens == 220
    assert result.output_tokens == 60
    assert client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_loop_terminates_immediately_when_no_tool_use():
    """Model returns text without using a tool — single iteration, no dispatch."""
    text_block = _block("text", text="straight answer")
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_response([text_block]))

    dispatch = AsyncMock()
    result = await AnthropicProvider().query_async_with_tools(
        model="claude-sonnet-4-6", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "x", "parameters": {"type": "object"}}],
        dispatch=dispatch, max_iters=5, kwargs={"max_tokens": 100},
        client=client,
    )

    assert result.content == "straight answer"
    dispatch.assert_not_awaited()
    assert client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_loop_dispatches_multiple_tools_per_round():
    """One assistant turn with two tool_use blocks → both dispatched in a single user turn."""
    use_a = _block("tool_use", id="t_a", name="alpha", input={})
    use_b = _block("tool_use", id="t_b", name="beta", input={"k": 1})
    final = _block("text", text="ok")

    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=[
        _response([use_a, use_b]),
        _response([final]),
    ])

    dispatched = []

    async def dispatch(name, params):
        dispatched.append(name)
        return f"r-{name}"

    await AnthropicProvider().query_async_with_tools(
        model="claude-sonnet-4-6", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[
            {"name": "alpha", "description": "a", "parameters": {"type": "object"}},
            {"name": "beta", "description": "b", "parameters": {"type": "object"}},
        ],
        dispatch=dispatch, max_iters=5, kwargs={"max_tokens": 100},
        client=client,
    )

    assert dispatched == ["alpha", "beta"]
    # Inspect the second call's messages — should contain both tool_results.
    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    last_user_msg = second_call_kwargs["messages"][-1]
    assert last_user_msg["role"] == "user"
    assert len(last_user_msg["content"]) == 2
    assert {b["tool_use_id"] for b in last_user_msg["content"]} == {"t_a", "t_b"}


@pytest.mark.asyncio
async def test_loop_respects_max_iters():
    """Looping forever on tool_use is bounded by max_iters."""
    tool_block = _block("tool_use", id="t1", name="echo", input={})
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_response([tool_block]))

    async def dispatch(name, params):
        return "still going"

    await AnthropicProvider().query_async_with_tools(
        model="claude-sonnet-4-6", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "x", "parameters": {"type": "object"}}],
        dispatch=dispatch, max_iters=3, kwargs={"max_tokens": 100},
        client=client,
    )

    assert client.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_dispatch_exception_yields_tool_error_message():
    """Handler raises → loop sends 'tool error: ...' as the result and continues."""
    tool_block = _block("tool_use", id="t1", name="boom", input={})
    final = _block("text", text="recovered")

    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=[
        _response([tool_block]),
        _response([final]),
    ])

    async def dispatch(name, params):
        raise ValueError("nope")

    await AnthropicProvider().query_async_with_tools(
        model="claude-sonnet-4-6", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[{"name": "boom", "description": "x", "parameters": {"type": "object"}}],
        dispatch=dispatch, max_iters=3, kwargs={"max_tokens": 100},
        client=client,
    )

    second_call_kwargs = client.messages.create.call_args_list[1].kwargs
    tool_result = second_call_kwargs["messages"][-1]["content"][0]
    assert "tool error" in tool_result["content"]
    assert "nope" in tool_result["content"]


@pytest.mark.asyncio
async def test_public_dispatcher_rejects_unsupported_provider():
    """`query_async_with_tools` raises NotImplementedError for providers
    without a tool-use implementation."""
    from owtn.llm.tool_use import query_async_with_tools

    async def dispatch(name, params):
        return ""

    with pytest.raises(NotImplementedError, match="provider"):
        # gemini intentionally not in _PROVIDERS_WITH_TOOL_USE
        await query_async_with_tools(
            model_name="gemini-2.5-pro", msg="m", system_msg="s",
            tools=[], dispatch=dispatch,
        )


# ─── DeepSeek tool-use ───────────────────────────────────────────────────


def _ds_tool_call(call_id, name, arguments_json):
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments_json
    tc = MagicMock()
    tc.id = call_id
    tc.function = fn
    return tc


def _ds_response(*, content=None, tool_calls=None, prompt_tokens=10, completion_tokens=20):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = ""
    choice = MagicMock()
    choice.message = msg
    r = MagicMock()
    r.choices = [choice]
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.completion_tokens_details = MagicMock(reasoning_tokens=0)
    r.usage.prompt_tokens_details = MagicMock(cached_tokens=0)
    r.usage.prompt_cache_hit_tokens = 0
    return r


@pytest.mark.asyncio
async def test_deepseek_loop_dispatches_then_returns_text():
    from owtn.llm.providers.deepseek import DeepSeekProvider

    tc = _ds_tool_call("call_1", "echo", '{"x": "hi"}')
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        _ds_response(tool_calls=[tc], prompt_tokens=100, completion_tokens=20),
        _ds_response(content="final answer", prompt_tokens=110, completion_tokens=10),
    ])

    dispatched: list[tuple[str, dict]] = []

    async def dispatch(name, params):
        dispatched.append((name, dict(params)))
        return f"r-{name}"

    result = await DeepSeekProvider().query_async_with_tools(
        model="deepseek-chat", msg="please", system_msg="sys",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "echo it",
                "parameters": {"type": "object",
                               "properties": {"x": {"type": "string"}}}}],
        dispatch=dispatch, max_iters=5, kwargs={"max_tokens": 1024},
        client=client,
    )

    assert dispatched == [("echo", {"x": "hi"})]
    assert result.content == "final answer"
    assert result.input_tokens == 210
    assert client.chat.completions.create.call_count == 2

    # Inspect second call's messages — must include assistant w/ tool_calls
    # and tool result.
    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assistant_msg = second_call_messages[-2]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    tool_msg = second_call_messages[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "r-echo"


@pytest.mark.asyncio
async def test_deepseek_translates_neutral_schema_to_function_wrapper():
    from owtn.llm.providers.deepseek import DeepSeekProvider

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_ds_response(content="ok"))

    async def dispatch(name, params):
        return ""

    await DeepSeekProvider().query_async_with_tools(
        model="deepseek-chat", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "echo",
                "parameters": {"type": "object", "properties": {}}}],
        dispatch=dispatch, max_iters=3, kwargs={},
        client=client,
    )
    sent_tools = client.chat.completions.create.call_args.kwargs["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["function"]["name"] == "echo"
    assert sent_tools[0]["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_deepseek_handles_unparseable_arguments():
    """Malformed JSON in tool_call.arguments → empty params dict, dispatch still runs."""
    from owtn.llm.providers.deepseek import DeepSeekProvider

    bad_tc = _ds_tool_call("c1", "echo", "{not valid json")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        _ds_response(tool_calls=[bad_tc]),
        _ds_response(content="recovered"),
    ])

    received_params: list[dict] = []

    async def dispatch(name, params):
        received_params.append(dict(params))
        return "r"

    result = await DeepSeekProvider().query_async_with_tools(
        model="deepseek-chat", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "x", "parameters": {"type": "object"}}],
        dispatch=dispatch, max_iters=3, kwargs={},
        client=client,
    )
    assert received_params == [{}]
    assert result.content == "recovered"


@pytest.mark.asyncio
async def test_deepseek_replays_reasoning_content_with_tool_calls():
    """DeepSeek reasoning models (deepseek-v4-pro) require `reasoning_content`
    to be passed back in the assistant message that contains tool_calls —
    omitting it triggers a 400 'The reasoning_content in the thinking mode
    must be passed back to the API.' Regression test for the live-API bug
    surfaced 2026-04-28."""
    from owtn.llm.providers.deepseek import DeepSeekProvider

    tc = _ds_tool_call("call_1", "echo", '{"x": "hi"}')
    response_with_reasoning = _ds_response(tool_calls=[tc])
    response_with_reasoning.choices[0].message.reasoning_content = "thinking about this"
    final_response = _ds_response(content="ok")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        response_with_reasoning,
        final_response,
    ])

    async def dispatch(name, params):
        return "result"

    await DeepSeekProvider().query_async_with_tools(
        model="deepseek-v4-pro", msg="m", system_msg="s",
        msg_history=[], system_prefix=None,
        tools=[{"name": "echo", "description": "x", "parameters": {"type": "object"}}],
        dispatch=dispatch, max_iters=3, kwargs={},
        client=client,
    )

    second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assistant_msg = next(m for m in second_call_messages if m["role"] == "assistant")
    assert assistant_msg.get("reasoning_content") == "thinking about this", (
        "reasoning_content must be replayed in the assistant message that "
        "contains tool_calls or DeepSeek's thinking-mode API rejects the call"
    )


@pytest.mark.asyncio
async def test_deepseek_dispatcher_via_public_entry():
    """End-to-end through `owtn.llm.tool_use.query_async_with_tools` —
    deepseek model name resolves to the deepseek provider and runs the loop.
    """
    from owtn.llm.providers.deepseek import DEEPSEEK
    from owtn.llm.tool_use import query_async_with_tools

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_ds_response(content="hello")
    )
    DEEPSEEK._async_client = client  # inject mock client

    async def dispatch(name, params):
        return ""

    try:
        result = await query_async_with_tools(
            model_name="deepseek-chat", msg="m", system_msg="s",
            tools=[{"name": "echo", "description": "x", "parameters": {"type": "object"}}],
            dispatch=dispatch,
        )
        assert result.content == "hello"
    finally:
        DEEPSEEK._async_client = None  # reset
