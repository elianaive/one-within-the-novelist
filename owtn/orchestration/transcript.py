"""Cross-stage transcript helpers.

Tool-use loops collapse to one composite `QueryResult` in the LLM call
log, which hides the back-and-forth. The helpers here render an agent's
full LLM chain (system + user + assistant + tool results) as a Markdown
chat transcript — readable like a chatbot history. Stage 3 and Stage 4
both use them.

Conventional path: `<session_log_dir>/agents/<agent_id>/<label>.transcript.md`.
The `write_transcript` helper resolves the path via `session_log_path`,
which no-ops when no session log dir is set (tests, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

from owtn.orchestration.session import session_log_path


logger = logging.getLogger(__name__)


def summarize_tool_calls(msg_history: list[dict]) -> list[dict]:
    """Walk a tool-use loop's message history and surface what tools fired.

    Returns one entry per tool call in order: `{tool, args_keys}`. Args
    are not included verbatim (large prose passages would balloon the
    trace); only the parameter names so a reader can see at a glance
    what the agent invoked.
    """
    out: list[dict] = []
    for msg in msg_history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "?")
            args_raw = fn.get("arguments", "")
            try:
                args = json.loads(args_raw) if args_raw else {}
                args_keys = sorted(args.keys()) if isinstance(args, dict) else []
            except Exception:
                args_keys = ["<unparsed>"]
            out.append({"tool": name, "args_keys": args_keys})
    return out


# Tools whose call+result pair is pure cognitive scaffolding (the agent
# logging a thought to itself). Their tool_use blocks render as a
# blockquote of the prose argument; the boilerplate tool_result is hidden.
_THINKING_TOOLS = {"think", "note_to_self"}

_RESULT_CHAR_CAP = 32000


def _iter_content_blocks(msg: dict) -> list[dict]:
    """Normalize a message's content into a list of block dicts.

    Handles three shapes:
      - Anthropic: `content` is a list of {type, ...} blocks.
      - OpenAI tool calls: `content` is a (possibly empty) string + a
        sibling `tool_calls` array; we synthesize tool_use blocks.
      - Plain string `content` from any provider: one synthetic text block.
    """
    raw = msg.get("content")
    blocks: list[dict] = []
    if isinstance(raw, list):
        blocks.extend(b for b in raw if isinstance(b, dict))
    elif isinstance(raw, str) and raw.strip():
        blocks.append({"type": "text", "text": raw})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args_raw = fn.get("arguments", "")
        try:
            args = json.loads(args_raw) if args_raw else {}
        except Exception:
            args = {"_raw_arguments": args_raw}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id"),
            "name": fn.get("name", "?"),
            "input": args if isinstance(args, dict) else {"value": args},
        })
    if msg.get("role") == "tool":
        # OpenAI tool-result message: synthesize the tool_result block.
        blocks.append({
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id"),
            "content": msg.get("content"),
        })
    return blocks


def _stringify_result_content(content) -> str:
    """Tool-result content can be a string or a list of {type:text, text:..} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                out.append(str(b["text"]))
            else:
                out.append(str(b))
        return "\n".join(out)
    return "" if content is None else str(content)


def _render_blockquote(parts: list[str], label: str, body: str) -> None:
    parts.append(f"> **{label}:**")
    parts.append(">")
    for ln in body.split("\n"):
        parts.append(f"> {ln}" if ln else ">")
    parts.append("")


def _is_multiline_string(val: object) -> bool:
    return isinstance(val, str) and "\n" in val


def _render_tool_use(parts: list[str], block: dict, hidden_ids: set[str]) -> None:
    name = block.get("name", "?")
    inp = block.get("input") or {}
    if name in _THINKING_TOOLS and isinstance(inp, dict):
        prose = inp.get("thought") or inp.get("text") or ""
        label = "thinking" if name == "think" else "note to self"
        if (uid := block.get("id")):
            hidden_ids.add(uid)
        _render_blockquote(parts, label, str(prose).strip())
        return
    parts.append(f"### → tool call: `{name}`")
    parts.append("")
    # If any argument is a multi-line string (write_file content, edit_file
    # patches, etc.), render fields field-by-field so newlines display as
    # newlines. JSON's `\n` escapes turn long prose into a single illegible
    # line — the readability cost dominates the structural neatness.
    if isinstance(inp, dict) and any(_is_multiline_string(v) for v in inp.values()):
        for key, val in inp.items():
            if _is_multiline_string(val):
                parts.append(f"**{key}:**")
                parts.append("")
                parts.append("```")
                parts.append(str(val).rstrip())
                parts.append("```")
                parts.append("")
            else:
                try:
                    rendered = json.dumps(val, ensure_ascii=False)
                except Exception:
                    rendered = repr(val)
                parts.append(f"**{key}:** `{rendered}`")
                parts.append("")
        return
    try:
        args_pretty = json.dumps(inp, indent=2, ensure_ascii=False)
    except Exception:
        args_pretty = str(inp)
    parts.append("```json")
    parts.append(args_pretty)
    parts.append("```")
    parts.append("")


def _render_tool_result(parts: list[str], block: dict, hidden_ids: set[str]) -> None:
    tcid = block.get("tool_use_id") or "?"
    if tcid in hidden_ids:
        return
    body = _stringify_result_content(block.get("content"))
    parts.append(f"### ← tool result (`{tcid}`)")
    parts.append("")
    parts.append("```")
    parts.append(body.strip()[:_RESULT_CHAR_CAP])
    if len(body) > _RESULT_CHAR_CAP:
        parts.append(f"... [{len(body) - _RESULT_CHAR_CAP} more chars truncated]")
    parts.append("```")
    parts.append("")


def _render_text(parts: list[str], block: dict) -> None:
    text = str(block.get("text") or "").strip()
    if text:
        parts.append(text)
        parts.append("")


def format_transcript(
    *,
    title: str,
    system_msg: str,
    msg_history: list[dict],
    final_output: str | None = None,
) -> str:
    """Render an agent's LLM chain as a Markdown chat transcript.

    Handles Anthropic-style content blocks, OpenAI tool_calls, and plain
    string content uniformly. `think` / `note_to_self` calls collapse to
    a blockquote of the prose argument so the per-call boilerplate
    doesn't drown out the substantive turns.

    `final_output`, if provided, is appended as a separate section —
    useful when a structured-commit response isn't already in
    msg_history (e.g., the explore→commit shape).
    """
    parts: list[str] = [f"# {title}", ""]
    parts.append("## System")
    parts.append("")
    parts.append("```")
    parts.append(system_msg.strip())
    parts.append("```")
    parts.append("")

    hidden_ids: set[str] = set()

    turn = 0
    for msg in msg_history:
        role = msg.get("role")
        blocks = _iter_content_blocks(msg)

        if role == "user":
            # If every block is a tool_result for a hidden think/note_to_self,
            # the whole turn is scaffolding — skip it entirely.
            if blocks and all(
                b.get("type") == "tool_result"
                and b.get("tool_use_id") in hidden_ids
                for b in blocks
            ):
                continue
            turn += 1
            parts.append(f"## Turn {turn} — User")
            parts.append("")
            for b in blocks:
                btype = b.get("type")
                if btype == "tool_result":
                    _render_tool_result(parts, b, hidden_ids)
                elif btype == "text":
                    _render_text(parts, b)
                else:
                    _render_text(parts, {"text": str(b)})

        elif role == "assistant":
            turn += 1
            parts.append(f"## Turn {turn} — Assistant")
            parts.append("")
            reasoning = msg.get("reasoning_content") or ""
            if reasoning and str(reasoning).strip():
                _render_blockquote(parts, "reasoning", str(reasoning).strip())
            for b in blocks:
                btype = b.get("type")
                if btype == "tool_use":
                    _render_tool_use(parts, b, hidden_ids)
                elif btype == "text":
                    _render_text(parts, b)
                elif btype == "thinking":
                    thought = str(b.get("thinking") or "").strip()
                    if thought:
                        _render_blockquote(parts, "extended thinking", thought)
                else:
                    _render_text(parts, {"text": str(b)})

        elif role == "tool":
            # OpenAI tool-result message; synthesized into one tool_result block.
            for b in blocks:
                _render_tool_result(parts, b, hidden_ids)

    if final_output is not None:
        parts.append("---")
        parts.append("")
        parts.append("## Final committed artifact")
        parts.append("")
        if isinstance(final_output, str):
            text = final_output
        else:
            try:
                text = json.dumps(final_output, indent=2, ensure_ascii=False, default=str)
            except Exception:
                text = str(final_output)
        parts.append("```")
        parts.append(text.strip())
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def write_transcript(
    *,
    agent_id: str,
    label: str,
    title: str,
    system_msg: str,
    msg_history: list[dict],
    final_output: object | None = None,
    sub_dirs: Iterable[str] = (),
) -> None:
    """Render `msg_history` as Markdown and write it to
    `<session_log_dir>/agents/<agent_id>/[<sub_dirs>/]<label>.transcript.md`.

    `sub_dirs` lets callers nest under per-cycle / per-iteration
    directories (e.g. `("phase_3_revise", "cycle_0")`). Silently
    no-ops when no session log dir is set.
    """
    parts: list[str] = ["agents", agent_id, *list(sub_dirs), f"{label}.transcript.md"]
    path = session_log_path(*parts)
    if path is None:
        return
    final_text: str | None
    if final_output is None:
        final_text = None
    elif isinstance(final_output, str):
        final_text = final_output
    elif hasattr(final_output, "model_dump_json"):
        final_text = final_output.model_dump_json(indent=2)
    else:
        final_text = str(final_output)
    md = format_transcript(
        title=title,
        system_msg=system_msg,
        msg_history=msg_history,
        final_output=final_text,
    )
    try:
        path.write_text(md, encoding="utf-8")
    except Exception as e:
        logger.warning("failed to write transcript %s: %s", path, e)
