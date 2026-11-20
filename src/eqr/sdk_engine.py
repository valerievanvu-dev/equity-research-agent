"""Run the analyst on the OpenAI Agents SDK, with the SDK's tracing as the observability layer.

What the SDK gives us, and what we keep:
- The SDK runs the loop, calls our tools, and records a trace (LLM calls, tool calls with inputs/outputs, timings,
  token usage) to the OpenAI Traces dashboard. Each run gets a trace id and link, stored in the local run file.
- Our tools, JSON schemas and Pydantic note schema are unchanged. We hand the SDK our hand-written schemas through
  FunctionTool, so there is exactly one definition of each tool for both engines.
- A custom finish rule ends the run only when `submit_note` was ACCEPTED. An invalid note goes back to the model as an
  error to fix, as in the hand-written loop.
- After the run we convert the SDK's items into the same canonical trace JSON the loop writes, so graders, the judge,
  the SFT export and the PDF builder work unchanged.

Privacy note: tracing uploads prompts, tool inputs and tool outputs (SEC data, web excerpts, the note) to the OpenAI
platform under your API key's organisation. Set EQR_TRACING=off to disable it.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agents import (Agent, FunctionTool, MaxTurnsExceeded, RunConfig, Runner, ToolsToFinalOutputResult, trace)
from agents.tracing import get_trace_provider
from pydantic import ValidationError

from .prompts import SYSTEM
from .runresult import RunResult
from .schemas import ResearchNote
from .tools import TOOLS, collect_sources, run_tool

TRACE_URL = "https://platform.openai.com/traces/trace?trace_id={}"
ERROR_PREFIX = "ERROR: "


@dataclass
class RunState:
    """Mutable per-run state shared by the tool wrappers (the SDK calls tools as independent async functions)."""
    note: ResearchNote | None = None
    sources_seen: set[str] = field(default_factory=set)
    tool_log: list[dict] = field(default_factory=list)


def make_tools(state: RunState) -> list[FunctionTool]:
    """Wrap each of our tools (same names and JSON schemas as the loop engine) as an SDK FunctionTool."""

    def build(spec: dict) -> FunctionTool:
        name = spec["name"]

        async def invoke(ctx, input_str: str) -> str:
            started, is_error, args = time.time(), False, {}
            try:
                args = json.loads(input_str or "{}")
                if name == "submit_note":
                    try:
                        state.note = ResearchNote.model_validate(args)
                        content = "Note accepted."
                    except ValidationError as e:  # the model sees this and can resubmit
                        is_error, content = True, f"Invalid note: {e}"
                else:
                    result = await asyncio.to_thread(run_tool, name, args)  # blocking network IO off the event loop
                    state.sources_seen |= collect_sources(result)
                    content = json.dumps(result)
            except Exception as e:  # tool errors are observations for the model, never crashes
                is_error, content = True, f"{type(e).__name__}: {e}"
            state.tool_log.append({"turn": None, "name": name, "input": args, "is_error": is_error,
                                   "latency_s": round(time.time() - started, 3), "output_chars": len(content)})
            return (ERROR_PREFIX + content) if is_error else content

        return FunctionTool(name=name, description=spec["description"], params_json_schema=spec["input_schema"],
                            on_invoke_tool=invoke, strict_json_schema=False, timeout_seconds=300)

    return [build(t) for t in TOOLS]


def finish_when_note_accepted(state: RunState):
    """tool_use_behavior: stop only once submit_note succeeded; otherwise let the model run again (and repair)."""
    def behavior(_ctx, _tool_results) -> ToolsToFinalOutputResult:
        if state.note is not None:
            return ToolsToFinalOutputResult(is_final_output=True, final_output="note submitted")
        return ToolsToFinalOutputResult(is_final_output=False)
    return behavior


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def items_to_messages(task: str, items) -> list[dict]:
    """SDK run items -> the canonical Anthropic-style messages the rest of the project reads."""
    msgs: list[dict] = [{"role": "user", "content": task}]
    asst: list[dict] = []
    results: list[dict] = []

    def flush_asst():
        if asst:
            msgs.append({"role": "assistant", "content": list(asst)})
            asst.clear()

    def flush_results():
        if results:
            msgs.append({"role": "user", "content": list(results)})
            results.clear()

    for it in items:
        kind, raw = getattr(it, "type", ""), getattr(it, "raw_item", None)
        if kind == "message_output_item":
            flush_results()
            text = "".join(_get(p, "text", "") for p in (_get(raw, "content", []) or []) if _get(p, "type") == "output_text")
            if text:
                asst.append({"type": "text", "text": text})
        elif kind == "tool_call_item":
            flush_results()
            try:
                args = json.loads(_get(raw, "arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": _get(raw, "arguments")}
            asst.append({"type": "tool_use", "id": _get(raw, "call_id"), "name": _get(raw, "name"), "input": args})
        elif kind == "tool_call_output_item":
            flush_asst()
            out = _get(raw, "output", "")
            out = out if isinstance(out, str) else json.dumps(out)
            results.append({"type": "tool_result", "tool_use_id": _get(raw, "call_id"), "content": out,
                            "is_error": out.startswith(ERROR_PREFIX)})
    flush_asst()
    flush_results()
    return msgs


def run_sdk(task: str, *, ticker: str | None, model: str, max_turns: int = 15, trace_dir: str | Path | None = "runs",
            meta: dict | None = None) -> RunResult:
    state = RunState()
    agent = Agent(name="equity-research-analyst", instructions=SYSTEM, model=model, tools=make_tools(state),
                  tool_use_behavior=finish_when_note_accepted(state))
    tracing_on = os.environ.get("EQR_TRACING", "on").lower() != "off"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{(ticker or 'run')}-{uuid.uuid4().hex[:6]}"
    t0, result, error, trace_id, run_data = time.time(), None, None, None, None

    try:
        with trace(workflow_name=f"equity-research:{ticker or 'run'}", group_id=ticker or run_id,
                   metadata={"run_id": run_id, "ticker": str(ticker), "model": model, "engine": "sdk",
                             **{str(k): str(v) for k, v in (meta or {}).items()}}, disabled=not tracing_on) as tr:
            trace_id = tr.trace_id if tracing_on else None
            result = Runner.run_sync(agent, task, max_turns=max_turns,
                                     run_config=RunConfig(tracing_disabled=not tracing_on, trace_include_sensitive_data=True))
    except MaxTurnsExceeded as e:
        error, run_data = f"max turns ({max_turns}) exceeded without an accepted note", getattr(e, "run_data", None)
    except Exception as e:  # API/SDK errors: keep what we have
        error, run_data = f"{type(e).__name__}: {e}", getattr(e, "run_data", None)
    finally:
        if tracing_on:
            try:
                get_trace_provider().force_flush()  # short-lived CLI processes must flush before exiting
            except Exception:
                pass

    src = result if result is not None else run_data
    items = list(getattr(src, "new_items", []) or [])
    usage = getattr(getattr(src, "context_wrapper", None), "usage", None)
    in_tok, out_tok = int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
    if state.note is None and error is None:
        error = "run ended without an accepted note"

    trace_dict = {
        "run_id": run_id, "model": model, "provider": "openai", "engine": "sdk", "ticker": ticker, "task": task,
        "meta": meta or {}, "system": SYSTEM, "turns": int(getattr(usage, "requests", 0) or 0),
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok}, "wall_s": round(time.time() - t0, 2),
        "tool_calls": state.tool_log, "sources_seen": sorted(state.sources_seen),
        "note": state.note.model_dump() if state.note else None, "error": error,
        "trace_id": trace_id, "trace_url": TRACE_URL.format(trace_id) if trace_id else None,
        "messages": items_to_messages(task, items),
    }
    path = None
    if trace_dir:
        path = Path(trace_dir) / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace_dict, indent=2))
    return RunResult(note=state.note, trace=trace_dict, trace_path=path, error=error)
