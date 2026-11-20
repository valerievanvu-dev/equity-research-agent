"""The agent entry point. Two engines behind one function:

- "sdk" (default): the OpenAI Agents SDK, with its tracing as the observability layer (see sdk_engine.py).
- "loop": a plain hand-written tool-use loop with no framework (below), kept for Claude models and for comparison.

Both return the same RunResult and write the same canonical trace JSON, so graders, evals and the PDF work on either.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from .llm import get_backend, provider_for
from .prompts import SYSTEM
from .runresult import RunResult
from .schemas import ResearchNote
from .tools import TOOLS, collect_sources, run_tool

DEFAULT_MODEL = os.environ.get("EQR_MODEL", "gpt-6.2-astra")



def run(
    task: str,
    *,
    ticker: str | None = None,
    model: str | None = None,
    max_turns: int = 15,
    trace_dir: str | Path | None = "runs",
    meta: dict | None = None,
    engine: str | None = None,
) -> RunResult:
    model = model or DEFAULT_MODEL
    engine = engine or os.environ.get("EQR_ENGINE", "sdk")
    if engine == "sdk" and provider_for(model) == "openai":  # the SDK path is OpenAI-only; Claude uses the loop
        from .sdk_engine import run_sdk

        return run_sdk(task, ticker=ticker, model=model, max_turns=max_turns, trace_dir=trace_dir, meta=meta)
    backend = get_backend(model)
    messages: list[dict] = [{"role": "user", "content": task}]
    tool_log: list[dict] = []
    sources_seen: set[str] = set()
    usage = {"input_tokens": 0, "output_tokens": 0}
    note: ResearchNote | None = None
    error: str | None = None
    turn = 0
    t0 = time.time()

    try:
        for turn in range(1, max_turns + 1):
            blocks, u = backend.complete(model=model, system=SYSTEM, tools=TOOLS, messages=messages, max_tokens=4096)
            usage["input_tokens"] += u["input_tokens"]
            usage["output_tokens"] += u["output_tokens"]
            messages.append({"role": "assistant", "content": blocks})

            tool_uses = [b for b in blocks if b["type"] == "tool_use"]
            if not tool_uses:
                messages.append({"role": "user", "content": "You have not submitted the note. Call submit_note now."})
                continue

            results = []
            for tu in tool_uses:
                started = time.time()
                is_error = False
                if tu["name"] == "submit_note":
                    try:
                        note = ResearchNote.model_validate(tu["input"])
                        content = "Note accepted."
                    except ValidationError as e:  # let the model self-repair
                        is_error, content = True, f"Invalid note: {e}"
                else:
                    try:
                        result = run_tool(tu["name"], tu["input"])
                        sources_seen |= collect_sources(result)
                        content = json.dumps(result)
                    except Exception as e:  # tool errors are observations, not crashes
                        is_error, content = True, f"{type(e).__name__}: {e}"
                tool_log.append(
                    {
                        "turn": turn,
                        "name": tu["name"],
                        "input": tu["input"],
                        "is_error": is_error,
                        "latency_s": round(time.time() - started, 3),
                        "output_chars": len(content),
                    }
                )
                results.append(
                    {"type": "tool_result", "tool_use_id": tu["id"], "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
            if note:
                break
    except Exception as e:  # API errors etc: record, don't lose the trace
        error = f"{type(e).__name__}: {e}"

    if note is None and error is None:
        error = f"no valid note after {turn} turns"

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{(ticker or 'run')}-{uuid.uuid4().hex[:6]}"
    trace = {
        "run_id": run_id,
        "model": model,
        "provider": provider_for(model),
        "ticker": ticker,
        "task": task,
        "meta": meta or {},
        "system": SYSTEM,
        "turns": turn,
        "usage": usage,
        "wall_s": round(time.time() - t0, 2),
        "tool_calls": tool_log,
        "sources_seen": sorted(sources_seen),
        "note": note.model_dump() if note else None,
        "error": error,
        "messages": messages,
    }
    path = None
    if trace_dir:
        path = Path(trace_dir) / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2))
    return RunResult(note=note, trace=trace, trace_path=path, error=error)
