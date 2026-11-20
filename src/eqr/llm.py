"""Provider backends. The agent loop and traces use ONE canonical format (Anthropic-style content blocks):

    assistant content: [{"type":"text","text":...}, {"type":"tool_use","id":...,"name":...,"input":{...}}]
    user content:      "text"  or  [{"type":"tool_result","tool_use_id":...,"content":str,"is_error":bool}]

Each backend translates to/from its provider's wire format. Keeping traces canonical means a trace from
Claude or GPT can feed the same graders and the same SFT export.
"""
from __future__ import annotations

import json
from typing import Protocol

# (assistant_blocks, usage) where usage = {"input_tokens": int, "output_tokens": int}
Completion = tuple[list[dict], dict]


class Backend(Protocol):
    provider: str

    def complete(self, *, model: str, system: str, tools: list[dict], messages: list[dict],
                 max_tokens: int) -> Completion: ...


def provider_for(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, *, model, system, tools, messages, max_tokens):
        resp = self.client.messages.create(model=model, max_tokens=max_tokens, system=system, tools=tools,
                                           messages=messages)
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        return blocks, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def to_responses_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "name": t["name"], "description": t["description"],
             "parameters": t["input_schema"]} for t in tools]


def to_responses_input(messages: list[dict]) -> list[dict]:
    """Canonical messages -> Responses API input items (stateless: we replay the whole history)."""
    items: list[dict] = []
    for m in messages:
        c = m["content"]
        if m["role"] == "user":
            if isinstance(c, str):
                items.append({"role": "user", "content": c})
                continue
            for b in c:
                if b["type"] == "tool_result":
                    prefix = "ERROR: " if b.get("is_error") else ""
                    items.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                                  "output": prefix + b["content"]})
                elif b["type"] == "text":
                    items.append({"role": "user", "content": b["text"]})
        else:
            for b in ([{"type": "text", "text": c}] if isinstance(c, str) else c):
                if b["type"] == "text" and b["text"]:
                    items.append({"role": "assistant", "content": b["text"]})
                elif b["type"] == "tool_use":
                    items.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                                  "arguments": json.dumps(b["input"])})
    return items


def from_responses_output(output) -> list[dict]:
    """Responses API output items -> canonical assistant blocks. Reasoning items are not replayed."""
    blocks: list[dict] = []
    for item in output:
        if item.type == "message":
            text = "".join(p.text for p in item.content if p.type == "output_text")
            if text:
                blocks.append({"type": "text", "text": text})
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": item.arguments}  # surfaces as a tool error, not a crash
            blocks.append({"type": "tool_use", "id": item.call_id, "name": item.name, "input": args})
    return blocks


class OpenAIBackend:
    """OpenAI via the Responses API. Newer reasoning models (e.g. gpt-6.2-astra) reject function tools on
    Chat Completions unless reasoning is disabled, so we use /v1/responses for every OpenAI model."""

    provider = "openai"

    def __init__(self):
        import openai

        self.client = openai.OpenAI()

    def complete(self, *, model, system, tools, messages, max_tokens):
        resp = self.client.responses.create(
            model=model,
            instructions=system,
            input=to_responses_input(messages),
            tools=to_responses_tools(tools),
            max_output_tokens=max(max_tokens, 16000),  # hidden reasoning counts against this
            store=False,
        )
        return from_responses_output(resp.output), {
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def get_backend(model: str) -> Backend:
    return AnthropicBackend() if provider_for(model) == "anthropic" else OpenAIBackend()


def complete_json(model: str, system: str, user: str) -> str:
    """One-shot JSON completion (no tools). Used for small extraction jobs."""
    if provider_for(model) == "anthropic":
        import anthropic

        r = anthropic.Anthropic().messages.create(model=model, max_tokens=2000, system=system,
                                                  messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")
    import openai

    r = openai.OpenAI().chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, max_completion_tokens=8000)
    return r.choices[0].message.content or ""
