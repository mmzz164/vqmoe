#!/usr/bin/env python
"""Anthropic Messages API -> OpenAI Chat Completions proxy for the VQ mlx_lm server.

Claude Code speaks the Anthropic Messages API (POST /v1/messages), but the VQ
model is served by vq_serve.py (= mlx_lm.server), which only implements the
OpenAI /v1/chat/completions API. This lightweight shim sits in front of the VQ
server and translates in both directions, including streaming (SSE) and
tool_use / tool_result. Run it under the same venv that has mlx_lm/fastapi.

Usage:
    VQ_UPSTREAM_URL=http://127.0.0.1:8090 \
        python vq_proxy.py --host 127.0.0.1 --port 8003

Env:
    VQ_UPSTREAM_URL     Base URL of the OpenAI-compatible VQ server
                        (default http://127.0.0.1:8090)
    VQ_UPSTREAM_MODEL   model id sent upstream. "default_model" makes
                        mlx_lm.server use its --model regardless of the id
                        Claude Code asks for (default "default_model").
    VQ_PROXY_TIMEOUT    upstream read timeout in seconds; empty/0 = no limit
                        (default: no limit, generations can be long).

Notes:
    * mlx_lm.server rejects non-text content parts in a list, so every message
      is flattened to a plain string; images become a short placeholder.
    * Upstream `reasoning` stream deltas are forwarded as Anthropic `thinking`
      blocks (set VQ_PROXY_THINKING=0 to drop them instead). Thinking blocks in
      *incoming* requests are always dropped (Qwen re-thinks each turn).
    * Non-leading system-role messages (Claude Code injects them mid-conversation)
      are folded into user turns — Qwen chat templates only allow system first.
"""
import argparse
import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _log_upstream_error(status, detail, oai_body):
    """Always land upstream failures in the proxy log — silent errors here surface
    as misleading generic messages in Claude Code ("model may not exist")."""
    try:
        req = json.dumps(oai_body, ensure_ascii=False)
    except Exception:
        req = repr(oai_body)
    roles = [m.get("role") for m in oai_body.get("messages", [])] if isinstance(oai_body, dict) else []
    print(f"[vq_proxy] upstream HTTP {status}: {detail[:800]}\n"
          f"[vq_proxy] roles={roles}\n"
          f"[vq_proxy] request was: {req[:2000]}", file=sys.stderr, flush=True)
    try:
        with open("/tmp/vq_failed_request.json", "w") as f:
            f.write(req)
    except Exception:
        pass

UPSTREAM_URL = os.environ.get("VQ_UPSTREAM_URL", "http://127.0.0.1:8090").rstrip("/")
UPSTREAM_MODEL = os.environ.get("VQ_UPSTREAM_MODEL", "default_model")
_timeout_env = os.environ.get("VQ_PROXY_TIMEOUT", "").strip()
if _timeout_env in ("", "0", "none", "None"):
    _READ_TIMEOUT = None
else:
    _READ_TIMEOUT = float(_timeout_env)

# The single GPU can only do one big prefill at a time without OOMing the Metal
# working set. mlx_lm.server batches concurrent HTTP requests, so if Claude Code
# retries/resends (or fires a background request) while a large prefill is in
# flight, two prefills stack on the GPU. Serialize upstream generations here so
# vq_serve never sees more than VQ_MAX_CONCURRENCY at once. Raise it only on a
# machine with unified memory to spare.
_MAX_CONCURRENCY = max(1, int(os.environ.get("VQ_MAX_CONCURRENCY", "1")))


@asynccontextmanager
async def lifespan(app):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(_READ_TIMEOUT, connect=10.0))
    app.state.sem = asyncio.Semaphore(_MAX_CONCURRENCY)
    try:
        yield
    finally:
        await app.state.client.aclose()


async def _with_sem(gen, sem):
    """Hold `sem` for the lifetime of async generator `gen`, releasing on normal
    completion OR on cancellation (client disconnect). This is what frees the slot
    so a queued retry can proceed instead of stacking a second upstream prefill."""
    await sem.acquire()
    try:
        async for item in gen:
            yield item
    finally:
        sem.release()


app = FastAPI(title="vq_proxy", lifespan=lifespan)


def _new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# request translation: Anthropic Messages -> OpenAI chat.completions
# ---------------------------------------------------------------------------
def _system_text(system):
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


def _blocks_to_text(content):
    """Flatten an Anthropic content block list to a plain string (text only)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "image":
            parts.append("[image omitted]")
        # tool_use / tool_result handled by the caller; ignore other types
    return "".join(parts)


def _tool_result_text(block):
    rc = block.get("content")
    if isinstance(rc, str):
        return rc
    if isinstance(rc, list):
        parts = []
        for b in rc:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("[image omitted]")
        return "".join(parts)
    if rc is None:
        return ""
    return json.dumps(rc, ensure_ascii=False)


def anthropic_messages_to_openai(body):
    messages = []
    sys_text = _system_text(body.get("system"))
    if sys_text:
        messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        text_parts = []
        tool_calls = []
        tool_msgs = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image":
                text_parts.append("[image omitted]")
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or _new_id("call"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                })
            elif btype == "tool_result":
                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": _tool_result_text(block),
                })
            # drop thinking / redacted_thinking / unknown blocks

        text = "".join(text_parts)

        if role == "assistant":
            m = {"role": "assistant"}
            m["content"] = text if text else ("" if not tool_calls else None)
            if tool_calls:
                m["tool_calls"] = tool_calls
            messages.append(m)
        else:  # user (or anything else): tool results first, then the user text
            messages.extend(tool_msgs)
            if text:
                messages.append({"role": "user", "content": text})
            elif not tool_msgs:
                messages.append({"role": "user", "content": ""})

    # Qwen chat templates raise unless the single system message is at position 0.
    # Claude Code (Agent SDK) injects extra system-role context messages into the
    # conversation — fold any non-leading system message into a user turn.
    for i, m in enumerate(messages):
        if i > 0 and m.get("role") == "system":
            messages[i] = {"role": "user",
                           "content": "<system-reminder>\n%s\n</system-reminder>"
                                      % (m.get("content") or "")}

    return messages


def anthropic_tools_to_openai(tools):
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def anthropic_tool_choice_to_openai(tc):
    if not tc:
        return None
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool" and tc.get("name"):
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def build_openai_request(body, stream):
    oai = {
        "model": UPSTREAM_MODEL,
        "messages": anthropic_messages_to_openai(body),
        "stream": bool(stream),
    }
    if body.get("max_tokens") is not None:
        oai["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        oai["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        oai["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        oai["stop"] = body["stop_sequences"]
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        oai["tools"] = tools
        choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if choice is not None:
            oai["tool_choice"] = choice
    if stream:
        oai["stream_options"] = {"include_usage": True}
    return oai


# ---------------------------------------------------------------------------
# response translation: OpenAI -> Anthropic
# ---------------------------------------------------------------------------
def _map_stop_reason(finish_reason, had_tool_calls):
    if had_tool_calls or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def openai_to_anthropic_response(oai, model_name):
    choice = (oai.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    blocks = []
    text = message.get("content")
    if text:
        blocks.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or _new_id("toolu"),
            "name": fn.get("name", ""),
            "input": args,
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    usage = oai.get("usage") or {}
    return {
        "id": oai.get("id") or _new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": blocks,
        "stop_reason": _map_stop_reason(choice.get("finish_reason"), bool(tool_calls)),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) or 0,
            "output_tokens": usage.get("completion_tokens", 0) or 0,
        },
    }


FORWARD_THINKING = os.environ.get("VQ_PROXY_THINKING", "1") != "0"


async def anthropic_stream(oai_body, model_name, est_input, request=None):
    """Consume the upstream OpenAI SSE stream, yield Anthropic SSE events."""
    msg_id = _new_id("msg")
    open_kind = None      # "think" | "text" | "tool" | None
    open_index = None
    next_index = 0
    text_index = None
    think_index = None
    tool_map = {}         # openai tool_call index -> anthropic block index
    output_tokens = 0
    finish = None

    def _close_block():
        """Events to close the currently open block (thinking needs a signature)."""
        evs = []
        if open_kind == "think":
            evs.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": open_index,
                "delta": {"type": "signature_delta", "signature": ""},
            }))
        evs.append(_sse("content_block_stop", {"type": "content_block_stop", "index": open_index}))
        return evs

    client = app.state.client
    async with client.stream("POST", f"{UPSTREAM_URL}/v1/chat/completions", json=oai_body) as resp:
        if resp.status_code != 200:
            detail = (await resp.aread()).decode("utf-8", "replace")
            _log_upstream_error(resp.status_code, detail, oai_body)
            yield _sse("error", {
                "type": "error",
                "error": {"type": "api_error",
                          "message": f"vq upstream HTTP {resp.status_code}: {detail[:500]}"},
            })
            return

        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model_name, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": est_input, "output_tokens": 0},
            },
        })
        yield _sse("ping", {"type": "ping"})

        async for line in resp.aiter_lines():
            # During a long prefill this generator yields nothing (keepalive lines
            # below are skipped), so Starlette can't detect a client disconnect on
            # a send. Poll it explicitly here — mlx_lm emits keepalive lines every
            # chunk, so this loop keeps ticking — and bail out on disconnect so the
            # async-with closes the upstream connection and vq_serve aborts the run.
            if request is not None:
                try:
                    if await request.is_disconnected():
                        break
                except Exception:
                    pass
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue

            choices = chunk.get("choices") or []
            usage = chunk.get("usage")
            if usage and not choices:
                output_tokens = usage.get("completion_tokens", output_tokens) or output_tokens
                continue
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning and FORWARD_THINKING:
                if open_kind != "think":
                    if open_kind is not None:
                        for ev in _close_block():
                            yield ev
                    if think_index is None:
                        think_index = next_index
                        next_index += 1
                    open_kind, open_index = "think", think_index
                    yield _sse("content_block_start", {
                        "type": "content_block_start", "index": open_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": open_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                })

            content = delta.get("content")
            if content:
                if open_kind != "text":
                    if open_kind is not None:
                        for ev in _close_block():
                            yield ev
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                    open_kind, open_index = "text", text_index
                    yield _sse("content_block_start", {
                        "type": "content_block_start", "index": open_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": open_index,
                    "delta": {"type": "text_delta", "text": content},
                })

            for tc in delta.get("tool_calls") or []:
                oai_idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                if oai_idx not in tool_map:
                    if open_kind is not None:
                        for ev in _close_block():
                            yield ev
                    aidx = next_index
                    next_index += 1
                    tool_map[oai_idx] = aidx
                    open_kind, open_index = "tool", aidx
                    yield _sse("content_block_start", {
                        "type": "content_block_start", "index": aidx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or _new_id("toolu"),
                            "name": fn.get("name") or "",
                            "input": {},
                        },
                    })
                else:
                    aidx = tool_map[oai_idx]
                    if open_kind != "tool" or open_index != aidx:
                        if open_kind is not None:
                            for ev in _close_block():
                                yield ev
                        open_kind, open_index = "tool", aidx
                args = fn.get("arguments")
                if args:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": aidx,
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    })

            fr = choice.get("finish_reason")
            if fr:
                finish = fr

    if open_kind is not None:
        for ev in _close_block():
            yield ev
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _map_stop_reason(finish, bool(tool_map)), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    model_name = body.get("model") or "vq"
    stream = bool(body.get("stream"))
    oai_body = build_openai_request(body, stream)

    if stream:
        est_input = max(1, len(json.dumps(oai_body.get("messages", []), ensure_ascii=False)) // 4)
        return StreamingResponse(
            _with_sem(anthropic_stream(oai_body, model_name, est_input, request), app.state.sem),
            media_type="text/event-stream",
        )

    try:
        async with app.state.sem:
            resp = await app.state.client.post(f"{UPSTREAM_URL}/v1/chat/completions", json=oai_body)
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={
            "type": "error",
            "error": {"type": "api_error", "message": f"vq upstream unreachable: {exc}"},
        })
    if resp.status_code != 200:
        _log_upstream_error(resp.status_code, resp.text, oai_body)
        return JSONResponse(status_code=resp.status_code, content={
            "type": "error",
            "error": {"type": "api_error",
                      "message": f"vq upstream HTTP {resp.status_code}: {resp.text[:500]}"},
        })
    return JSONResponse(content=openai_to_anthropic_response(resp.json(), model_name))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    oai = anthropic_messages_to_openai(body)
    text = _system_text(body.get("system")) + json.dumps(oai, ensure_ascii=False)
    return JSONResponse(content={"input_tokens": max(1, len(text) // 4)})


@app.get("/v1/models")
async def models():
    try:
        resp = await app.state.client.get(f"{UPSTREAM_URL}/v1/models")
        if resp.status_code == 200:
            return JSONResponse(content=resp.json())
    except httpx.HTTPError:
        pass
    return JSONResponse(content={"object": "list", "data": []})


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_URL}


def main():
    parser = argparse.ArgumentParser(description="Anthropic->OpenAI proxy for the VQ server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
