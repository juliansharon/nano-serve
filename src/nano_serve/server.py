"""Phase 1 server: streaming generation + live metrics + dashboard.

Endpoints:
  POST /generate        — JSON in, JSON out; pass {"stream": true} for SSE streaming
  GET  /metrics.json    — live metrics snapshot (the dashboard polls this)
  GET  /dashboard       — human dashboard: charts, request log, live playground
  GET  /healthz         — liveness + model/device info

Streaming uses SSE (Server-Sent Events): a plain HTTP response that stays open and
sends `data: {...}\n\n` frames as tokens arrive — same protocol OpenAI/Anthropic use.

There is still no queue and no batching. FastAPI handles requests concurrently at
the HTTP layer, but the Engine's lock funnels them through the GPU one at a time;
fire 8 concurrent requests and 7 sit blocked on the lock. The dashboard now makes
that visible: in-flight climbs, throughput doesn't. Phases 2-3 fix it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import anyio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from nano_serve.engine import DEFAULT_MODEL, Engine
from nano_serve.metrics import Metrics

app = FastAPI(title="nano-serve", version="0.1.0")
_engine: Engine | None = None
_metrics = Metrics()

STATIC_DIR = Path(__file__).parent / "static"


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    stream: bool = False


@app.on_event("startup")
def _load() -> None:
    global _engine
    _engine = Engine(DEFAULT_MODEL)
    print(f"[nano-serve] ready: {_engine.info()}")


@app.get("/healthz")
def healthz() -> dict:
    if _engine is None:
        return {"status": "loading"}
    return {"status": "ok", **_engine.info()}


@app.get("/metrics.json")
def metrics_json() -> dict:
    snap = _metrics.snapshot()
    snap["engine"] = _engine.info() if _engine else None
    return snap


@app.get("/dashboard")
def dashboard() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "dashboard.html").read_text(encoding="utf-8"))


def _record(req: GenRequest, done: dict, streamed: bool) -> None:
    _metrics.request_finished({
        "prompt": req.prompt[:80],
        "prompt_tokens": done["prompt_tokens"],
        "output_tokens": done["output_tokens"],
        "ttft_ms": round(done["ttft_ms"], 1),
        "tpot_ms": round(done["tpot_ms"], 2),
        "total_ms": round(done["total_ms"], 1),
        "finish_reason": done["finish_reason"],
        "stream": streamed,
    })


def _sse_events(req: GenRequest) -> Iterator[str]:
    """Blocking generator (runs in a worker thread): engine events -> SSE frames."""
    assert _engine is not None
    _metrics.request_started()
    finished = False
    try:
        for ev in _engine.stream(req.prompt, req.max_tokens, req.temperature, req.top_p):
            if ev["type"] == "token":
                _metrics.token_generated()
                yield f'data: {json.dumps({"text": ev["text"]})}\n\n'
            else:
                _record(req, ev, streamed=True)
                finished = True
                done = {k: round(v, 2) if isinstance(v, float) else v
                        for k, v in ev.items() if k != "type"}
                yield f'data: {json.dumps({"done": True, **done})}\n\n'
    finally:
        # Client hung up mid-stream: the StreamingResponse closes this generator,
        # so we still have to balance the in-flight gauge.
        if not finished:
            _metrics.request_aborted()


@app.post("/generate")
async def generate(req: GenRequest):
    assert _engine is not None, "engine not loaded"
    if req.stream:
        return StreamingResponse(
            iterate_in_threadpool(_sse_events(req)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path consumes the same generator, just assembled server-side.
    def _run() -> tuple[str, dict]:
        _metrics.request_started()
        try:
            parts: list[str] = []
            done: dict = {}
            for ev in _engine.stream(req.prompt, req.max_tokens, req.temperature, req.top_p):
                if ev["type"] == "token":
                    _metrics.token_generated()
                    parts.append(ev["text"])
                else:
                    done = ev
            _record(req, done, streamed=False)
            return "".join(parts), done
        except BaseException:
            _metrics.request_aborted()
            raise

    text, done = await anyio.to_thread.run_sync(_run)
    return {
        "text": text,
        "prompt_tokens": done["prompt_tokens"],
        "output_tokens": done["output_tokens"],
        "ttft_ms": round(done["ttft_ms"], 1),
        "tpot_ms": round(done["tpot_ms"], 2),
        "total_ms": round(done["total_ms"], 1),
        "finish_reason": done["finish_reason"],
    }
