"""Phase 0 server: a thin FastAPI wrapper over the naive Engine.

There is no queue and no batching here. FastAPI handles requests concurrently at the
HTTP layer, but the Engine's lock funnels them through the GPU one at a time. So if you
fire 8 concurrent requests, 7 of them are blocked waiting for the lock — throughput
stays flat while latency balloons. Proving that with examples/bench.py is the whole
point of Phase 0.
"""

from __future__ import annotations

import anyio
from fastapi import FastAPI
from pydantic import BaseModel, Field

from nano_serve.engine import DEFAULT_MODEL, Engine

app = FastAPI(title="nano-serve", version="0.0.0")
_engine: Engine | None = None


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)


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


@app.post("/generate")
async def generate(req: GenRequest) -> dict:
    assert _engine is not None, "engine not loaded"
    # The decode loop is blocking + CPU/GPU-bound. Run it in a worker thread so we
    # don't freeze the event loop — the Engine's own lock still serializes the GPU.
    result = await anyio.to_thread.run_sync(
        lambda: _engine.generate(
            req.prompt, req.max_tokens, req.temperature, req.top_p
        )
    )
    return {
        "text": result.text,
        "prompt_tokens": result.prompt_tokens,
        "output_tokens": result.output_tokens,
        "ttft_ms": round(result.ttft_ms, 1),
        "tpot_ms": round(result.tpot_ms, 2),
        "total_ms": round(result.total_ms, 1),
        "finish_reason": result.finish_reason,
    }
