"""Phase 3 server: same API, continuous batching underneath.

Endpoints:
  POST /generate        — JSON in, JSON out; pass {"stream": true} for SSE streaming
  GET  /metrics.json    — live metrics snapshot (the dashboard polls this)
  GET  /dashboard       — human dashboard: charts, request log, live playground
  GET  /healthz         — liveness + model/device/scheduler info

The HTTP layer no longer touches the model. A handler submits a Job to the
Scheduler's queue and consumes token events from the Job's own queue; the
scheduler thread owns the GPU and batches every active request per decode step.
Fire 8 concurrent requests now and they share the GPU instead of queueing behind
a lock — watch the dashboard's batch-size gauge fill up and throughput climb.
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
from nano_serve.scheduler import Job, Scheduler

app = FastAPI(title="nano-serve", version="0.3.0")
_engine: Engine | None = None
_scheduler: Scheduler | None = None
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
    global _engine, _scheduler
    _engine = Engine(DEFAULT_MODEL)
    _scheduler = Scheduler(_engine, metrics=_metrics)
    _scheduler.start()
    print(f"[nano-serve] ready: {_engine.info()} scheduler={_scheduler.info()}")


@app.get("/healthz")
def healthz() -> dict:
    if _engine is None or _scheduler is None:
        return {"status": "loading"}
    return {"status": "ok", **_engine.info(), "scheduler": _scheduler.info()}


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


def _job_events(req: GenRequest) -> Iterator[dict]:
    """Blocking: submit to the scheduler, relay events until done.

    If the consumer disappears (client hung up), the finally-block flags the job
    cancelled and the scheduler evicts it from the batch at the next step — a
    freed slot, not a zombie sequence burning GPU.
    """
    assert _scheduler is not None
    job: Job = _scheduler.submit(req.prompt, req.max_tokens, req.temperature, req.top_p)
    try:
        while True:
            ev = job.events.get()
            yield ev
            if ev["type"] == "done":
                break
    finally:
        job.cancelled = True  # no-op if already finished


def _sse_events(req: GenRequest) -> Iterator[str]:
    _metrics.request_started()
    finished = False
    try:
        for ev in _job_events(req):
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
        if not finished:
            _metrics.request_aborted()


@app.post("/generate")
async def generate(req: GenRequest):
    assert _scheduler is not None, "scheduler not ready"
    if req.stream:
        return StreamingResponse(
            iterate_in_threadpool(_sse_events(req)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _run() -> tuple[str, dict]:
        _metrics.request_started()
        try:
            parts: list[str] = []
            done: dict = {}
            for ev in _job_events(req):
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
