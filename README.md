# nano-serve

A mini LLM inference server you can actually read — built from scratch to learn the
systems behind vLLM/TGI: continuous batching, KV-cache management, scheduling,
streaming, and observability.

The goal isn't to beat vLLM. It's to **build the core ideas by hand** and *measure*
every improvement, so the four numbers that define serving become second nature:

```
TTFT        Time To First Token     — how fast a reply starts
TPOT        Time Per Output Token   — how fast it streams
Throughput  total output tokens/sec — across ALL concurrent users
p99         the slow-tail latency   — worst-case user experience
```

## Roadmap

| Phase | Builds | Concept |
|-------|--------|---------|
| **0** | Naive server: 1 request at a time + benchmark | baseline TTFT/TPOT/throughput |
| 1 | Token streaming (SSE) | async streaming generation |
| 2 | Request queue + background generation loop | decouple HTTP from the GPU |
| 3 | ⭐ Continuous (dynamic) batching | the marquee feature — 5–10× throughput |
| 4 | KV-cache management (paged-attention ideas) | GPU memory as the real bottleneck |
| 5 | Multi-model routing + auth + rate limiting | a real gateway |
| 6 | Observability (Prometheus + Grafana) | prove performance, don't claim it |
| 7 | Scale-out: replicas + LB + Docker/k8s + CI | distributed & deployable |

## Status: Phase 0 — naive baseline

One request at a time, no batching. This is the number every later phase has to beat.

### Hardware
RTX 3050 Laptop (4 GB). Model: **Qwen2.5-0.5B-Instruct** (fp16, ~1 GB) — small enough
to iterate fast; the *techniques* are identical at 70B.

## Quickstart

```powershell
# 1. install deps (torch is installed separately with CUDA — see below)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# torch with CUDA 12.4 (one-time):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. run the server (downloads the model on first start)
.\.venv\Scripts\python.exe -m uvicorn nano_serve.server:app --app-dir src --port 8000

# 3. in another shell, benchmark it
.\.venv\Scripts\python.exe examples/bench.py --concurrency 1
.\.venv\Scripts\python.exe examples/bench.py --concurrency 8   # watch throughput NOT scale (yet)
```

## API

`POST /generate`
```json
{ "prompt": "Explain KV-cache in one sentence.", "max_tokens": 64, "temperature": 0.7 }
```
returns
```json
{ "text": "...", "prompt_tokens": 12, "output_tokens": 64,
  "ttft_ms": 41.2, "tpot_ms": 8.7, "total_ms": 597.1 }
```

`GET /healthz` — liveness + model/device info.
