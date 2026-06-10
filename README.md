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
| **1** | Token streaming (SSE) + live dashboard | perceived latency; observability v0 |
| 2 | Request queue + background generation loop | decouple HTTP from the GPU |
| 3 | ⭐ Continuous (dynamic) batching | the marquee feature — 5–10× throughput |
| 4 | KV-cache management (paged-attention ideas) | GPU memory as the real bottleneck |
| 5 | Multi-model routing + auth + rate limiting | a real gateway |
| 6 | Observability (Prometheus + Grafana) | prove performance, don't claim it |
| 7 | Scale-out: replicas + LB + Docker/k8s + CI | distributed & deployable |

## Status: Phase 0 — naive baseline ✅

One request at a time, no batching. This is the number every later phase has to beat.

### Measured baseline (RTX 3050, Qwen2.5-0.5B fp16, 64 max tokens)

| concurrency | throughput | p50 latency | p99 latency |
|------------:|-----------:|------------:|------------:|
| 1 | 13.8 tok/s | 3.3 s | 5.8 s |
| 8 | 21.7 tok/s | **17.2 s** | **21.1 s** |

The lesson of Phase 0 in one table: **8× the load barely moved throughput** (the small
bump is CPU-side overhead pipelining, not GPU parallelism) **while p99 latency grew 3.6×.**
The GPU serves one request at a time; everyone else waits behind the lock. Model loads in
~5 s and uses ~1 GB VRAM, leaving plenty of headroom on the 4 GB card.

## Phase 1 — token streaming (SSE) + live dashboard ✅

The engine's decode loop is now a generator: each token is yielded the moment it's
sampled and pushed to the client as a Server-Sent Event (`{"stream": true}` on
`/generate` — same protocol OpenAI/Anthropic use). Streaming doesn't make generation
faster; it changes *when the user sees the first word*:

| mode (same load, c=1, 64 tok) | throughput | client TTFT p50 | client TTFT p99 |
|---|---:|---:|---:|
| non-stream (JSON) | 28.2 tok/s | **2,186 ms** | 2,313 ms |
| stream (SSE) | 27.6 tok/s | **39 ms** | 45 ms |

Same GPU work, **56× lower perceived latency.** (Throughput here is higher than the
Phase 0 table because the GPU was warm — cold/warm variance is exactly why bench
methodology gets warmup + median-of-3 in a later phase.)

### Live dashboard — `http://127.0.0.1:8000/dashboard`

Observability v0, built in (Prometheus/Grafana replace it in Phase 6): rolling
throughput and GPU-utilization charts, TTFT/latency percentiles over the last 50
requests, in-flight gauge, a request log, and a playground that streams tokens live.
The serving concepts it teaches:

- **counters** (total requests/tokens), **gauges** (in-flight now), **rolling windows**
  (tok/s over 10 s, percentiles over last 50 requests)
- **percentiles over averages** — p99 is the unluckiest 1%'s experience; averages hide it
- watch `in_flight` climb while throughput stays flat under load: that's the Phase 0
  bottleneck made visible, and the graph Phase 3 (continuous batching) will fix

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
#    HF_HUB_DISABLE_XET=1 avoids a Windows hang in HuggingFace's Xet downloader.
$env:HF_HOME = "$PWD\.cache"; $env:HF_HUB_DISABLE_XET = "1"
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
