"""Load-test nano-serve and report the four numbers that define serving.

    python examples/bench.py --concurrency 1     # baseline latency
    python examples/bench.py --concurrency 8     # watch throughput stay flat (Phase 0)

It fires `--requests` total requests, `--concurrency` of them in flight at once, and
reports aggregate throughput plus the latency distribution. In Phase 0 throughput is
roughly constant no matter the concurrency, because the GPU serves one at a time. The
job of Phases 2–4 is to make this number climb as concurrency rises.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

PROMPTS = [
    "Explain what a KV-cache is in one sentence.",
    "Write a haiku about fast GPUs.",
    "What is continuous batching? Be brief.",
    "List three ways to reduce inference latency.",
    "Why does p99 latency matter more than the average?",
]


async def one(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        url, json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    )
    r.raise_for_status()
    body = r.json()
    body["client_latency_ms"] = (time.perf_counter() - t0) * 1000
    return body


async def run(args: argparse.Namespace) -> None:
    url = f"{args.host}/generate"
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def worker(i: int) -> None:
        async with sem:
            prompt = PROMPTS[i % len(PROMPTS)]
            results.append(await one(client, url, prompt, args.max_tokens))

    async with httpx.AsyncClient(timeout=120) as client:
        wall0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(args.requests)))
        wall = time.perf_counter() - wall0

    lat = sorted(r["client_latency_ms"] for r in results)
    ttft = [r["ttft_ms"] for r in results]
    out_tokens = sum(r["output_tokens"] for r in results)

    def pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print(f"\n=== nano-serve benchmark ===")
    print(f"requests={args.requests}  concurrency={args.concurrency}  "
          f"max_tokens={args.max_tokens}")
    print(f"wall clock          : {wall:.2f} s")
    print(f"output tokens total : {out_tokens}")
    print(f"THROUGHPUT          : {out_tokens / wall:.1f} tok/s   <-- the number to beat")
    print(f"TTFT     mean/p99   : {statistics.mean(ttft):.1f} / {pct(sorted(ttft),0.99):.1f} ms")
    print(f"latency  p50/p99    : {pct(lat,0.50):.0f} / {pct(lat,0.99):.0f} ms")
    print(f"latency  min/max    : {lat[0]:.0f} / {lat[-1]:.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    asyncio.run(run(ap.parse_args()))
