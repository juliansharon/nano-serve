"""In-process metrics for nano-serve — observability v0.

Three kinds of measurement, the same trio every monitoring system is built on:

  counters  — only go up: total requests, total tokens generated
  gauges    — a value *right now*: requests in flight, GPU memory used
  windows   — recent history we compute stats over: tokens/sec in the last 10 s,
              p50/p99 latency over the last 50 requests

Percentiles instead of averages because serving is judged on the *tail*: an average
hides the one user who waited 20 s. p99 = "the experience of the unluckiest 1%".

Everything is guarded by one lock — recording happens from generation worker threads
while /metrics.json reads from the event loop. A background daemon thread samples
nvidia-smi once a second for GPU utilization/memory (good enough for a dashboard;
Phase 6 replaces all of this with Prometheus).
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


class Metrics:
    def __init__(self, gpu_poll_seconds: float = 1.0):
        self._lock = threading.Lock()
        self.started_at = time.time()

        # counters
        self.total_requests = 0
        self.total_output_tokens = 0
        self.total_prompt_tokens = 0
        self.total_aborted = 0

        # gauges
        self.in_flight = 0

        # windows
        self._token_times: deque[float] = deque(maxlen=100_000)  # one stamp per token
        self._requests: deque[dict] = deque(maxlen=200)          # completed requests
        self._gpu: deque[dict] = deque(maxlen=120)               # ~2 min of samples

        # scheduler gauges (Phase 3) — updated by the scheduler loop every step
        self._sched = {"queue_depth": 0, "batch_size": 0, "max_batch": 0}
        # KV pool gauges (Phase 4) — None when running the dense scheduler
        self._kv: dict | None = None

        t = threading.Thread(target=self._gpu_sampler, args=(gpu_poll_seconds,), daemon=True)
        t.start()

    # -- recording (called from worker threads) ----------------------------------

    def request_started(self) -> None:
        with self._lock:
            self.in_flight += 1

    def token_generated(self) -> None:
        with self._lock:
            self._token_times.append(time.time())

    def request_finished(self, record: dict) -> None:
        """record: prompt, prompt_tokens, output_tokens, ttft_ms, tpot_ms, total_ms,
        finish_reason, stream"""
        with self._lock:
            self.in_flight -= 1
            self.total_requests += 1
            self.total_output_tokens += record["output_tokens"]
            self.total_prompt_tokens += record["prompt_tokens"]
            self._requests.append({**record, "ts": time.time()})

    def request_aborted(self) -> None:
        """Client disconnected mid-generation (stream closed early)."""
        with self._lock:
            self.in_flight -= 1
            self.total_aborted += 1

    def set_scheduler(self, queue_depth: int, batch_size: int, max_batch: int) -> None:
        with self._lock:
            self._sched = {"queue_depth": queue_depth, "batch_size": batch_size,
                           "max_batch": max_batch}

    def set_kv(self, kv: dict) -> None:
        with self._lock:
            self._kv = kv

    # -- GPU sampling -------------------------------------------------------------

    def _gpu_sampler(self, interval: float) -> None:
        query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"]
        while True:
            try:
                out = subprocess.run(query, capture_output=True, text=True, timeout=5)
                util, used, total = (float(x) for x in out.stdout.strip().split(","))
                with self._lock:
                    self._gpu.append({"ts": time.time(), "util_pct": util,
                                      "mem_used_mb": used, "mem_total_mb": total})
            except Exception:
                pass  # no nvidia-smi (CPU box) — dashboard just shows no GPU data
            time.sleep(interval)

    # -- reading ------------------------------------------------------------------

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            token_times = list(self._token_times)
            requests = list(self._requests)
            gpu = list(self._gpu)
            sched = dict(self._sched)
            kv = dict(self._kv) if self._kv else None
            totals = {
                "requests": self.total_requests,
                "output_tokens": self.total_output_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "aborted": self.total_aborted,
            }
            in_flight = self.in_flight

        # rolling throughput: tokens stamped within the last N seconds / N
        def tput(window: float) -> float:
            cutoff = now - window
            return sum(1 for t in token_times if t >= cutoff) / window

        # per-second buckets for the last 60 s — the dashboard's throughput chart
        series = [0] * 60
        for t in token_times:
            age = int(now - t)
            if 0 <= age < 60:
                series[59 - age] += 1

        recent = requests[-50:]
        ttfts = sorted(r["ttft_ms"] for r in recent)
        totals_ms = sorted(r["total_ms"] for r in recent)
        tpots = [r["tpot_ms"] for r in recent if r["output_tokens"] > 1]

        return {
            "uptime_s": round(now - self.started_at, 1),
            "in_flight": in_flight,
            "scheduler": sched,
            "kv": kv,
            "totals": totals,
            "throughput_tok_s": {"10s": round(tput(10), 1), "60s": round(tput(60), 1)},
            "throughput_series_60s": series,
            "latency": {
                "ttft_ms_p50": round(_percentile(ttfts, 0.50), 1),
                "ttft_ms_p99": round(_percentile(ttfts, 0.99), 1),
                "tpot_ms_mean": round(sum(tpots) / len(tpots), 2) if tpots else 0.0,
                "total_ms_p50": round(_percentile(totals_ms, 0.50), 1),
                "total_ms_p99": round(_percentile(totals_ms, 0.99), 1),
                "window": len(recent),
            },
            "recent_requests": [
                {k: r[k] for k in ("ts", "prompt", "prompt_tokens", "output_tokens",
                                   "ttft_ms", "tpot_ms", "total_ms", "finish_reason",
                                   "stream")}
                for r in requests[-20:]
            ][::-1],
            "gpu": gpu[-1] if gpu else None,
            "gpu_series": [{"util_pct": g["util_pct"], "mem_used_mb": g["mem_used_mb"]}
                           for g in gpu[-60:]],
        }
