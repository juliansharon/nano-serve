"""Phases 2+3: request queue + continuous-batching scheduler.

Phases 0/1 served one request at a time: an HTTP handler grabbed a lock, ran the
whole decode loop, released it. Everyone queued *implicitly* behind the lock and
the GPU ran batch=1 forever.

Two moves happen here:

  Phase 2 — decouple HTTP from the GPU. Handlers never touch the model anymore;
  they submit a Job to a queue and read token events back. ONE scheduler thread
  owns the GPU. (No speedup by itself — but now there's a brain that can decide.)

  Phase 3 — continuous batching. Every iteration of the scheduler loop:
      admit:  pull waiting jobs into the batch (up to max_batch_size)
      step:   ONE forward pass advances EVERY active sequence by one token
      reap:   finished sequences leave the batch IMMEDIATELY — their slot is
              free for a waiting request at the very next step

Why this multiplies throughput: decoding is memory-bandwidth-bound. A decode step
reads ~all model weights whether it advances 1 sequence or 8, so batched tokens
are nearly free. Static batching (collect N, run lockstep) wastes the win twice:
requests wait for a batch to form, then short requests wait for the longest one.
Continuous batching has neither wait — sequences join and leave mid-flight.

Mechanics worth reading closely:
  * Sequences in a batch have different lengths, so the KV-cache is LEFT-padded
    to a common length, with an attention mask zeroing the padding and per-row
    position_ids keeping RoPE correct — the same trick HF generate() uses for
    batched decode, here done by hand.
  * A new request needs a PREFILL (full forward over its prompt) before it can
    join the decode batch. We prefill on admission, then splice its KV tensors
    into the batch. The running batch stalls during that prefill — the
    head-of-line tradeoff vLLM addresses with chunked prefill (later phase).
  * No PagedAttention yet: caches are dense padded tensors. Fine at this scale;
    Phase 4 is where KV memory management becomes the story.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from nano_serve.engine import Engine


@dataclass
class Job:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    cancelled: bool = False
    # runtime state — owned by the scheduler thread after submit()
    prompt_tokens: int = 0
    generated: list[int] = field(default_factory=list)
    emitted_text: str = ""
    t_submit: float = 0.0
    t_first: float | None = None

    def emit_delta(self, tokenizer) -> None:
        text = tokenizer.decode(self.generated, skip_special_tokens=True)
        if text.endswith("�"):  # incomplete UTF-8 sequence — hold back
            return
        delta = text[len(self.emitted_text):]
        if delta:
            self.emitted_text = text
            self.events.put({"type": "token", "text": delta})


class Scheduler:
    def __init__(self, engine: Engine, max_batch_size: int | None = None, metrics=None):
        self.engine = engine
        self.max_batch = max_batch_size or int(os.environ.get("NANO_SERVE_MAX_BATCH", "8"))
        self.metrics = metrics
        self.waiting: "queue.Queue[Job]" = queue.Queue()
        self.active: list[Job] = []

        # Batch state — one row per active job, KV left-padded to a common length.
        self.kv: tuple | None = None           # ((k, v) per layer), each [B, H, L, D]
        self.attn: torch.Tensor | None = None  # [B, L], 0 over left padding
        self.last_ids: torch.Tensor | None = None  # [B, 1] last sampled token per row

        # Chat models often have several stop tokens (e.g. <|im_end|> AND
        # <|endoftext|> for Qwen) — collect them all.
        eos = engine.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if engine.tokenizer.eos_token_id is not None:
            self.eos_ids.add(engine.tokenizer.eos_token_id)

        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, prompt: str, max_tokens: int, temperature: float,
               top_p: float) -> Job:
        job = Job(prompt, max_tokens, temperature, top_p)
        job.t_submit = time.perf_counter()
        self.waiting.put(job)
        return job

    def info(self) -> dict:
        return {"max_batch": self.max_batch, "active": len(self.active),
                "queue_depth": self.waiting.qsize()}

    # -- the loop -----------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            if not self.active:
                self._admit(self.waiting.get())  # idle: block until work arrives
            while len(self.active) < self.max_batch:
                try:
                    self._admit(self.waiting.get_nowait())
                except queue.Empty:
                    break
            if self.active:
                self._step()
            if self.metrics is not None:
                self.metrics.set_scheduler(self.waiting.qsize(), len(self.active),
                                           self.max_batch)

    def _finish(self, job: Job, reason: str) -> None:
        now = time.perf_counter()
        n = len(job.generated)
        first = job.t_first if job.t_first is not None else now
        job.events.put({
            "type": "done",
            "prompt_tokens": job.prompt_tokens,
            "output_tokens": n,
            "ttft_ms": (first - job.t_submit) * 1000,  # includes queue wait — honest
            "tpot_ms": ((now - first) / max(n - 1, 1)) * 1000,
            "total_ms": (now - job.t_submit) * 1000,
            "finish_reason": reason,
        })

    # -- admit: prefill, then splice into the batch ---------------------------------

    @torch.inference_mode()
    def _admit(self, job: Job) -> None:
        if job.cancelled:
            self._finish(job, "cancelled")
            return
        dev = self.engine.device
        ids = self.engine._build_inputs(job.prompt)              # [1, P]
        out = self.engine.model(input_ids=ids, use_cache=True)   # PREFILL
        job.prompt_tokens = ids.shape[1]
        next_id = self.engine._sample(out.logits[0, -1, :], job.temperature, job.top_p)
        if dev == "cuda":
            torch.cuda.synchronize()
        job.t_first = time.perf_counter()

        if next_id in self.eos_ids:
            self._finish(job, "stop")
            return
        job.generated.append(next_id)
        job.emit_delta(self.engine.tokenizer)
        if len(job.generated) >= job.max_tokens:
            self._finish(job, "length")
            return

        kv = out.past_key_values.to_legacy_cache()               # [1, H, P, D] per layer
        mask = torch.ones((1, ids.shape[1]), dtype=torch.long, device=dev)
        last = torch.tensor([[next_id]], dtype=torch.long, device=dev)
        if self.kv is None:
            self.kv, self.attn, self.last_ids = kv, mask, last
        else:
            self._merge(kv, mask, last)
        self.active.append(job)

    @staticmethod
    def _pad_left(kv: tuple, attn: torch.Tensor, n: int) -> tuple[tuple, torch.Tensor]:
        if n == 0:
            return kv, attn
        padded = []
        for k, v in kv:
            z = k.new_zeros((k.shape[0], k.shape[1], n, k.shape[3]))
            padded.append((torch.cat([z, k], dim=2), torch.cat([z, v], dim=2)))
        attn = torch.cat([attn.new_zeros((attn.shape[0], n)), attn], dim=1)
        return tuple(padded), attn

    def _merge(self, kv: tuple, mask: torch.Tensor, last: torch.Tensor) -> None:
        L = max(self.attn.shape[1], mask.shape[1])
        self.kv, self.attn = self._pad_left(self.kv, self.attn, L - self.attn.shape[1])
        kv, mask = self._pad_left(kv, mask, L - mask.shape[1])
        self.kv = tuple(
            (torch.cat([k0, k1], dim=0), torch.cat([v0, v1], dim=0))
            for (k0, v0), (k1, v1) in zip(self.kv, kv)
        )
        self.attn = torch.cat([self.attn, mask], dim=0)
        self.last_ids = torch.cat([self.last_ids, last], dim=0)

    # -- step: one forward pass advances the whole batch ----------------------------

    @torch.inference_mode()
    def _step(self) -> None:
        dev = self.engine.device
        # one new column of attention for the token each row is about to process
        attn = torch.cat([self.attn, self.attn.new_ones((self.attn.shape[0], 1))], dim=1)
        # per-row position of that token = number of REAL tokens before it
        pos = attn.sum(dim=1, keepdim=True) - 1
        cache = DynamicCache.from_legacy_cache(self.kv)
        out = self.engine.model(input_ids=self.last_ids, attention_mask=attn,
                                position_ids=pos, past_key_values=cache, use_cache=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        self.kv = out.past_key_values.to_legacy_cache()
        self.attn = attn

        keep_rows: list[int] = []
        keep_last: list[int] = []
        keep_jobs: list[Job] = []
        for i, job in enumerate(self.active):
            if job.cancelled:
                self._finish(job, "cancelled")
                continue
            nid = self.engine._sample(out.logits[i, -1, :], job.temperature, job.top_p)
            if nid in self.eos_ids:
                self._finish(job, "stop")
                continue
            job.generated.append(nid)
            job.emit_delta(self.engine.tokenizer)
            if len(job.generated) >= job.max_tokens:
                self._finish(job, "length")
                continue
            keep_rows.append(i)
            keep_last.append(nid)
            keep_jobs.append(job)

        self.active = keep_jobs
        if not keep_jobs:
            self.kv = self.attn = self.last_ids = None
            return
        if len(keep_rows) != attn.shape[0]:  # someone left — shrink the batch NOW
            idx = torch.tensor(keep_rows, dtype=torch.long, device=dev)
            self.kv = tuple((k.index_select(0, idx), v.index_select(0, idx))
                            for k, v in self.kv)
            self.attn = self.attn.index_select(0, idx)
        # drop left padding no remaining row needs
        trim = int(self.attn.shape[1] - self.attn.sum(dim=1).max())
        if trim > 0:
            self.kv = tuple((k[:, :, trim:, :], v[:, :, trim:, :]) for k, v in self.kv)
            self.attn = self.attn[:, trim:]
        self.last_ids = torch.tensor(keep_last, dtype=torch.long, device=dev).unsqueeze(1)
