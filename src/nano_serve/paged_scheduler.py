"""Phase 4: continuous batching on a PAGED KV-cache.

Same loop as Phase 3 (admit → step → reap), different storage:

  Phase 3: one dense left-padded rectangle; every admit/evict rebuilds it,
           every row pads to the longest sequence.
  Phase 4: a BlockManager pool. A sequence owns a block TABLE; admission and
           eviction are table edits, the only padding waste is a sequence's
           own last-block tail, and a freed block is instantly reusable.

New behavior unlocked by paging — PREEMPTION: when the pool runs dry mid-step,
the youngest sequence is evicted entirely (blocks freed, job pushed back to the
front of the queue). When memory frees up, it is rebuilt by re-prefilling its
prompt PLUS everything it already generated — recompute buys back memory.
Tokens already streamed to the client are never re-emitted (emit_delta only
sends text beyond what was already sent), so the user never notices.

Per-step flow:
  1. ensure every active sequence has a free slot for its incoming token's KV
     (allocate a block when its table is full; preempt if the pool is empty)
  2. GATHER each sequence's blocks into a contiguous left-padded temp batch
     (the per-step copy a real PagedAttention kernel avoids — see block_manager)
  3. one forward pass advances everyone; the new K/V column is written back
     into each sequence's block slot; the temp is discarded
  4. sample, emit, finish/evict — freed sequences return blocks to the pool
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

import torch
from transformers.cache_utils import DynamicCache

from nano_serve.block_manager import BlockManager
from nano_serve.engine import Engine
from nano_serve.scheduler import Job


class PagedScheduler:
    def __init__(self, engine: Engine, metrics=None):
        self.engine = engine
        self.metrics = metrics
        self.max_batch = int(os.environ.get("NANO_SERVE_MAX_BATCH", "8"))
        self.block_size = int(os.environ.get("NANO_SERVE_BLOCK_SIZE", "16"))
        num_blocks = int(os.environ.get("NANO_SERVE_KV_BLOCKS", "256"))

        cfg = engine.model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.bm = BlockManager(
            num_layers=cfg.num_hidden_layers,
            kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            block_size=self.block_size,
            num_blocks=num_blocks,
            device=engine.device,
            dtype=engine.model.dtype,
        )

        eos = engine.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if engine.tokenizer.eos_token_id is not None:
            self.eos_ids.add(engine.tokenizer.eos_token_id)

        self.waiting: deque[Job] = deque()
        self._cond = threading.Condition()
        self.active: list[Job] = []
        self.preemptions = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, prompt: str, max_tokens: int, temperature: float,
               top_p: float) -> Job:
        job = Job(prompt, max_tokens, temperature, top_p)
        job.t_submit = time.perf_counter()
        with self._cond:
            self.waiting.append(job)
            self._cond.notify()
        return job

    def info(self) -> dict:
        return {"impl": "paged", "max_batch": self.max_batch,
                "active": len(self.active), "queue_depth": len(self.waiting),
                "preemptions": self.preemptions, "kv": self.bm.stats()}

    # -- the loop -------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            if not self.active:
                with self._cond:
                    while not self.waiting:
                        self._cond.wait()
            self._admit_waiting()
            if self.active:
                self._step()
            self._report()

    def _report(self) -> None:
        if self.metrics is None:
            return
        self.metrics.set_scheduler(len(self.waiting), len(self.active), self.max_batch)
        stored = sum(j.cache_len for j in self.active)
        stats = self.bm.stats()
        used_slots = stats["blocks_used"] * self.block_size
        dense_slots = (len(self.active) * max((j.cache_len for j in self.active), default=0))
        self.metrics.set_kv({
            **stats,
            "preemptions": self.preemptions,
            "tokens_stored": stored,
            "util_paged_pct": round(stored / used_slots * 100, 1) if used_slots else 0.0,
            "util_dense_pct": round(stored / dense_slots * 100, 1) if dense_slots else 0.0,
        })

    def _finish(self, job: Job, reason: str) -> None:
        if getattr(job, "block_table", None):
            self.bm.free_blocks(job.block_table)  # eviction = a free-list append
            job.block_table = []
        now = time.perf_counter()
        n = len(job.generated)
        first = job.t_first if job.t_first is not None else now
        job.events.put({
            "type": "done", "prompt_tokens": job.prompt_tokens, "output_tokens": n,
            "ttft_ms": (first - job.t_submit) * 1000,
            "tpot_ms": ((now - first) / max(n - 1, 1)) * 1000,
            "total_ms": (now - job.t_submit) * 1000,
            "finish_reason": reason,
        })

    # -- admission (also used to RESUME preempted jobs) --------------------------------

    def _admit_waiting(self) -> None:
        while len(self.active) < self.max_batch:
            with self._cond:
                if not self.waiting:
                    return
                job = self.waiting[0]
                if job.cancelled:
                    self.waiting.popleft()
                    self._finish(job, "cancelled")
                    continue
                # admission control: don't start what the pool can't hold
                if not hasattr(job, "prompt_ids"):
                    job.prompt_ids = self.engine._build_inputs(job.prompt)
                    job.prompt_tokens = job.prompt_ids.shape[1]
                worst_case = math.ceil((job.prompt_tokens + job.max_tokens) / self.block_size)
                if worst_case > self.bm.num_blocks:
                    self.waiting.popleft()
                    self._finish(job, "rejected_too_long")
                    continue
                needed = math.ceil((job.prompt_tokens + len(job.generated) + 1) / self.block_size)
                if needed > self.bm.num_free():
                    return  # head-of-line waits for memory; preemption will free some
                self.waiting.popleft()
            self._prefill(job)

    @torch.inference_mode()
    def _prefill(self, job: Job) -> None:
        dev = self.engine.device
        # On preemption-resume, generated tokens are FORCED back through prefill —
        # the cache is rebuilt deterministically, already-streamed text stays valid.
        ids = job.prompt_ids
        if job.generated:
            gen = torch.tensor([job.generated], dtype=torch.long, device=dev)
            ids = torch.cat([ids, gen], dim=1)
        out = self.engine.model(input_ids=ids, use_cache=True)
        n = ids.shape[1]
        job.block_table = self.bm.allocate(math.ceil(n / self.block_size))
        job.cache_len = n
        legacy = out.past_key_values.to_legacy_cache()
        k_all = torch.stack([k[0] for k, _ in legacy])  # [layers, H, P, D]
        v_all = torch.stack([v[0] for _, v in legacy])
        self.bm.write_prefill(job.block_table, k_all, v_all, n)

        next_id = self.engine._sample(out.logits[0, -1, :], job.temperature, job.top_p)
        if dev == "cuda":
            torch.cuda.synchronize()
        if job.t_first is None:
            job.t_first = time.perf_counter()

        if next_id in self.eos_ids:
            self._finish(job, "stop")
            return
        job.generated.append(next_id)
        job.emit_delta(self.engine.tokenizer)
        if len(job.generated) >= job.max_tokens:
            self._finish(job, "length")
            return
        job.next_input = next_id
        self.active.append(job)

    # -- preemption ---------------------------------------------------------------

    def _preempt(self) -> None:
        """Evict the YOUNGEST sequence: free its blocks now, recompute it later."""
        victim = self.active.pop()
        self.bm.free_blocks(victim.block_table)
        victim.block_table = []
        victim.cache_len = 0
        self.preemptions += 1
        with self._cond:
            self.waiting.appendleft(victim)  # front of the line when memory frees

    # -- step ---------------------------------------------------------------------

    @torch.inference_mode()
    def _step(self) -> None:
        dev = self.engine.device
        bs = self.block_size

        # 1) every sequence needs a slot for the incoming token's K/V
        for job in list(self.active):
            if job not in self.active:
                continue  # got preempted by an earlier iteration
            if job.cache_len == len(job.block_table) * bs:  # table is full
                while self.bm.num_free() == 0 and job in self.active:
                    self._preempt()
                if job not in self.active:
                    continue
                job.block_table.append(self.bm.allocate(1)[0])
        if not self.active:
            return

        # 2) gather every sequence's blocks into one left-padded temp batch
        B = len(self.active)
        L = max(j.cache_len for j in self.active)
        cfg = self.engine.model.config
        H = cfg.num_key_value_heads
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        n_layers = cfg.num_hidden_layers
        dtype = self.engine.model.dtype

        k_t = torch.zeros((n_layers, B, H, L, D), dtype=dtype, device=dev)
        v_t = torch.zeros((n_layers, B, H, L, D), dtype=dtype, device=dev)
        attn = torch.zeros((B, L + 1), dtype=torch.long, device=dev)
        for i, job in enumerate(self.active):
            idx = torch.tensor(job.block_table, dtype=torch.long, device=dev)
            off = L - job.cache_len
            k, v = self.bm.gather(idx, job.cache_len)  # all layers in one read
            k_t[:, i, :, off:] = k
            v_t[:, i, :, off:] = v
            attn[i, off:] = 1  # real cache cols + the incoming token's col

        last_ids = torch.tensor([[j.next_input] for j in self.active],
                                dtype=torch.long, device=dev)
        pos = attn.sum(dim=1, keepdim=True) - 1
        cache = DynamicCache.from_legacy_cache(
            tuple((k_t[l], v_t[l]) for l in range(n_layers)))

        # 3) one forward pass advances everyone
        out = self.engine.model(input_ids=last_ids, attention_mask=attn,
                                position_ids=pos, past_key_values=cache,
                                use_cache=True)
        if dev == "cuda":
            torch.cuda.synchronize()

        # write each sequence's NEW K/V column into its own block slot
        updated = out.past_key_values.to_legacy_cache()
        k_new = torch.stack([k[:, :, -1, :] for k, _ in updated])  # [layers, B, H, D]
        v_new = torch.stack([v[:, :, -1, :] for _, v in updated])
        for i, job in enumerate(self.active):
            block = job.block_table[job.cache_len // bs]
            slot = job.cache_len % bs
            self.bm.write_token(block, slot, k_new[:, i], v_new[:, i])
            job.cache_len += 1

        # 4) sample, emit, reap — finished sequences give their blocks back
        survivors: list[Job] = []
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
            job.next_input = nid
            survivors.append(job)
        self.active = survivors
