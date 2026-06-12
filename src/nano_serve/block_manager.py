"""Phase 4: the BlockManager — a tiny "OS for note cards".

All KV memory is pre-allocated ONCE at startup as a pool of fixed-size blocks
(default 16 tokens each). A sequence never owns a contiguous strip; it owns a
BLOCK TABLE — an ordered list of block ids — and its cards live wherever the
free-list happened to have space. Logical order lives in the table; physical
order is irrelevant.

Address translation (the entire PagedAttention idea):
    card #i of a sequence  →  block = table[i // block_size]
                              slot  = i %  block_size

What this buys over the Phase 3 padded rectangle:
  * admission/eviction = table edits, never tensor copies
  * no inter-sequence padding: a request's only waste is its own last block's tail
  * a freed block is instantly reusable by anyone (D moves into B's old rooms)

Honest caveat: real vLLM's CUDA kernel READS the pool in place by following the
table mid-attention. We can't write CUDA here, so the scheduler gathers each
sequence's blocks into a contiguous temp before the forward pass — we pay a
copy per step to keep the code readable. The storage/bookkeeping wins (and the
utilization numbers) are the lesson; the kernel is what we borrow conceptually.
"""

from __future__ import annotations

from collections import deque

import torch


class BlockManager:
    def __init__(self, num_layers: int, kv_heads: int, head_dim: int,
                 block_size: int, num_blocks: int, device: str, dtype: torch.dtype):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        # ALL layers in one tensor, blocks on dim 1 — so one indexed read/write
        # serves every layer at once instead of a 24-iteration Python loop.
        # This allocation is the whole KV budget; nothing else is ever
        # allocated for cache storage.
        shape = (num_layers, num_blocks, kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.free: deque[int] = deque(range(num_blocks))
        self.pool_bytes = 2 * self.k_pool.numel() * self.k_pool.element_size()

    def num_free(self) -> int:
        return len(self.free)

    def allocate(self, n: int) -> list[int]:
        if len(self.free) < n:
            raise RuntimeError(f"KV pool exhausted: need {n}, free {len(self.free)}")
        return [self.free.popleft() for _ in range(n)]

    def free_blocks(self, blocks: list[int]) -> None:
        self.free.extend(blocks)

    # -- writes -------------------------------------------------------------------

    def write_prefill(self, table: list[int], k: torch.Tensor, v: torch.Tensor,
                      length: int) -> None:
        """Chop a prefill's K/V ([layers, H, P, D]) into this sequence's blocks."""
        bs = self.block_size
        for j, b in enumerate(table):
            s = j * bs
            e = min(s + bs, length)
            if s >= length:
                break
            self.k_pool[:, b, :, : e - s] = k[:, :, s:e]
            self.v_pool[:, b, :, : e - s] = v[:, :, s:e]

    def write_token(self, block: int, slot: int,
                    k: torch.Tensor, v: torch.Tensor) -> None:
        """Write one new token's K/V ([layers, H, D]) into its block slot —
        all layers in one assignment."""
        self.k_pool[:, block, :, slot] = k
        self.v_pool[:, block, :, slot] = v

    # -- reads --------------------------------------------------------------------

    def gather(self, idx: torch.Tensor, length: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk a block table (as an index tensor) and return the sequence's
        K/V in logical order for ALL layers at once: [layers, H, length, D].
        This is the per-step copy the real PagedAttention kernel avoids."""
        n = idx.shape[0]
        k = self.k_pool.index_select(1, idx)  # [L, n, H, bs, D]
        v = self.v_pool.index_select(1, idx)
        nl, h, d = k.shape[0], k.shape[2], k.shape[4]
        k = k.permute(0, 2, 1, 3, 4).reshape(nl, h, n * self.block_size, d)[:, :, :length]
        v = v.permute(0, 2, 1, 3, 4).reshape(nl, h, n * self.block_size, d)[:, :, :length]
        return k, v

    def stats(self) -> dict:
        used = self.num_blocks - len(self.free)
        return {
            "block_size": self.block_size,
            "blocks_total": self.num_blocks,
            "blocks_used": used,
            "blocks_free": len(self.free),
            "pool_mb": round(self.pool_bytes / 1e6, 1),
        }
