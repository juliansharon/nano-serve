"""Phase 1 engine: load a model once, generate one request at a time, stream tokens.

The decode loop is written by hand (instead of model.generate) for two reasons:
  1. We can time the FIRST token (TTFT) separately from the rest (TPOT).
  2. Every later phase (batching, paged KV-cache) is a modification of this exact
     loop — so it pays to see it in the open.

Phase 1 change: `stream()` is now the core — a generator that yields each token the
moment it's sampled. Streaming doesn't make generation faster; it makes it *feel*
faster: the user sees the first word at TTFT (~100 ms) instead of waiting for the
whole reply (~3 s). Perceived latency drops ~30x for free.

A single global lock still serializes generation: the GPU does one request at a
time. That serialization IS the bottleneck Phases 2-3 remove.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float        # time to first token
    tpot_ms: float        # mean time per output token (after the first)
    total_ms: float
    finish_reason: str     # "stop" | "length"


class Engine:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0

        # Phase 1: one request on the GPU at a time.
        self._lock = threading.Lock()

    # -- helpers ----------------------------------------------------------------

    def _build_inputs(self, prompt: str) -> torch.Tensor:
        """Wrap the prompt in the model's chat template, return input ids [1, T]."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """Pick the next token id from the last-position logits [vocab]."""
        if temperature <= 0.0:
            return int(torch.argmax(logits))
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative - sorted_probs > top_p  # keep tokens up to the cutoff
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            choice = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[choice])
        return int(torch.multinomial(probs, num_samples=1))

    # -- the loop ---------------------------------------------------------------

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Iterator[dict]:
        """Yield {"type": "token", "text": ...} per token, then one {"type": "done", ...}.

        Tokens are byte-pair pieces, not words — a single emoji can span two tokens.
        So we decode the full sequence each step and emit the *delta* text, holding
        back when the tail is an incomplete UTF-8 sequence (shows up as U+FFFD).
        """
        with self._lock:  # serialize: one request at a time
            input_ids = self._build_inputs(prompt)
            prompt_len = input_ids.shape[1]
            eos_id = self.tokenizer.eos_token_id

            start = time.perf_counter()
            ttft = None
            generated: list[int] = []
            emitted_text = ""
            past = None
            cur = input_ids
            finish_reason = "length"

            for _step in range(max_tokens):
                out = self.model(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = self._sample(out.logits[0, -1, :], temperature, top_p)

                if self.device == "cuda":
                    torch.cuda.synchronize()  # honest timing — kernels are async
                if ttft is None:
                    ttft = time.perf_counter() - start

                if next_id == eos_id:
                    finish_reason = "stop"
                    break

                generated.append(next_id)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if not text.endswith("�"):  # don't emit half a multibyte char
                    delta = text[len(emitted_text):]
                    if delta:
                        emitted_text = text
                        yield {"type": "token", "text": delta}

                # After the prefill, we feed back only the single new token; the rest
                # of the context lives in the KV-cache (`past`). This is the whole
                # point of the cache — we never re-process the prompt.
                cur = torch.tensor([[next_id]], device=self.device)

            total = time.perf_counter() - start
            n_out = len(generated)
            # TPOT = time spent on tokens after the first, averaged.
            tpot = ((total - ttft) / max(n_out - 1, 1)) if ttft is not None else 0.0

            yield {
                "type": "done",
                "prompt_tokens": prompt_len,
                "output_tokens": n_out,
                "ttft_ms": (ttft or 0.0) * 1000,
                "tpot_ms": tpot * 1000,
                "total_ms": total * 1000,
                "finish_reason": finish_reason,
            }

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7,
                 top_p: float = 0.95) -> GenResult:
        """Non-streaming convenience: consume the stream, return the assembled result."""
        parts: list[str] = []
        done: dict = {}
        for ev in self.stream(prompt, max_tokens, temperature, top_p):
            if ev["type"] == "token":
                parts.append(ev["text"])
            else:
                done = ev
        return GenResult(text="".join(parts), prompt_tokens=done["prompt_tokens"],
                         output_tokens=done["output_tokens"], ttft_ms=done["ttft_ms"],
                         tpot_ms=done["tpot_ms"], total_ms=done["total_ms"],
                         finish_reason=done["finish_reason"])

    def info(self) -> dict:
        gpu = None
        if self.device == "cuda":
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "mem_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                "mem_reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
            }
        return {
            "model": self.model_name,
            "device": self.device,
            "load_seconds": round(self.load_seconds, 2),
            "gpu": gpu,
        }
