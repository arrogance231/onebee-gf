from __future__ import annotations

import json
import random
import time
from pathlib import Path

from pydantic import BaseModel

from onebee.inference.engine import GenerationConfig, Generator


class BenchResult(BaseModel):
    context_length: int
    batch_size: int
    ttft_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    peak_vram_mb: float | None


_SYNTHETIC_WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "a",
    "is",
    "that",
    "for",
    "it",
    "as",
    "with",
    "was",
    "on",
    "be",
    "this",
    "by",
    "not",
    "are",
    "from",
    "at",
    "or",
    "an",
    "they",
    "which",
    "one",
    "would",
    "all",
    "will",
    "there",
    "say",
    "who",
    "make",
    "when",
    "can",
    "more",
    "if",
    "no",
    "man",
    "out",
    "other",
    "so",
    "what",
    "time",
    "up",
    "go",
    "about",
    "than",
    "into",
    "could",
    "state",
    "only",
    "new",
    "year",
    "some",
    "take",
    "come",
    "these",
    "know",
    "see",
]


def _synthesize_filler_text(target_tokens: int, *, seed: int = 42) -> str:
    rng = random.Random(seed)
    words: list[str] = []
    while len(words) < target_tokens:
        words.append(rng.choice(_SYNTHETIC_WORDS))
    return " ".join(words[:target_tokens])


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _build_prompt(
    target_length: int,
    prompt_source: str | None = None,
    *,
    seed: int = 42,
) -> str:
    if prompt_source is not None:
        source_words = prompt_source.split()
        if _estimate_tokens(prompt_source) >= target_length:
            return " ".join(source_words[:target_length])
        else:
            pad_needed = target_length - _estimate_tokens(prompt_source)
            filler = _synthesize_filler_text(pad_needed, seed=seed)
            return prompt_source + " " + filler
    return _synthesize_filler_text(target_length, seed=seed)


def run_latency_bench(
    engine: Generator,
    context_lengths: list[int] | None = None,
    batch_size: int = 1,
    n_repeats: int = 3,
    prompt_source: str | None = None,
) -> list[BenchResult]:
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096]

    config = GenerationConfig(max_new_tokens=64)

    results: list[BenchResult] = []

    for ctx_len in context_lengths:
        prompt = _build_prompt(ctx_len, prompt_source)
        messages = [{"role": "user", "content": prompt}]

        all_ttft: list[float] = []
        all_decode_tok_s: list[float] = []

        for _ in range(n_repeats):
            gen_result = engine.generate(messages, config)
            all_ttft.append(gen_result.ttft_ms)
            decode_time_s = (gen_result.total_ms - gen_result.ttft_ms) / 1000.0
            all_decode_tok_s.append(
                gen_result.completion_tokens / decode_time_s if decode_time_s > 0 else 0.0
            )

        avg_ttft = sum(all_ttft) / len(all_ttft) if all_ttft else 0.0
        avg_decode = sum(all_decode_tok_s) / len(all_decode_tok_s) if all_decode_tok_s else 0.0

        prefill_config = GenerationConfig(max_new_tokens=1)
        prefill_start = time.perf_counter()
        engine.generate(messages, prefill_config)
        prefill_total_s = time.perf_counter() - prefill_start
        prefill_tok_s_val = (
            _estimate_tokens(prompt) / prefill_total_s if prefill_total_s > 0 else 0.0
        )

        peak_vram: float | None = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        results.append(
            BenchResult(
                context_length=ctx_len,
                batch_size=batch_size,
                ttft_ms=avg_ttft,
                prefill_tok_s=prefill_tok_s_val,
                decode_tok_s=avg_decode,
                peak_vram_mb=peak_vram,
            )
        )

    return results


def save_bench_results(
    results: list[BenchResult],
    path: str,
    *,
    engine_name: str,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output: dict = {
        "engine": engine_name,
        "results": [r.model_dump() for r in results],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
