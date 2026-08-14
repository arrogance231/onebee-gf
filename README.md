# onebee-gf

**Stretching a small multimodal LLM through post-training and external cognitive architecture.**

How much apparent capability can be recovered from a small (~1–4B parameter), vision-capable
language model through post-training, external memory, retrieval, state modeling, distillation,
and inference-time cognitive architecture — and how much of that survives quantization and runs
locally on a smartphone? This project builds a small multimodal companion model — able to see
and respond to images the user shares, not just text — augmented with an external memory system
(short-term, episodic, and semantic tiers), hybrid dense+BM25 retrieval, a token-budgeted
context builder, and a LoRA/DPO post-training pipeline, and evaluates it against an adversarial
personalized-memory benchmark (PMB) with abstention and contradiction traps — not just recall.
The stress test is a persona expected to hold continuity across **years** of conversation, which
is why external memory rather than raw context length is the project's central bet.

See [`docs/research_questions.md`](docs/research_questions.md) for the full research question
hierarchy and hypotheses this project is testing.

## Results so far (real, measured, not projected)

Base model: `gemma4-e2b` (`google/gemma-4-E2B-it`), chosen via a real multimodal bake-off — see
[`docs/adr/0001-model-selection.md`](docs/adr/0001-model-selection.md).

| System | Description | pra_lenient | uar |
|---|---|---|---|
| A | raw model, no memory | 0.16% | 13.75% |
| B | + LoRA SFT, no memory | 0.16% | 16.25% |
| D | + hybrid retrieval memory (k=8) | 15.10% | 8.75% |
| **E** | **+ SFT and memory together** | **17.76%** | **33.75%** |

Memory retrieval alone recovers real personalized-recall accuracy from a model with zero
context (H1); SFT alone does not (H4, as expected); combining both beats either alone (H5) at
v0 data scale. A k-sweep confirms an inverted-U in retrieved-memory count peaking at k=8 (H10).
DPO preference optimization on top of SFT was also tried (H6-H7) — full writeup in
[`docs/dpo_results.md`](docs/dpo_results.md).

**Follow-up at proper training scale** (10x the data — 40 personas, 2242 SFT examples, 2277 DPO
pairs): DPO's effect on persona consistency becomes clearly distinguishable (a 39.0% vs 20.0%
pairwise win rate, up from v0's 24.8% vs 21.9% — not just noise anymore). An apparent
calibration regression turned out, on investigation, to be two real bugs: a dedup step was
silently collapsing ~227 intended abstention training examples down to 1, and the eval
harness's rule-based abstention detector didn't recognize the model's (correct) trained
phrasing. Fixing both revealed the true story — UAR jumps to 96.25% (far above v0's 33.75%) —
but also exposed a genuine new tradeoff: the restored abstention signal now makes the model
over-hedge on 69.2% of genuinely answerable questions. Reported honestly, root-caused, not
hidden. Full writeup: [`docs/proper_scale_results.md`](docs/proper_scale_results.md).

Full writeups: [`docs/day3_memory_results.md`](docs/day3_memory_results.md) (memory system),
[`docs/day4_sft_results.md`](docs/day4_sft_results.md) (SFT), and
[`docs/model_quirks.md`](docs/model_quirks.md) (real environment/API issues found and fixed
along the way — worth reading before assuming any of this "just works").

## Repo layout

- `src/onebee/` — installable package: memory, retrieval, context, state, inference, training,
  evaluation.
- `configs/` — Hydra configs; every experiment is a composed config, no hardcoded params.
- `scripts/` — bake-off, benchmark construction, contamination check, figure generation.
- `data/` — versioned benchmarks, SFT/preference/distillation/CPT data, populated memory stores.
- `experiments/` — one directory per experiment, pre-registered hypothesis before results.
- `results/` — canonical numbers and figures, versioned by release tag.
- `mobile/` — on-device runtime build/convert scripts (llama.cpp/MLC/ExecuTorch).
- `docs/` — ADRs, hardware notes, reproduction guide, failure taxonomy.
- `paper/` — LaTeX source for the accompanying paper.

## Install

```bash
uv sync            # base install (CPU-only, lint/test)
uv sync --extra gpu # + torch/transformers/vllm/etc. for training and inference
```

## Run tests

```bash
pytest
```

## License

Code: Apache-2.0 (`LICENSE`). Benchmarks/data: CC-BY-4.0 (`LICENSE-DATA`).
