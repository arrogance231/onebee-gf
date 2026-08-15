# small-mind-companion

**Stretching a small multimodal LLM through post-training and external cognitive architecture.**

An active, open-source research project: real training runs, real evaluation harnesses, real
bugs found and fixed with full root-cause writeups — not a toy demo. Every result below links
to a doc with the honest methodology and limitations behind it, including negative/inconclusive
findings reported as such.

## Table of contents

- [Overview](#overview)
- [Key results](#key-results)
- [Engineering highlights](#engineering-highlights)
- [Architecture](#architecture)
- [Research questions & hypotheses](#research-questions--hypotheses)
- [Model weights (HF Hub)](#model-weights-hf-hub)
- [Quantization](#quantization)
- [Repo layout](#repo-layout)
- [Install & quickstart](#install--quickstart)
- [Testing](#testing)
- [Full documentation index](#full-documentation-index)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

How much apparent capability can be recovered from a small (~2–8B parameter), vision-capable
language model through post-training, external memory, retrieval, state modeling, distillation,
and inference-time cognitive architecture — and how much of that survives quantization and runs
locally on a smartphone?

This project builds a small multimodal companion model — able to see and respond to images the
user shares, not just text — augmented with:

- an **external memory system** (short-term, episodic, and semantic tiers)
- **hybrid dense + BM25 retrieval** with recency/importance weighting
- a **token-budgeted context builder**
- a **LoRA SFT → DPO post-training pipeline**
- **GGUF quantization** for on-device inference

and evaluates it against an adversarial **personalized-memory benchmark (PMB)** with abstention
and contradiction traps — not just recall. The stress test is a persona expected to hold
continuity across **years** of conversation, which is why external memory rather than raw
context length is the project's central bet.

Full research-question hierarchy and hypotheses: [`docs/research_questions.md`](docs/research_questions.md).

## Key results

Base model: `gemma-4-E2B-it` (`google/gemma-4-E2B-it`), chosen via a real multimodal bake-off
— see [`docs/adr/0001-model-selection.md`](docs/adr/0001-model-selection.md).

| System | Description | pra_lenient | uar |
|---|---|---|---|
| A | raw model, no memory | 0.16% | 13.75% |
| B | + LoRA SFT, no memory | 0.16% | 16.25% |
| D | + hybrid retrieval memory (k=8) | 15.10% | 8.75% |
| **E** | **+ SFT and memory together** | **17.76%** | **33.75%** |

Memory retrieval alone recovers real personalized-recall accuracy from a model with zero
context (H1); SFT alone does not (H4, as expected); combining both beats either alone (H5) at
v0 data scale. A k-sweep confirms an inverted-U in retrieved-memory count peaking at k=8 (H10).

**At proper training scale** (10x the data — 40 personas, 2242 SFT examples, 2277 DPO pairs):

- **UAR: 70.0%** (2x the v0 baseline of 33.75%), with false-abstention on answerable questions
  cut to **32.1%** after a real root-cause-and-fix cycle (see [Engineering highlights](#engineering-highlights)).
- **DPO pairwise win-rate gap: 45.7% vs 21.0%** (24.7 percentage points) — the strongest and
  cleanest preference-optimization signal observed across every run in this project.

Full writeups: [`docs/proper_scale_results.md`](docs/proper_scale_results.md) (the current
authoritative results doc), [`docs/day3_memory_results.md`](docs/day3_memory_results.md)
(memory system), [`docs/day4_sft_results.md`](docs/day4_sft_results.md) (SFT),
[`docs/dpo_results.md`](docs/dpo_results.md) (DPO).

## Engineering highlights

This project treats "the number looked good" as a signal to investigate, not a result to trust
— every finding below was caught by that discipline, not luck.

- **Root-caused a calibration regression to two independent bugs, then fixed both and
  re-verified end-to-end.** Scaling the training data 10x appeared to make the model *worse* at
  abstaining on unanswerable questions. Investigation found: (1) a naive text-based dedup step
  in the data-generation script was silently collapsing ~227 intended abstention training
  examples down to 1 (fixed-template responses looked like exact duplicates); (2) after fixing
  that, the eval harness's own abstention detector didn't recognize the model's newly-correct
  phrasing, making a genuine improvement look like a further regression. Fixing both exposed a
  real third issue — over-correction into excessive hedging — which was then also fixed by
  rebalancing the training-data ratios. Three real, sequential root causes, each confirmed by
  reading actual model outputs, not just trusting aggregate metrics. Full trail:
  [`docs/proper_scale_results.md`](docs/proper_scale_results.md),
  [`docs/model_quirks.md`](docs/model_quirks.md) #16-17.
- **Found and fixed a rubric-construction bug that inflated a zero-context baseline to ~94%
  accuracy** — Python operator-precedence in a string-concatenation expression silently dropped
  the gold answer from the judge's rubric whenever a probe had no listed alternatives (the
  common case). Caught because a model with no memory access scoring 94% on personalized-recall
  questions is *definitionally* impossible — treated as a bug signal, not a result.
  [`docs/model_quirks.md`](docs/model_quirks.md).
- **21 real environment/API/tooling bugs found, fixed, and documented** across the stack — wrong
  `AutoModel` class for multimodal models, a cuDNN/Blackwell/conv3d incompatibility, four
  `trl`/`transformers` breaking API changes across versions, a chat-template bug that silently
  dropped user text for one model family, an `hf` GGUF conversion tool that only exports the
  vision projector unless invoked twice, a cross-version tokenizer-config format mismatch
  between training and conversion environments, and CLI-flag traps that look exactly like real
  performance bugs (an interactive-mode default that hangs silently on closed stdin) until
  traced to their actual cause. Full log: [`docs/model_quirks.md`](docs/model_quirks.md) — every
  entry has a "why this happened" and "how it was fixed," not just "it broke."
- **A blocked technique (ORPO) documented as a real blocker, not silently skipped or faked
  around.** `trl`'s ORPO support doesn't exist in this environment's pinned version at all —
  rather than pin a risky alternate version that could break the working SFT/DPO pipeline, this
  is tracked openly as deferred, with the exact missing symbols and available alternatives
  recorded.

## Architecture

```
User message + image
        │
        ▼
┌─────────────────────┐      ┌──────────────────────────┐
│  Hybrid Retriever    │◄────►│  Memory Store (SQLite)   │
│  dense (e5-small) +  │      │  short-term / episodic /  │
│  BM25, RRF fusion     │      │  semantic tiers           │
└──────────┬───────────┘      └──────────────────────────┘
           │ top-k memories
           ▼
┌─────────────────────┐
│  Context Builder      │  persona card + memories + recent turns,
│  (token-budgeted)     │  assembled as real system/user role turns
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  gemma-4-E2B-it       │  LoRA SFT → DPO post-trained,
│  (multimodal)         │  quantized to GGUF for on-device inference
└──────────────────────┘
```

Evaluated against **PMB** (Personalized Memory Benchmark) — 688 adversarial probes across 8
categories (factual, episodic, temporal, preference, continuity, outdated-fact, distractor,
unanswerable), scored by an LLM judge with position-bias control (dual-order scoring) plus
rule-based abstention detection.

## Research questions & hypotheses

The project runs on a pre-registered hypothesis discipline — git history itself is the
pre-registration record (hypotheses and eval design committed before results). Full RQ/H list,
Week 1/2/3 scope, and design notes for unbuilt future work:
[`docs/research_questions.md`](docs/research_questions.md).

| Area | Status |
|---|---|
| Memory + retrieval (H1, H3, H10, H11) | Done — [`docs/day3_memory_results.md`](docs/day3_memory_results.md) |
| SFT (H4, H5) | Done — [`docs/day4_sft_results.md`](docs/day4_sft_results.md) |
| DPO (H6, H7) | Done — [`docs/dpo_results.md`](docs/dpo_results.md), [`docs/proper_scale_results.md`](docs/proper_scale_results.md) |
| GGUF quantization | Done — [`docs/quantization_results.md`](docs/quantization_results.md) |
| Distillation (H23) | In progress — `src/onebee/training/distill.py` |
| ORPO | Blocked upstream (Week 3) — [`docs/model_quirks.md`](docs/model_quirks.md) #15 |
| Abliteration research (H22) | Scoped, not started — [`docs/research_questions.md`](docs/research_questions.md) |

## Model weights (HF Hub)

Repo names still carry the project's old `onebee-gf` name (from before this repo was renamed
to `small-mind-companion`) — renaming them would mean recreating and re-uploading tens of GB
per repo, not worth it for a naming-only change. Every repo's model card links back here.

| Checkpoint | Description |
|---|---|
| [`onebee-gf-sft-v0`](https://huggingface.co/arrochi112/onebee-gf-sft-v0) | Day 4 v0 SFT (202 train examples) |
| [`onebee-gf-sft-v1`](https://huggingface.co/arrochi112/onebee-gf-sft-v1) | Proper-scale SFT, rebalanced (2232 train examples) — **current best SFT** |
| [`onebee-gf-dpo-v0`](https://huggingface.co/arrochi112/onebee-gf-dpo-v0) | Week 2 DPO v0 (1 epoch, 200 pairs) |
| [`onebee-gf-dpo-v1-4epoch`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-4epoch) | DPO v0 data, 4 epochs (overfitting experiment) |
| [`onebee-gf-dpo-v1-scale`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale) | Proper-scale DPO, rebalanced base — **current best overall** |
| [`onebee-gf-dpo-v1-scale-gguf`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations of the current-best checkpoint |

## Quantization

The current-best checkpoint is quantized to the full standard GGUF spread (F16 through Q2_K,
12 levels, plus the vision projector) for `llama.cpp`-based local/on-device inference.

| Quant | Size | Generation speed (CPU, 30 threads) |
|---|---|---|
| F16 | 8.64 GiB | 26.15 t/s |
| Q8_0 | 4.61 GiB | 43.07 t/s |
| **Q4_K_M** | **3.18 GiB** | **58.00 t/s** (recommended default) |

Real generation and multimodal (vision) capability verified post-quantization, not just
file-size checks — four real `llama.cpp` tooling bugs found and fixed along the way. Full
writeup: [`docs/quantization_results.md`](docs/quantization_results.md).

## Repo layout

- `src/onebee/` — installable package: memory, retrieval, context, state, inference, training,
  evaluation.
- `configs/` — every experiment is a composed YAML config, no hardcoded params.
- `scripts/` — bake-off, benchmark construction, contamination checking, figure generation.
- `data/` — versioned benchmarks, SFT/preference/distillation data, populated memory stores.
- `results/` — canonical numbers and figures, versioned by pass.
- `mobile/` — on-device runtime build/convert scripts (llama.cpp/MLC/ExecuTorch).
- `docs/` — ADRs, hardware notes, results writeups, and the full bug/quirk log.
- `tests/` — 415+ unit tests, all real (no vacuous assertions), run in CI.

## Install & quickstart

```bash
uv sync             # base install (CPU-only, lint/test)
uv sync --extra gpu --extra judge --extra dev  # + training/inference/eval deps
```

## Testing

```bash
pytest   # 415+ tests, ~3s
```

## Full documentation index

| Doc | What it covers |
|---|---|
| [`research_questions.md`](docs/research_questions.md) | Full RQ/H hierarchy, Week 1-3 scope, future design notes |
| [`adr/0001-model-selection.md`](docs/adr/0001-model-selection.md) | Base model bake-off and decision rationale |
| [`day3_memory_results.md`](docs/day3_memory_results.md) | Memory + retrieval system results |
| [`day4_sft_results.md`](docs/day4_sft_results.md) | Day 4 SFT results (v0 scale) |
| [`dpo_results.md`](docs/dpo_results.md) | DPO results (v0 + overfitting experiment) |
| [`proper_scale_results.md`](docs/proper_scale_results.md) | Proper-scale SFT/DPO + the full bug-investigation trail |
| [`quantization_results.md`](docs/quantization_results.md) | GGUF quantization, benchmarks, bugs found |
| [`model_quirks.md`](docs/model_quirks.md) | Every real environment/API/tooling bug found, with root cause and fix |
| [`gpu_box_bootstrap.md`](docs/gpu_box_bootstrap.md) | Fresh-machine setup + "where things stand" status log |
| [`hardware.md`](docs/hardware.md) | Training hardware specs and measured throughput |

## Roadmap

- **Distillation (H23)** — on-policy distillation from `gemma-4-E4B-it` (8B), in progress.
- **ORPO** (Week 3) — blocked on upstream `trl` support, tracked openly.
- **Imatrix-calibrated GGUF requantization** — should improve quality at aggressive quant
  levels beyond what's already shipped.
- **Abliteration research (H22)** — a real, pre-registered experiment on the relationship
  between refusal capability and judgment quality, not a "ship an uncensored model" feature.
  Eval design required before any model work.
- Real PCS (Persona Consistency Score) metric, voice/TTS feasibility, the full companion
  persona-card schema, mobile deployment, and a final open-source runnable app — see
  [`docs/research_questions.md`](docs/research_questions.md) for design notes on all of these.

## License

Code: Apache-2.0 (`LICENSE`). Benchmarks/data: CC-BY-4.0 (`LICENSE-DATA`).
