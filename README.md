# onebee-gf

**Stretching a 1B parameter LLM through post-training and external cognitive architecture.**

How much apparent capability can be recovered from a ~1B parameter language model through
post-training, external memory, retrieval, state modeling, distillation, and inference-time
cognitive architecture — and how much of that survives quantization and runs locally on a
smartphone? This project builds a ~1B parameter conversational model augmented with an
external memory system (short-term, episodic, and semantic tiers), hybrid dense+BM25
retrieval, a token-budgeted context builder, and a LoRA/DPO post-training pipeline, and
evaluates it against an adversarial personalized-memory benchmark (PMB) with abstention and
contradiction traps — not just recall.

See [`docs/research_questions.md`](docs/research_questions.md) for the full research question
hierarchy and hypotheses this project is testing.

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
