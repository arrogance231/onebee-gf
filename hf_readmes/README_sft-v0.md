---
license: gemma
base_model: google/gemma-4-E2B-it
tags:
  - companion
  - multimodal
  - lora
---

# onebee-gf-sft-v0

Part of **[small-mind-companion](https://github.com/arrogance231/small-mind-companion)** — an
open-source research project exploring how much apparent capability a small (~2-4B parameter),
vision-capable language model can recover through post-training, external memory, and
retrieval, rather than raw parameter scale.

Day 4 v0 LoRA SFT checkpoint (202 train examples) — the small proof-of-concept SFT run, superseded by `sft-v1`.

**Full results, methodology, and honest limitations**: see
[`docs/day4_sft_results.md`](https://github.com/arrogance231/small-mind-companion/blob/main/docs/day4_sft_results.md) in the project
repo. This project reports negative/inconclusive results as honestly as positive ones — read
the docs before assuming any number here is a clean win.

## Other checkpoints from this project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arrochi112/onebee-gf-sft-v0) | Day 4 v0 SFT |
| [onebee-gf-sft-v1](https://huggingface.co/arrochi112/onebee-gf-sft-v1) | Proper-scale SFT — current best SFT |
| [onebee-gf-dpo-v0](https://huggingface.co/arrochi112/onebee-gf-dpo-v0) | Week 2 DPO v0 |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale) | Proper-scale DPO — current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations of the current-best checkpoint |

## License

Inherits Gemma's license terms from the base model (`google/gemma-4-E2B-it`).

