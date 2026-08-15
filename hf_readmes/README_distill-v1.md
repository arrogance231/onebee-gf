---
license: gemma
base_model: google/gemma-4-E2B-it
tags:
  - companion
  - multimodal
  - lora
  - distillation
---

# onebee-gf-distill-v1

**Table of contents:** [Overview](#onebee-gf-distill-v1) · [H23 result](#h23-result) ·
[Other checkpoints](#other-checkpoints-from-this-project) · [License](#license)

Part of **[small-mind-companion](https://github.com/arrogance231/small-mind-companion)** — an
open-source research project exploring how much apparent capability a small (~2-4B parameter),
vision-capable language model can recover through post-training, external memory, and
retrieval, rather than raw parameter scale. The project treats "the number looked good" as a
signal to investigate, not a result to trust — see the
[README's Engineering highlights](https://github.com/arrogance231/small-mind-companion#engineering-highlights)
for real bugs found, root-caused, and fixed along the way (not just "it worked").

**Current best checkpoint overall.** SFT + DPO (`dpo-v1-scale`) + on-policy distillation
(H23) from a larger local teacher (`google/gemma-4-E4B-it`, 8B). This is the checkpoint the
project's headline distillation results are actually about — merged LoRA weights, ready to
load directly with `transformers`.

## H23 result

On-policy distillation (student generates its own completions, matched to the teacher's
distribution via generalized JSD) on top of `dpo-v1-scale`, improved response quality without
degrading persona consistency:

| System | pra_lenient | uar |
|---|---|---|
| dpo-v1-scale (pre-distillation) | 15.30% | 70.0% |
| **distill-v1 (this checkpoint)** | **18.59%** | **71.25%** |

Pairwise persona-consistency judging favored this checkpoint 38.1% vs 30.5% against its
pre-distillation predecessor (33 ties) — despite the teacher not being persona-tuned, a real
risk the hypothesis flagged going in. A second, independent no-API stylometric-consistency
signal (writing-style self-consistency, not semantic content) also came out slightly higher
post-distillation (0.524 vs 0.509), agreeing with the judge-based result.

**Full methodology, training-time anomalies (and why they didn't predict the real-eval
outcome), and honest limitations**: see
[`docs/distillation_results.md`](https://github.com/arrogance231/small-mind-companion/blob/main/docs/distillation_results.md)
in the project repo. Single seed, single data scale (2008 prompts, 125 steps) — not yet a
multi-seed-confirmed result.

## Other checkpoints from this project

| Repo | Description |
|---|---|
| [onebee-gf-sft-v0](https://huggingface.co/arrochi112/onebee-gf-sft-v0) | Day 4 v0 SFT |
| [onebee-gf-sft-v1](https://huggingface.co/arrochi112/onebee-gf-sft-v1) | Proper-scale SFT |
| [onebee-gf-dpo-v0](https://huggingface.co/arrochi112/onebee-gf-dpo-v0) | Week 2 DPO v0 |
| [onebee-gf-dpo-v1-4epoch](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-4epoch) | DPO overfitting experiment |
| [onebee-gf-dpo-v1-scale](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale) | Proper-scale DPO, pre-distillation |
| [onebee-gf-distill-v1](https://huggingface.co/arrochi112/onebee-gf-distill-v1) | **This repo** — SFT+DPO+distillation, current best overall |
| [onebee-gf-dpo-v1-scale-gguf](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale-gguf) | GGUF quantizations (pre-distillation checkpoint) |

## License

Inherits Gemma's license terms from the base model (`google/gemma-4-E2B-it`).
