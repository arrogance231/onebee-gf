---
license: gemma
base_model: google/gemma-4-E2B-it
tags:
  - gguf
  - llama.cpp
  - multimodal
  - companion
  - quantized
---

# onebee-gf-dpo-v1-scale (GGUF)

**Table of contents:** [Overview](#onebee-gf-dpo-v1-scale-gguf) ·
[Files](#files) · [Benchmarks](#real-benchmark-numbers-cpu-30-threads-llama-bench) ·
[Usage](#usage) · [Limitations](#known-limitations) · [License](#license)

GGUF conversion and quantization of the current-best checkpoint from
**[small-mind-companion](https://github.com/arrogance231/small-mind-companion)** — an
open-source research project exploring how much apparent capability a small (~2-4B parameter),
vision-capable language model can recover through post-training, external memory, and
retrieval, rather than raw parameter scale.

This checkpoint is a `gemma-4-E2B-it` base, LoRA-fine-tuned via memory-aware SFT then DPO on a
persona-consistency preference objective (223→2049 preference pairs across two scaling passes),
at [`arrochi112/onebee-gf-dpo-v1-scale`](https://huggingface.co/arrochi112/onebee-gf-dpo-v1-scale)
in `safetensors` form. This repo is the same model, converted to GGUF for `llama.cpp`-based
local/on-device inference.

**Full results, methodology, and honest limitations**: see the project's
[`docs/`](https://github.com/arrogance231/small-mind-companion/tree/main/docs) directory,
especially
[`proper_scale_results.md`](https://github.com/arrogance231/small-mind-companion/blob/main/docs/proper_scale_results.md)
(training results) and
[`quantization_results.md`](https://github.com/arrogance231/small-mind-companion/blob/main/docs/quantization_results.md)
(this conversion's real benchmark numbers, generation-quality checks, and bugs found along the
way). This project reports negative/inconclusive results as honestly as positive ones — read
the docs before assuming any number here is a clean win.

## Files

| File | Quant | Size | Notes |
|---|---|---|---|
| `onebee-dpo-v1-scale-f16.gguf` | F16 | 8.64 GiB | Full precision, reference quality |
| `onebee-dpo-v1-scale-Q8_0.gguf` | Q8_0 | 4.61 GiB | Near-lossless |
| `onebee-dpo-v1-scale-Q6_K.gguf` | Q6_K | 3.57 GiB | |
| `onebee-dpo-v1-scale-Q5_K_M.gguf` | Q5_K_M | 3.37 GiB | |
| `onebee-dpo-v1-scale-Q5_K_S.gguf` | Q5_K_S | 3.34 GiB | |
| `onebee-dpo-v1-scale-Q5_0.gguf` | Q5_0 | 3.34 GiB | |
| `onebee-dpo-v1-scale-Q4_K_M.gguf` | Q4_K_M | 3.18 GiB | **Recommended default** — best size/quality tradeoff |
| `onebee-dpo-v1-scale-Q4_K_S.gguf` | Q4_K_S | 3.12 GiB | |
| `onebee-dpo-v1-scale-Q4_0.gguf` | Q4_0 | 3.12 GiB | |
| `onebee-dpo-v1-scale-Q3_K_L.gguf` | Q3_K_L | 3.05 GiB | |
| `onebee-dpo-v1-scale-Q3_K_M.gguf` | Q3_K_M | 2.97 GiB | |
| `onebee-dpo-v1-scale-Q3_K_S.gguf` | Q3_K_S | 2.89 GiB | |
| `onebee-dpo-v1-scale-Q2_K.gguf` | Q2_K | 2.78 GiB | Smallest, real quality loss expected |
| `mmproj-onebee-dpo-v1-scale-f16.gguf` | F16 | 940 MiB | Vision projector — needed for image input, use with any of the above |

No importance-matrix (imatrix) calibration was used for these quants — a real next step, not
yet done (see the project's roadmap doc).

## Real benchmark numbers (CPU, 30 threads, `llama-bench`)

| Quant | Size | Prompt (pp512) | Generation (tg128) |
|---|---|---|---|
| F16 | 8.62 GiB | 585.07 ± 0.44 t/s | 26.15 ± 0.28 t/s |
| Q8_0 | 4.59 GiB | 492.33 ± 1.21 t/s | 43.07 ± 0.24 t/s |
| Q4_K_M | 3.17 GiB | 633.00 ± 1.22 t/s | 58.00 ± 0.51 t/s |

## Usage

Text:
```bash
llama-cli -m onebee-dpo-v1-scale-Q4_K_M.gguf -p "Hello!" -st
```

Vision (needs `--jinja` — this model's chat template isn't supported by the default non-jinja
parser):
```bash
llama-mtmd-cli -m onebee-dpo-v1-scale-Q4_K_M.gguf \
  --mmproj mmproj-onebee-dpo-v1-scale-f16.gguf \
  --image your_image.png -p "What is in this image?" --jinja
```

## Known limitations

- No accuracy/quality regression measured against the project's own PMB-v0 eval harness at
  each quant level — verified "coherent and on-topic" via real generation tests, not
  "measurably as accurate as F16." See `docs/quantization_results.md` for the exact checks run.
- This is a research checkpoint from an active, in-progress open-source project — expect real,
  documented limitations (see the linked docs) rather than a polished consumer product.

## License

Inherits Gemma's license terms from the base model (`google/gemma-4-E2B-it`).
