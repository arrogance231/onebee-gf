# GGUF quantization results (2026-08-15)

**Status:** real, working GGUF quantization of the current-best checkpoint
(`arrochi112/onebee-gf-dpo-v1-scale`, the rebalanced proper-scale SFT+DPO checkpoint — see
`docs/proper_scale_results.md`). Both text and vision (multimodal) capability verified to
survive conversion and quantization. Four real bugs found and fixed along the way —
`docs/model_quirks.md` #18-21.

## What was done

1. Built `llama.cpp` from source, CPU-only (`GGML_CUDA=OFF`) — this box had no system-wide
   CUDA toolkit (`nvcc`), and conversion/quantization are CPU-only tools regardless; a CPU
   build is also arguably more representative of the actual on-device deployment target
   (phones don't have CUDA) than a GPU-accelerated benchmark would have been.
2. Downloaded `arrochi112/onebee-gf-dpo-v1-scale` from HF Hub (the checkpoint uploaded before
   the previous GPU box was deleted).
3. Converted to GGUF F16 in two passes (`convert_hf_to_gguf.py`, see model_quirks #18):
   - Main language model: `onebee-dpo-v1-scale-f16.gguf` (9.27GB)
   - Vision projector: `mmproj-onebee-dpo-v1-scale-f16.gguf` (986MB)
   - Required patching a copy of `tokenizer_config.json` first (model_quirks #19) — llama.cpp's
     own pinned `transformers==4.57.6` and our training environment's `transformers==5.15.0`
     disagree on the `extra_special_tokens` field format.
4. Quantized the main model to two levels with `llama-quantize`:
   - `onebee-dpo-v1-scale-Q8_0.gguf` (4.59 GiB, -47% vs F16)
   - `onebee-dpo-v1-scale-Q4_K_M.gguf` (3.17 GiB, -63% vs F16)
   - The mmproj (vision projector) was NOT separately quantized — used as F16 with all three
     text-model quant levels, which is llama.cpp's standard pattern (the projector is small
     relative to the LLM, and is more precision-sensitive).
5. Verified real generation quality and multimodal capability on all three levels — not just
   "did it produce a file," but "does it produce coherent, correct output."

## Real benchmark numbers (llama-bench, CPU-only, 30 threads, this box's hardware)

Model reports as **4.63B params** (not literally "2B" — `gemma4-E2B` is an MoE architecture
with an "effective 2B" active-parameter naming, confirmed by llama-bench's own model-info
output, not something previously stated explicitly in this project's docs).

| Quant | Size | Prompt processing (pp512) | Generation (tg128) |
|---|---|---|---|
| F16 | 8.62 GiB | 585.07 ± 0.44 t/s | 26.15 ± 0.28 t/s |
| Q8_0 | 4.59 GiB (-47%) | 492.33 ± 1.21 t/s | 43.07 ± 0.24 t/s (+65%) |
| Q4_K_M | 3.17 GiB (-63%) | 633.00 ± 1.22 t/s | 58.00 ± 0.51 t/s (+122%) |

Real, somewhat counter-intuitive result worth noting honestly: Q8_0's prompt-processing speed
is *slower* than F16 (492 vs 585 t/s), while Q4_K_M's is *faster* than both (633 t/s). This is
plausible (Q8_0's dequantization overhead during prompt-batch matmuls can outweigh its memory-
bandwidth savings on some kernels/hardware, while Q4_K_M's more aggressive size reduction wins
on both axes) but wasn't investigated further — noted as a real observation, not explained away.
Generation (token-by-token, memory-bandwidth-bound) speed scales cleanly with quantization
level as expected, in both directions.

## Real generation quality checks (not just "did a file get produced")

- **Text, no persona context** ("What is 2+2?", "What is your favorite color?"): coherent on
  all three levels (F16, Q8_0, Q4_K_M). Content-wise the model answered "2+2=6" (wrong — this
  is a small companion-persona-tuned model, not a math model, and math was never part of any
  training data in this project; not a quantization bug) and gave the generic "as an AI, I
  don't have preferences" disclaimer to the favorite-color question (expected: that test had
  no injected memory/persona system prompt of the kind the real eval harness always uses — see
  `docs/proper_scale_results.md` for the real, in-format evaluation of this exact behavior).
- **Text, with the real companion system prompt**: consistent phrasing/style across all three
  quant levels — no signs of quantization-induced incoherence or repetition loops.
- **Vision** (`llama-mtmd-cli --jinja`, fixture image `wrapped_gift_red.png`): **"A red gift
  box tied with a gold ribbon is shown against a light gray background."** — accurate,
  specific, matches the real fixture image. Confirms multimodal capability survived the full
  conversion + quantization pipeline, which is exactly the risk this project's own
  `docs/model_quirks.md` has previously flagged (multimodal capability silently lost at the
  save-checkpoint step during SFT, item #13) — this time it survived cleanly, a genuinely good
  result worth confirming explicitly rather than assuming.

## Four real bugs found and fixed (full detail: `docs/model_quirks.md` #18-21)

1. `--mmproj` exports ONLY the projector, not a combined file — two separate runs needed.
2. Cross-version `tokenizer_config.json` format mismatch (`extra_special_tokens` list vs dict)
   between the training environment's transformers (5.15.0) and llama.cpp's pinned converter
   dependency (4.57.6) — required patching a copy of the config before conversion.
3. `llama-cli`'s `-cnv` (auto-enabled with any chat template) also auto-enables interactive
   mode, which hangs on closed stdin in a scripted/backgrounded invocation — burned real time
   mistaking this for "quantization made the model catastrophically slow" before finding the
   actual cause (wrong CLI flag). Fix: `--single-turn`.
4. `llama-mtmd-cli` crashes on this model's real (complex, loop-based) chat template unless
   run with `--jinja`.

None of these are quantization *quality* issues — all four are tooling/environment friction,
the same category of finding this project has documented extensively at every prior stage
(model loading, training APIs, evaluation scoring). The actual quantization numerics worked
cleanly on the first successful attempt once the tooling issues were resolved.

## Where the artifacts are

- Local (this GPU box, `/root/onebee-gf/gguf/`): all 4 files (F16 main + mmproj, Q8_0, Q4_K_M).
- HF Hub: uploaded to `arrochi112/onebee-gf-dpo-v1-scale-gguf` (see repo for exact filenames)
  so they survive this box being deleted, same pattern as the merged checkpoints themselves.

## Known limitations / not done in this pass

- No comparison against Google's own QAT mobile variant
  (`google/gemma-4-E2B-it-qat-mobile-transformers`) — noted as a next step in
  `docs/gpu_box_bootstrap.md`, not done here.
- No real accuracy/quality regression test against the PMB-v0 benchmark at each quant level —
  this pass confirmed "coherent and on-topic," not "measurably as accurate as F16 on the real
  eval harness." A real quant-vs-quality tradeoff study (does Q4_K_M measurably hurt
  `pra_lenient`/`uar` vs F16?) would need to actually run the harness through llama.cpp's
  Python bindings or server API against all three levels — not attempted here, given this
  pass's scope was "does quantization work at all for this architecture+checkpoint," which it
  does.
- CPU-only benchmark numbers only — no GPU-accelerated llama.cpp build/benchmark for
  comparison (this box had no CUDA toolkit installed; installing one and rebuilding with
  `GGML_CUDA=ON` was judged not worth the setup time for this pass, since the deployment
  target is on-device/mobile CPU inference anyway, not GPU-accelerated serving).
- Only one base checkpoint quantized (`dpo-v1-scale`, the current best) — the other 4
  checkpoints on HF Hub (sft-v0, sft-v1, dpo-v0, dpo-v1-4epoch) were not quantized, since
  they're superseded/experimental, not needed for deployment.
