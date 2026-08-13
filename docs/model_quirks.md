# Model-loading quirks and environment findings

Discovered via real download + generation smoke tests on the GPU training box
(`docs/hardware.md`), one per bake-off candidate, before running the full bake-off for real.
Keep this updated whenever a new environment/model surprise turns up — the point is that a
fresh session (or a future week's re-test) doesn't have to rediscover any of this.

## Environment-wide fixes (already applied in code / required at runtime)

1. **`torch.bf16` doesn't exist.** `HFEngine`'s `dtype` param defaulted to `"bf16"`, but the
   real torch attribute is `bfloat16`. Fixed in `src/onebee/inference/engine.py` via a
   `_DTYPE_ALIASES` map (`bf16`→`bfloat16`, `fp16`→`float16`, `fp32`→`float32`, `half`→`float16`)
   so both shorthand and full names work. **If this ever resurfaces:** it means a new alias
   wasn't added to that map, not that the underlying bug came back.

2. **`torch_dtype` kwarg is deprecated in transformers 5.15.0** — use `dtype` instead.
   `AutoModelForImageTextToText.from_pretrained(..., dtype=...)`, not `torch_dtype=...`.
   Already fixed in `engine.py`.

3. **Multimodal models need `AutoModelForImageTextToText`, not `AutoModelForCausalLM`.**
   `AutoModelForCausalLM.from_pretrained` raises `ValueError: Unrecognized configuration class`
   for any VLM config (e.g. `SmolVLMConfig`). `HFEngine.load()` now tries
   `AutoModelForImageTextToText` first when `self._is_multimodal` is True.

4. **`TORCH_CUDNN_V8_API_DISABLED=1` is REQUIRED for Qwen3-VL on this GPU.** Without it, any
   Qwen3-VL forward pass (specifically the vision tower's `Conv3d` patch-embed) crashes with:
   ```
   RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed...
   CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
   ```
   This is a cuDNN9-vs-Blackwell(sm_120)-vs-conv3d incompatibility, not a torch/transformers
   version problem — setting `LD_LIBRARY_PATH` to prioritize torch's bundled cuDNN did **not**
   fix it; only disabling the cuDNN v8 API path did. **Always export
   `TORCH_CUDNN_V8_API_DISABLED=1` before running anything that touches Qwen3-VL on this
   hardware** (other candidates load fine without it, but setting it globally on this box is
   harmless and simplest — it was left set for all 4 candidate tests below).

5. **`num2words` is a real (missing) dependency for SmolVLM's processor.** Without it,
   `AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-2.2B-Instruct')` raises
   `ImportError: Package 'num2words' is required to run SmolVLM processor.` Added to the `gpu`
   extra in `pyproject.toml`. **This surfaced a worse bug**: `HFEngine.load()`'s exception
   handler around `AutoProcessor.from_pretrained` was silently swallowing this and falling back
   to text-only mode — i.e. a genuinely multimodal model would have silently lost vision
   support with no error, which is much worse than crashing. Fixed to `warnings.warn(...)` loud
   ly on any processor-load failure instead of failing silently. **If a future model hits a
   similar missing-dependency error, you'll now see a Python warning naming the exact exception
   instead of the model quietly running text-only.**

6. **`gpt-5.6-luna` (the OpenAI judge) rejects any non-default `temperature`.** Raises
   `400 Unsupported value: 'temperature' does not support 0 with this model. Only the default
   (1) value is supported.` `OpenAIJudge._request_json` now retries once without `temperature`
   on this specific error and remembers not to send it again for the rest of that judge
   instance's life (`self._temperature_unsupported`). **If a future judge model has the same
   restriction, this handles it automatically — no code change needed.**

7. **SmolVLM2's chat template silently drops plain-string message content.** The FIRST real
   bake-off run scored `smolvlm2-2.2b` a flat `0.0` on every text-only category
   (instruction/en_dialogue/ja_dialogue/structured_context) — not because the model answered
   badly, but because `apply_chat_template` handed it `content: "some text"` (a plain string,
   the normal shape for text-only messages) and SmolVLM's Jinja template only knows how to
   iterate typed content parts (`[{"type": "text", "text": ...}]`). The rendered prompt came out
   as `'<|im_start|>User: <end_of_utterance>\nAssistant:'` — **the user's question was silently
   missing entirely**, so the model was free-associating with no input, not underperforming.
   Fixed generically in `HFEngine.apply_chat_template` via
   `_normalize_content_for_multimodal_template()`, which converts plain-string content to
   list-of-parts form before handing off to ANY multimodal processor's chat template — this
   fixes it for every VLM with this template style, not just SmolVLM2. **This is the most
   important finding in this file**: always re-run the bake-off after this fix if it's ever
   reverted, and treat a suspiciously flat/zero score on one model as a wiring bug to
   investigate (check the actual rendered prompt via `engine.apply_chat_template(messages)`)
   before assuming the model is just bad.

## Per-candidate results (real download + generate, 2026-08-13)

All 4 confirmed: load successfully, correctly detected as multimodal
(`HFEngine._is_multimodal == True`), and answer real questions about the real committed fixture
images (`data/fixtures/bakeoff_images/`) correctly.

| Candidate | HF repo | Processor class | Quirks | Sample result | tok/s (unoptimized) |
|---|---|---|---|---|---|
| `lfm2.5-vl-1.6b` | `LiquidAI/LFM2.5-VL-1.6B` | `Lfm2VlProcessor` | None beyond the environment-wide fixes above | "How many candles?" → `3` (correct, cake has 3) | ~2.8–7.7 |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | `Qwen3VLProcessor` | **Requires `TORCH_CUDNN_V8_API_DISABLED=1`** (see above) or every image forward pass crashes | "What color is the ribbon?" → `yellow` (correct, gold bow) | not yet measured (crashed on first attempt, measured after fix) |
| `gemma4-e2b` | `google/gemma-4-E2B-it` | `Gemma4Processor` | None beyond the environment-wide fixes | "What is this a picture of?" → `A brown teddy bear.` (correct) | ~4.8 |
| `smolvlm2-2.2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `SmolVLMProcessor` | **Requires `num2words`** (see above); benign warning `Kwargs passed to processor.__call__...` — no crash, safe to ignore | "How many flowers?" → `8` | ~4.1 |

All tok/s numbers above are from a single unoptimized `max_new_tokens=48` generation, not the
real latency bench (`inference/bench.py`, still pending) — treat as rough signal only, not a
result.

## Day 3 findings: System D (memory-augmented) wiring

8. **`MemoryStore.search()`'s FTS5 query crashed on real questions.** Fixed in
   `src/onebee/memory/store.py` via `_sanitize_fts_query()` — see that commit's message for
   detail. Any natural-language query with `?`, `'`, `-`, `:` would crash before this fix.

9. **A flat single-message prompt (persona + memories + question all as one "user" turn) gets
   ignored by the raw model — it needs real `system`/`user` role separation.** First System D
   smoke test: the exact correct memory ("Alice is a designer at Stripe.") was retrieved at
   rank 1 and correctly present in the assembled context string, but `gemma4-e2b` still
   answered "I am a large language model, developed by Google DeepMind" — it wasn't reading
   the context as something to *use*, just as more text. Fixed by sending the persona+memories
   block as a `system` message and the actual question as a separate `user` message (see
   `run_system_d.py`'s `response_fn`).

10. **Even with proper system/user roles, the raw (non-SFT) model conflates "you" in a
    companion-framed question with itself, not the user.** "What company do you work for?"
    got answered as if asking about the AI ("I am a large language model...") rather than the
    user, even with the correct fact in context — because nothing tells the model that
    second-person companion-style questions are usually the user asking to be told about
    *themselves*. Mitigated (not fully solved) with an explicit system-prompt instruction
    ("questions addressed to 'you' are usually asking you to recall something about THEM, not
    about yourself") — this measurably improved behavior (correct recall when relevant memory
    was retrieved, honest abstention rather than confabulation when it wasn't) but is not
    perfect. **This is treated as a real, reportable finding, not chased to perfection** — it's
    exactly the kind of raw-model limitation Day 4's memory-aware SFT (H5) is hypothesized to
    fix. Further prompt-format tuning belongs to the Week 2+ context-format ablation (H19), not
    this pass.

## Day 4 findings: LoRA SFT training on real trl/transformers versions

11. **`transformers.TrainingArguments` dropped `warmup_ratio` entirely** in this environment's
    version (5.15.0) — raises `TypeError: unexpected keyword argument 'warmup_ratio'`. Fixed in
    `build_training_arguments()`: tries `warmup_ratio` first, falls back to computing an
    equivalent `warmup_steps` from `num_training_examples` on `TypeError`.

12. **Modern `trl.SFTTrainer` restructured its API**: (a) `tokenizer=` kwarg renamed to
    `processing_class=`; (b) SFT-specific fields (`max_seq_length`→`max_length`, `packing`,
    `neftune_noise_alpha`) moved from `SFTTrainer.__init__` kwargs into `trl.SFTConfig` (a
    `TrainingArguments` subclass) passed as `args=`; (c) `train_dataset`/`eval_dataset` must be
    a real `datasets.Dataset`, not a plain list of dicts. All three fixed in `run_sft`'s default
    `trainer_factory` — the function's own kwargs contract (what injectable-fake tests assert
    on) is unchanged, only the real-trl adaptation inside the default factory changed. See the
    "Fix SFTTrainer TypeError" and "Convert train/eval datasets" commits.

13. **Saving a merged multimodal SFT checkpoint via `tokenizer.save_pretrained()` alone silently
    drops vision capability.** The merged model directory was missing `preprocessor_config.json`
    — `HFEngine` loading it fell back to text-only with a warning. Fixed: also save the full
    `AutoProcessor` (not just the tokenizer) when the base model has one. **If a future merged
    checkpoint loads as text-only unexpectedly, check for this exact regression first** —
    `engine._is_multimodal` should be `True` after loading a merged multimodal SFT checkpoint.

14. **A LoRA-tuned model is sensitive to *exact* prompt formatting matching its training
    distribution — this is expected behavior, not a bug, but easy to mistake for one.** The
    Day-4 SFT adapter answered a companion-framed question correctly when tested with the exact
    `ContextBuilder`-produced system prompt (same shape as training data), but answered "I am a
    large language model, trained by Google" when tested with a hand-written system prompt
    conveying the same *content* in different *wording/structure*. Always sanity-test a trained
    checkpoint using the exact same context-assembly code path it saw during training/eval, not
    an ad-hoc paraphrase — a wrong-looking result may just be a format mismatch.

15. **`trl.ORPOTrainer`/`ORPOConfig` do not exist in this environment's trl (1.10.0) at all** —
    `ImportError: cannot import name 'ORPOConfig' from 'trl'`. This isn't a rename like the
    earlier SFT/DPO API breaks; ORPO support was dropped entirely from this trl version.
    Available preference/RL trainers in this environment: `DPOConfig`/`DPOTrainer`,
    `GRPOConfig`/`GRPOTrainer`, `KTOConfig`/`KTOTrainer`, `RLOOConfig`/`RLOOTrainer`,
    `RewardConfig`/`RewardTrainer`, `DistillationConfig`/`DistillationTrainer`. **H3.1 ("does
    ORPO match DPO at lower compute") cannot be tested without either pinning an older trl
    version (risk: could break the already-working SFT/DPO pipeline, which depends on this
    exact trl version's API) or hand-implementing the ORPO loss.** Neither was attempted given
    time budget — documented as a real blocker, not silently skipped. If ORPO is needed later,
    check trl's current version support before assuming `ORPOTrainer` exists.

## How to re-run these smoke tests

```bash
cd /root/onebee-gf
export PATH=$HOME/.local/bin:$PATH
source /root/.env.onebee && export HF_TOKEN OPENAI_API_KEY JUDGE_MODEL
export TORCH_CUDNN_V8_API_DISABLED=1   # harmless for non-Qwen3-VL models, required for it
uv run python -c "
from onebee.inference.engine import HFEngine, GenerationConfig
engine = HFEngine('<hf-repo-id>')
engine.load()
r = engine.generate([{'role':'user','content':[
    {'type':'image','image':'data/fixtures/bakeoff_images/<file>.png'},
    {'type':'text','text':'<question>'}
]}], GenerationConfig(max_new_tokens=48))
print(r.text)
"
```
