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
    check trl's current version support before assuming `ORPOTrainer` exists. **Moved to Week
    3's scope (2026-08-15)** — Week 2 is now considered closed on DPO alone (H6-H7 done at
    proper scale, see `docs/proper_scale_results.md`), with distillation (H8-H9) as the
    remaining Week 2 item; ORPO deferred rather than blocking Week 2's close-out.

16. **`generate_sft_data.py`'s response-text dedup silently collapsed the abstention and
    irrelevant-retrieval training categories down to a single surviving example each,
    regardless of how many were generated.** These two categories use a FIXED template
    response by design (e.g. `"I don't think you've told me that — I don't want to guess."`)
    repeated across many different personas/contexts — that repetition IS the intended
    training signal (learn to always give this response when you don't know, in any context).
    But the post-generation filter deduped by exact assistant-response text, which is correct
    for `memory_relevant` examples (real teacher text, naturally unique) but wrong for these:
    after the first abstention example, every subsequent one looked like an exact duplicate
    and got silently dropped. Confirmed via the datasheet output: `data/sft/v0/DATASHEET.md`
    and `data/sft/v1/DATASHEET.md` both show `"abstention": 1, "irrelevant_retrieval": 1"`
    despite the code intending ~10%/~15% of the dataset. Caught because System E's real-eval
    UAR (unanswerable-abstention rate) regressed sharply between v0 (33.75%) and the properly-
    scaled v1 run (16.25%) even though the v1 SFT dataset was 10x larger and used the same
    generation code — the single surviving abstention example became proportionally 10x more
    diluted in the larger dataset, which is exactly the mechanism that explains the gap (v0's
    training signal was already broken, just less severely so at v0's smaller scale). Manually
    inspected real failing eval responses (fabricated "childhood nickname: Panda", a retrieved-
    but-wrong "Cape Town" vacation memory misapplied to a distractor-style unanswerable
    question) confirmed this wasn't a scoring/wiring bug — see `docs/proper_scale_results.md`.
    **Fix:** only dedup `memory_relevant` examples by response text; dedup
    `abstention`/`irrelevant_retrieval` examples by the full `(system, user, response)` tuple
    instead, so only true exact repeats (identical context AND response) get dropped.

17. **After fixing #16, `_ABSTENTION_PHRASES` in `src/onebee/evaluation/graders/rule.py`
    (the rule-based `detect_abstention` used to compute UAR) didn't include the literal
    template strings `generate_sft_data.py` trains the model to produce** (e.g. `"I don't
    think you've told me that — I don't want to guess."`), so a model that had correctly
    learned to reproduce that exact intended phrasing was scored as NOT abstaining. This
    masked a second, independent bug that only became visible after fixing #16: with the real
    abstention signal restored, the model started reproducing the literal training phrase
    verbatim far more often, which the old phrase list didn't recognize, making UAR look like
    it had gotten *worse* (2.5%) right after the fix that should have improved it. Rescoring
    the same saved responses with the fixed phrase list revealed the true number: **UAR 96.25%**
    — but also, checked in the same pass, a real new problem: 69.2% false-abstention rate on
    *answerable* probes (up from the broken run's 9.9%). The dedup fix (#16) worked as intended,
    but the abstention/irrelevant-retrieval training ratio it restored (~17% of the dataset)
    turned out to be enough to make the model reflexively hedge on a large fraction of
    perfectly answerable questions too — a genuine precision/recall tradeoff exposed by fixing
    the measurement, not a new training bug. See `docs/proper_scale_results.md`'s fix-
    verification section for the full before/after numbers. **Lesson:** when training data and
    evaluation code are generated/maintained somewhat independently (different scripts, written
    at different times), a literal-string rule-based grader silently drifts out of sync with
    whatever phrasing the training data actually teaches — worth periodically checking that a
    correctly-behaving model would actually score well under the current metric code, not just
    that a bad model scores badly.

18. **`convert_hf_to_gguf.py`'s `--mmproj` flag exports ONLY the vision projector, not the main
    language model** — despite it looking like a natural "also include multimodal" flag. Ran
    once with `--mmproj` expecting a combined output and got a suspiciously small 985MB GGUF
    (should be ~9GB for a 4.6B-param model); the vision-only tensors (`v.blk.*`) in the log
    were the giveaway. **Fix:** two separate conversion runs are required — one without
    `--mmproj` for the main model, one with `--mmproj` for the projector (which the tool
    auto-prefixes `mmproj-` unless you pick your own name, as I did).

19. **`convert_hf_to_gguf.py`'s own bundled transformers version (4.57.6, pinned in
    `requirements/requirements-convert_hf_to_gguf.txt`) is older than the training
    environment's (5.15.0), and the two disagree on `tokenizer_config.json`'s
    `extra_special_tokens` format** — 5.15.0 writes it as a bare list (`["<|video|>"]`), 4.57.6's
    `_set_model_specific_special_tokens` expects a dict (`{"video_token": "<|video|>"}`) and
    crashes with `AttributeError: 'list' object has no attribute 'keys'` otherwise. **Fix:**
    patch a COPY of the checkpoint's `tokenizer_config.json` (never the canonical
    HF-Hub-downloaded one) before conversion — set `extra_special_tokens` to a dict, using the
    same `<token_name>_token` key convention already visible in the sibling
    `model_specific_special_tokens` field (e.g. `image_token`, `audio_token`) to infer the
    missing key name (`video_token` for `<|video|>`). This will recur for any future GGUF
    conversion pass unless llama.cpp's pinned transformers version is bumped upstream — worth
    checking `requirements-convert_hf_to_gguf.txt` on future attempts to see if it's been fixed.

20. **`llama-cli`'s conversation mode (`-cnv`, auto-enabled whenever a chat template is
    present — which ours always is) also auto-enables interactive mode**, which then blocks
    waiting on stdin after the first response. Piping through `nohup ... &` detaches/closes
    stdin, so the process doesn't error — it just spins at ~100% CPU forever printing an empty
    `>` prompt in a tight loop, reading EOF repeatedly. This looked exactly like "quantization
    made inference catastrophically slow" for the first ~15 minutes of testing (10+ min of
    real CPU time burned on what should be a 30-second generation) before the actual cause
    (wrong CLI flag, not a slow model) was found. **Fix:** use `--single-turn` (`-st`) for any
    one-shot/scripted generation test — it runs the conversation for exactly one turn using
    `--prompt` as the first turn, then exits cleanly, without needing an interactive terminal.
    **Lesson generalized:** when a background/scripted CLI invocation of an interactive tool
    hangs with no output, check whether it silently dropped into an interactive prompt loop on
    a closed stdin before assuming the underlying computation itself is slow — burned real time
    on this exact confusion.

21. **`llama-mtmd-cli` crashes (`std::runtime_error: this custom template is not supported`,
    `terminate called after throwing`) on our model's chat template** unless run with
    `--jinja`. The default (non-jinja) template handling in `llama-mtmd-cli` apparently can't
    parse the full complexity of this checkpoint's real chat template (loop-based
    tool-call/channel/turn logic — see the template excerpt in `docs/quantization_results.md`).
    **Fix:** always pass `--jinja` when using `llama-mtmd-cli` (or presumably `llama-server`)
    against this model family. Once added, vision inference worked correctly and accurately
    (correctly described a real fixture image) — the crash was a CLI templating issue, not a
    real multimodal-capability loss from quantization.

22. **`trl.DistillationConfig`'s `teacher_model_name_or_path` field is metadata only — it does
    NOT cause the trainer to actually load a teacher.** `DistillationTrainer` needs the teacher
    passed as its own separate `teacher_model` constructor argument (a string ID or an
    instantiated model, per its docstring). Passing `teacher_model_name_or_path` on the config
    alone (a very natural mistake — that's exactly the field name you'd expect to control this)
    leaves `trainer.teacher_model` as `None`, which doesn't error until the very first real
    training step: `AttributeError: 'NoneType' object has no attribute 'eval'`, deep inside
    `DistillationTrainer.compute_loss`. **A `--dry-run` (build the trainer, skip `.train()`)
    does NOT catch this** — teacher loading is apparently lazy/step-triggered, not done at
    trainer construction, so the bug only surfaced on a real (paid) training run. **Fix:** pass
    `teacher_model=<model id>` explicitly to the `DistillationTrainer` constructor, in addition
    to (or instead of) setting it on the config. **Lesson:** for any trl trainer with both a
    config field and a same-named-ish constructor argument, verify which one actually does the
    work before trusting a dry-run to have validated it — this project's SFT/DPO dry-runs never
    needed this distinction since neither has an analogous "external second model" concept.

## How to re-run these smoke tests

```bash
cd /root/small-mind-companion
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
