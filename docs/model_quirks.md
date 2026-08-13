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
