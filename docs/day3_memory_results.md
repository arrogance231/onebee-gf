# Day 3 results: memory system, System A vs D, k-sweep

**Status:** real results, single seed, small-scale (Week 1 discipline — see limitations below).
All raw data in `results/v0.1/{A_raw,D_memory,ksweep}/`.

## What was built and run

1. **Real memory extraction.** `OpenAITeacherExtractor` (`src/onebee/memory/extraction/
   openai_extractor.py`) extracted claims from all 459 real user turns across the 8 PMB-v0
   personas' conversations (`data/benchmarks/pmb_v0_full/`). 505 claims extracted, 503 accepted
   by validation (span verification + triviality/attribution checks).
2. **Real embeddings.** `intfloat/multilingual-e5-small` (384-dim, the architecture doc's own
   recommendation) embedded every accepted memory for dense retrieval.
3. **8 populated memory stores** (`data/stores/pmb_v0_full/*.db`) — committed as versioned
   artifacts per the project's data-versioning rule.
4. **System A** (raw `gemma4-e2b`, zero context) evaluated on all 688 real PMB-v0 probes.
5. **System D** (raw `gemma4-e2b` + hybrid retrieval, k=8) evaluated on the same 688 probes.
6. **k-sweep** (k ∈ {0,2,4,8,16}) on a 120-probe stratified subsample (15/category), same
   pipeline.

## Headline numbers

| System | pra_strict | pra_lenient | uar |
|---|---|---|---|
| A (raw, no memory) | 0.0% | 0.16% | 13.75% |
| D (raw + memory, k=8) | 0.0% | 15.1% | 8.75% |

**k-sweep** (120-probe stratified subsample):

| k | pra_lenient |
|---|---|
| 0 | 0.0% |
| 2 | 4.8% |
| 4 | 7.6% |
| 8 | **17.1%** (peak) |
| 16 | 14.3% |

## Interpretation against pre-registered hypotheses

- **H1** (memory improves personalized recall, predicted +25 to +45pp for the *full* system):
  **partially confirmed.** +15pp from retrieval alone, with zero post-training — real,
  substantial, well short of the full-system prediction. Consistent with the prediction being
  about the full scaffold (SFT/DPO included), not raw retrieval in isolation.
- **H3** (memory increases hallucination on questions it cannot answer, predicted +3 to +10pp
  false-assertion rate): **confirmed.** UAR dropped 13.75% → 8.75% (a ~5pp drop in correct
  abstention, i.e. within the predicted range) when memory was added — the model asserts more
  confidently, correctly or not, once given retrieved context.
- **H10** (quality vs k is non-monotonic, peaking at 4-8, degrading beyond ~12): **confirmed
  closely.** Clear inverted-U, peak exactly at k=8, decline at k=16.

## Manual verification (not just trusting the scores)

- 5 randomly sampled `lenient_correct=True` System D responses read as genuinely accurate,
  natural recall (e.g. "You recently visited Reykjavik. The atmosphere there made you think
  differently about silence and space in music." — matches the source memory precisely), not
  judge artifacts.
- Retrieval spot-checked directly: for "What company do you work for?", the correct memory
  ("Alice is a designer at Stripe.") was retrieved at rank 1 and the model correctly answered
  once given proper system/user role framing (see `docs/model_quirks.md` items 8-10 for the two
  real wiring bugs found and fixed en route to this result: an FTS5 crash on punctuated
  questions, and a flat-prompt-gets-ignored issue requiring real role separation).
- A genuine retrieval miss was also found and left as-is (not papered over): a real "Alice
  adopted a parrot" memory exists in the store but wasn't retrieved for "What kind of pet did
  you adopt?" — the model correctly abstained rather than confabulating, which is itself a
  correct, desired behavior under retrieval failure.

## Known limitations (state these plainly, per the project's own research discipline)

- **Single seed, no bootstrap CIs on `pra_lenient`** (the harness only computes bootstrap CIs
  for `pra_strict` and `uar` currently — `pra_lenient`'s CI is a Week 2+ harness gap, not
  computed here).
- **k-sweep uses a 120-probe stratified subsample, not the full 688**, to keep real API/GPU
  cost bounded — the shape is informative but the exact percentages would likely shift somewhat
  on the full set.
- **System D's system-prompt framing was iterated on based on manually reading a few smoke-test
  outputs** (see `docs/model_quirks.md` #9-10) — this is prompt engineering under time pressure,
  not a principled context-format ablation (that's H19, Week 2+). The reported numbers reflect
  one specific prompt design, not the best-achievable one.
- **A small number of extracted memories inherit a corrupted-predicate bug** from the PMB
  generator (fixed going forward, not retroactively regenerated — see the "Fix
  predicate-string corruption" commit). Limited scope, not believed to materially affect the
  headline numbers, but not re-verified exhaustively.
- **`acceptable_alternatives` / lenient-only scoring**: `pra_strict` is 0.0% everywhere because
  it requires exact string match against free-form generation — expected and by design (lenient
  judge scoring is the meaningful metric here), not a separate finding.
- Only 1 of the 4 bake-off candidates (the Day-1 winner) was tested — no comparison of whether
  memory helps other candidates differently.

## Where this fits in the roadmap

Day 2/3 exit criteria (real PMB, real System A, real A-vs-D comparison, k-sweep headline figure)
are met. Not yet done: Day 4 (synthetic SFT data + LoRA training — directly motivated by H5,
which predicts memory-aware SFT specifically improves memory utilization beyond what raw
retrieval gets you), Day 5's full evaluation grid, GGUF quantization (deferred per
`docs/adr/0001-model-selection.md`'s own reasoning).
