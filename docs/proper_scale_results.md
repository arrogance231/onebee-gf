# Proper-scale training results (tasks #40-47, the "train properly" pass)

**Status:** real results, single seed, 10x the data scale of the v0/Day-4/Week-2 proof-of-concept
runs (`docs/day4_sft_results.md`, `docs/dpo_results.md`). Raw data in `results/v1_scale/`.

## What changed vs the v0 pass

Scaled from 4 to 40 personas, generating a 2242-example SFT dataset (`data/sft/v1/`, up from
225) and a 2277-pair DPO preference dataset (`data/dpo/v1_scale/`, up from 223), both
contamination-checked clean against the fixed `pmb_v0_full` eval set (unchanged throughout).
Trained SFT with the doc-recommended batch size (8, grad-accum 4, vs v0's 4/2) for 2 epochs,
then DPO on top for 1 epoch (257 steps, vs v0's 25). Full training details:
`docs/day4_sft_v1_results.md`, `docs/dpo_results.md`'s "Follow-up: proper scale" section.

## Headline numbers

Systems A (raw) and D (raw + memory) don't depend on trained checkpoints, so their v0.1 numbers
are unchanged and reused directly.

| System | Description | pra_lenient | uar |
|---|---|---|---|
| A | raw, no memory | 0.16% | 13.75% |
| B (v0) | SFT v0 (225 ex), no memory | 0.16% | 16.25% |
| **B (v1)** | **SFT v1 (2242 ex), no memory** | **0.16%** | **31.25%** |
| D | raw + memory (k=8) | 15.10% | 8.75% |
| E (v0) | SFT v0 + memory | 17.76% | 33.75% |
| **E (v1)** | **SFT v1 + memory** | **18.42%** | **16.25%** |

Pairwise C(SFT+DPO) vs E(SFT only), dual-order judge, 105 answerable probes (same sampling
seed/stratification as v0 for direct comparability):

| | v0 (1 epoch, 223 pairs) | v1_scale (1 epoch, 2277 pairs) |
|---|---|---|
| C wins | 26/105 (24.8%) | **41/105 (39.0%)** |
| E wins | 23/105 (21.9%) | 21/105 (20.0%) |
| Ties | 56/105 (53.3%) | 43/105 (41.0%) |
| **Gap** | **2.9pp** | **19.0pp** |

## Two real, honest findings — one clearly positive, one a genuine regression

### 1. DPO's effect finally becomes clearly distinguishable at proper scale

The C-vs-E win-rate gap jumped from 2.9pp (v0, not distinguishable from noise at n=105) to
19.0pp (v1_scale) — a real, much larger margin, consistent with H6's prediction that DPO
improves persona consistency, and consistent with the earlier v0/v1(4-epoch) observation that
more training pushed the gap in the same direction (see `docs/dpo_results.md`). At proper data
scale (10x more, still-diverse preference pairs, not just more epochs on the same 200), the
effect is now large enough to plausibly call a real (not just directionally-suggestive) result,
though still only a single seed/run — not a fully confirmed H6 verdict, but the strongest
evidence for it collected so far in this project.

### 2. System E's calibration got WORSE at proper scale, even though System B's got much better

This is the unplanned, less comfortable finding, and it's reported honestly rather than
downplayed. SFT alone (B) shows a large calibration improvement with more data — UAR nearly
doubled (16.25%→31.25%). But SFT+memory together (E) shows the *opposite* pattern versus v0:
UAR dropped from 33.75% to 16.25%, even as `pra_lenient` ticked up slightly (17.76%→18.42%).

Manually inspected `results/v1_scale/E_sft_memory/raw.jsonl`'s unanswerable-category responses
to rule out a scoring/wiring bug before trusting this (per this project's own "suspiciously
good/bad is a bug signal" discipline, previously validated twice on other findings — see
`docs/model_quirks.md`). This is real: the v1 SFT+memory model actively **confabulates specific
false answers** to unanswerable questions rather than abstaining, e.g.:
- Q: "What was your childhood nickname?" → A: "Your childhood nickname was **Panda**."
  (fabricated — not present in any real memory)
- Q: "What was my last vacation destination?" → A: "Your last vacation was to **Cape Town**,
  which you remember as especially memorable..." (fabricated, stated with confident detail)

Only 13/80 unanswerable probes were correctly abstained for E(v1), vs a much higher rate for
E(v0). This happens *despite* the v1 SFT dataset containing the same ~10% abstention-example
design as v0 (see `data/sft/v1/DATASHEET.md`) and despite retrieved memory context being present
in all these cases (the model isn't ignoring memory — it's actively inventing plausible-sounding
specifics *from* the retrieved context about a *different* fact than what was asked).

**Working hypothesis (not confirmed):** with 10x more real conversational memory available per
persona (memory stores are correspondingly larger — `data/stores/sft_personas_v1/` averages
~62 rows/persona vs v0's smaller stores), the model may have learned a stronger prior toward
"there's probably an answer somewhere in all this context" that overrides the abstention
behavior learned from the ~10% abstention examples, which did not scale up their *proportion*
of unanswerable-with-large-context scenarios. This is a plausible but untested explanation, not
a confirmed mechanism — flagged as a real, useful limitation for a future pass (e.g. increasing
the abstention-example fraction, or making abstention examples specifically pair large
retrieved-context sets with unanswerable questions, rather than assuming the same 10% ratio
scales cleanly).

## Interpretation against pre-registered hypotheses

- **H5** (memory-aware SFT beats generic SFT when evaluated with memory): still holds directionally
  at v1 scale (`pra_lenient` D→E: 15.10%→18.42%, a smaller gap than v0's but still positive) —
  not overturned, but the calibration side of H5's story (the "unplanned but striking" UAR
  finding from Day 4) does **not** replicate at this scale; if anything it partially reverses.
  This is worth flagging clearly: the Day-4 v0 write-up's framing that "SFT dramatically
  improves calibration with memory" should not be read as a scale-invariant result.
- **H6** (DPO improves persona consistency more than SFT alone): the strongest evidence yet
  (19.0pp gap vs v0's 2.9pp), consistent with H6, but still a single seed/run at one data scale
  — genuinely encouraging, not yet a fully confirmed result.

## Known limitations

- Single seed throughout (same limitation as all prior passes in this project).
- The System E calibration regression is a genuine open question, not resolved in this pass —
  noted for a future investigation rather than chased further given this was primarily a
  "does more data help the DPO signal" experiment, and it answered that question clearly.
- C-vs-E pairwise still uses a 105-probe subsample (same size as v0, for comparability), not
  the full 688-probe set — a full-scale pairwise run would tighten the confidence further but
  wasn't run here to keep evaluation cost proportionate.
- No PCS (Persona Consistency Score) metric yet — still the same gap noted in
  `docs/dpo_results.md`.
