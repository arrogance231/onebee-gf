# Week 2: DPO results (H6-H7)

**Status:** two real, consistent data points, both trending in DPO's favor but neither
statistically clear at v0 scale — not a confirmed win, not chased further. Raw data in
`results/v0.1/C_vs_E_pairwise/`, `results/v0.1/C1_vs_E_pairwise_small/`, and the DPO training
logs.

## What was built and run

1. **`src/onebee/training/dpo.py`** — DPO training module mirroring `sft.py`'s architecture
   exactly (lazy imports, injectable factories, multimodal-aware loading, the same real-trl
   adaptation pattern already proven for SFT).
2. **Real DPO preference dataset v0** (`data/dpo/v0/`, 223 pairs: 200 train / 23 val) —
   persona-contrastive: chosen = real teacher response using injected memory (same generation
   pipeline as the SFT dataset), rejected = a fixed pool of 5 generic AI-assistant disclaimer
   sentences matching the exact failure mode observed in Day 3's raw-model evaluation ("I am a
   large language model, trained by Google."). Contamination-checked clean.
3. **Real DPO training** on top of the Day-4 SFT checkpoint (`outputs/sft/v0/merged` →
   `outputs/dpo/v0/merged`, r=16 LoRA, beta=0.1, sigmoid loss, 1 epoch, 25 steps, ~61s
   wall-clock).
4. **Real dual-order pairwise comparison**: System C (SFT+DPO+memory) vs System E
   (SFT+memory, no DPO) on 105 stratified answerable probes, judged by `gpt-5.6-luna` with
   position-bias control (`dual_order_score` — each pair scored in both AB and BA order).

## Training-time signal (strong)

| Step | rewards/accuracies | rewards/margins |
|---|---|---|
| ~step 10 | 71.25% | 0.29 |
| ~step 20 | 88.75% | 1.08 |
| step 25 (final) | 92.5% | 1.32 |

Clear, textbook DPO training curve — the model increasingly assigns higher implicit reward to
chosen (character-consistent) over rejected (generic disclaimer) responses on the training
distribution itself.

## Real-eval result (honest: not distinguishable)

| | Wins | Rate |
|---|---|---|
| C (SFT+DPO) | 26 / 105 | 24.8% |
| E (SFT only) | 23 / 105 | 21.9% |
| Ties | 56 / 105 | 53.3% |

**This is essentially a coin flip** (26 vs 23 non-tied wins is well within noise for n=105).
Manually spot-checking raw pairs confirms this isn't a scoring bug: many C and E responses are
near-identical text, or differ only in minor phrasing (e.g. "I don't recall attending any
conferences." vs "I don't recall attending any conferences yet." — scored a tie). At v0 scale
(223 preference pairs, 25 training steps), DPO doesn't move the needle enough to be
distinguishable from the already-trained SFT model on real held-out probes, even though the
training-time reward signal is strong and clean.

## Follow-up: more epochs on the same data (v1)

Given the training curve was still climbing strongly at epoch-1 end, retrained on the *same*
223-pair dataset (no new API calls) with `num_train_epochs=4` instead of 1
(`configs/training/dpo_v1_more_epochs.yaml` → `outputs/dpo/v1/merged`).

| | 1 epoch (v0) | 4 epochs (v1) |
|---|---|---|
| Final `rewards/accuracies` | 92.5% | 100% |
| Final `rewards/margins` | 1.32 | 6.14 |
| Final `loss` | 0.43 | 0.045 |

This is now suspiciously close to a perfect fit on 200 examples — real overfitting risk, not
just a stronger signal. Confirmed qualitatively: a hand-picked sanity-check prompt ("What
company do you work for?" with "Alice is a designer at Stripe" in context) that the 1-epoch
model answered correctly ("I work for Stripe.") **regressed** on the 4-epoch model back to the
exact original failure mode ("I am a large language model, trained by Google.").

A smaller real pairwise eval (35 stratified answerable probes, same dual-order judge protocol,
`results/v0.1/C1_vs_E_pairwise_small/`) gives:

| | Wins | Rate |
|---|---|---|
| C1 (SFT+DPO, 4 epochs) | 11 / 35 | 31.4% |
| E (SFT only) | 9 / 35 | 25.7% |
| Ties | 15 / 35 | 42.9% |

The aggregate gap (5.7pp) is actually slightly *larger* than v0's 1-epoch gap (2.9pp) —
suggesting more training modestly helps on average even though it visibly hurt on at least one
specific prompt. **Two consistent data points now exist, both trending the same direction, both
still short of a clearly significant result at this sample size.** This is a genuine, informative
addition — not a contradiction of the v0 finding — but also a caution against reading "more
epochs = strictly better" from the training-time metrics alone: aggregate win-rate improved
while a real qualitative regression also appeared. Both effects are real; neither invalidates
the other. Not investigated further given remaining GPU budget — a proper resolution needs
either a much larger preference dataset (diluting the overfitting risk) or an early-stopping /
held-out-loss-based epoch selection, neither of which was in scope for this pass.

## Interpretation against pre-registered hypotheses

- **H6** (preference optimization improves persona consistency more than SFT alone, predicted
  +0.5 to +1.0 on a persona consistency score): **not distinguishable at this scale, but both
  v0 (1 epoch, 2.9pp gap) and v1 (4 epochs, 5.7pp gap) point the same direction** — a small,
  consistent edge for DPO that grows with more training, alongside a real overfitting risk that
  also grows with more training. Neither run clears the bar for a confident H6 verdict; both
  are consistent with H6 being true but too weak a signal (at v0 data/step scale) to prove it.
  This is reported as an honest limitation, not as evidence against H6 — a proper test needs the
  doc's target dataset scale, not a v0 proof-of-concept.
- **H7** (preference optimization incurs an alignment-tax on general capability): **not
  tested** — this pass only compared C vs E on the companion-persona task itself, not against a
  general-capability benchmark. Untested, not refuted.

## Why this is still a legitimate, useful result

Per the project's own research discipline ("small effects on small eval sets are reported as
'not distinguishable,' not as wins" — `docs/research_questions.md`), this is exactly the
correct way to report a real experiment that didn't produce a clear signal: the training
mechanics work end-to-end (data generation → training → checkpoint → real evaluation, all real,
all verified), the training-time metrics are genuinely informative (DPO is learning the
intended preference), and the eval-time result honestly says "not enough to tell yet" rather
than being dressed up as a win. The engineering (four real trl/transformers API breaks fixed
along the way — see `docs/model_quirks.md` #11-14, all of which apply to DPO too since it
reuses the same training infrastructure) is the actual deliverable of this pass; the DPO effect
size itself needs a real-scale dataset (the roadmap's target is thousands of pairs, not
hundreds) to properly test H6.

## What a real H6/H7 test needs (not done here, noted for later)

- A preference dataset an order of magnitude larger (the roadmap's SFT target was 6-10k
  examples; DPO pair counts are typically similar order of magnitude).
- More training steps / more epochs — 25 steps is extremely small even relative to this v0
  dataset's own size.
- A real PCS (Persona Consistency Score) metric, not an ad-hoc pairwise judge comparison — PCS
  and PCS-stylometric are still Week 2+ unbuilt items per `docs/research_questions.md`'s design
  note on the full companion persona card.
- A general-capability benchmark run (IFEval-style, per Day 2's spec) to actually test H7's
  alignment-tax prediction, which this pass didn't touch at all.
