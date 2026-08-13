# Week 2: DPO results (H6-H7)

**Status:** real result, honestly inconclusive at v0 scale — not a win, not chased further.
Raw data in `results/v0.1/C_vs_E_pairwise/` and the DPO training log.

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

## Interpretation against pre-registered hypotheses

- **H6** (preference optimization improves persona consistency more than SFT alone, predicted
  +0.5 to +1.0 on a persona consistency score): **not distinguishable at this scale** — neither
  confirmed nor refuted. The training curve shows DPO is *learning something* (the reward
  margin genuinely grows), but 223 pairs / 25 steps is not enough signal to produce a
  measurable difference on 105 real held-out probes. This is reported as an honest limitation,
  not as evidence against H6 — a proper test needs the doc's target dataset scale, not a v0
  proof-of-concept.
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
