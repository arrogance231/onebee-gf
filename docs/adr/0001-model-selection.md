# ADR-0001: Base model selection

**Status:** pending — bake-off not yet run

## Context

The project needs to pin a ~1B parameter instruction-tuned base model as the substrate for all
post-training and scaffolding experiments (per the research design's commitment to hold the
base model fixed as long as possible). Candidates under consideration:

- Qwen3-1.7B-Instruct
- Llama-3.2-1B-Instruct
- Gemma-3-1b-it

## Decision process

`scripts/model_bakeoff.py` runs a 40-prompt smoke set (10 instruction-following, 10 EN
dialogue, 10 JA dialogue, 10 structured-context-adherence — the most predictive test for this
project's memory-injection use case) against each candidate, scored by a judge plus a manual
read of all 120 outputs.

## Decision

TBD — to be filled in once the bake-off runs, with the comparison table, the pinned revision
SHA, and the rejected alternatives with reasons.

## Consequences

Once pinned, `configs/model/` selects this model everywhere; changing it later re-baselines
every downstream result per the project's experimental philosophy.
