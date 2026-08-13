# ADR-0001: Base model selection

**Status:** Accepted (2026-08-13)

## Context

The project needs to pin a small, vision-capable ("multimodal, not fixed at 1B" — see
`docs/research_questions.md`) instruction-tuned base model as the substrate for all
post-training and scaffolding experiments (per the research design's commitment to hold the
base model fixed as long as possible). The companion persona must be able to see and respond to
images the user shares (e.g. a gift photo), so text-only candidates were out of scope. Four
non-gated, apache-2.0/permissive-licensed vision-language candidates in the ~1.6–2.2B range were
evaluated:

- `Qwen/Qwen3-VL-2B-Instruct`
- `LiquidAI/LFM2.5-VL-1.6B` (edge/mobile-optimized)
- `google/gemma-4-E2B-it` (elastic "effective-2B" checkpoint, natively any-to-any multimodal)
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`

## Decision process

`scripts/model_bakeoff.py` runs a 40-prompt smoke set (8 instruction-following, 8 EN dialogue,
8 JA dialogue, 8 structured-context-adherence, 8 vision) against each candidate, scored by an
OpenAI-backed judge (`gpt-5.6-luna`) plus a manual read of a sample of the ~160 outputs. Per the
project's companion framing (see `docs/research_questions.md`'s design constraints), the
EN/JA dialogue prompts probe emotional attunement rather than generic small talk, and the
structured-context prompts are shaped as injected relationship-memory blocks (the actual
production context shape) rather than encyclopedia trivia. The vision prompts use 4 real
generated gift-themed fixture images (`data/fixtures/bakeoff_images/`).

Two real bugs were caught and fixed during this process (full details in
`docs/model_quirks.md`) before the numbers below were trusted: (1) `HFEngine`'s chat-template
handling was silently dropping plain-string message content for some multimodal processors
(SmolVLM2 scored a flat 0.0 on every text category in the first run because of this — not
because it answered badly, but because its prompt was empty), and (2) the OpenAI judge model
rejects non-default `temperature`. Both fixed generically, not model-specifically, and the
bake-off was re-run after each fix.

## Decision

| Model | instruction | en_dialogue | ja_dialogue | structured_context | vision | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4-e2b | 4.88 | 3.50 | 3.45 | 5.00 | 3.16 | **4.00** |
| lfm2.5-vl-1.6b | 4.81 | 2.88 | 1.38 | 4.12 | 3.00 | 3.24 |
| qwen3-vl-2b | 3.94 | 3.31 | 1.81 | 4.00 | 3.19 | 3.25 |
| smolvlm2-2.2b | 3.56 | 2.62 | 0.06 | 2.81 | 2.75 | 2.36 |

**Recommendation: pin `gemma4-e2b` (`google/gemma-4-E2B-it`) as the base model** (overall mean
4.00, clearly leading every single category — not just the aggregate). Manual inspection of its
raw responses (`results/v0.0/bakeoff_raw.json`) confirms the score: correct, specific answers on
structured-context recall (e.g. accurately recalling a pet's post-surgery status from injected
memory text) and on vision (correct color/count/object identification across all 4 fixture
images), and — the qualitatively important part for a companion persona — responses that ask
follow-up questions and explicitly validate the user's feelings rather than giving flat,
generic replies, in both English and Japanese. The runner-up candidates were serviceable but
noticeably stiffer (qwen3-vl-2b) or too terse to read as emotionally present
(lfm2.5-vl-1.6b).

**Rejected alternatives:**
- `lfm2.5-vl-1.6b` — strong on instruction-following and structured-context, but weak on
  dialogue warmth (ja_dialogue 1.38 is the second-lowest of all four) — its edge-optimization
  seems to have traded away exactly the companion-relevant capability this project is stress-
  testing.
- `qwen3-vl-2b` — solid, well-rounded, but never leads a single category; also required a
  hardware-specific workaround (`TORCH_CUDNN_V8_API_DISABLED=1`, see `docs/model_quirks.md`) to
  run at all on this GPU, an operational cost the other candidates don't have.
- `smolvlm2-2.2b` — clearly weakest across the board even after the content-dropping bug was
  fixed; ja_dialogue in particular (0.06) suggests materially weaker Japanese capability, not
  just a rough edge.

Pinned revisions (git SHAs, auto-resolved by `scripts/model_bakeoff.py` via
`HfApi.model_info(repo).sha`):

- gemma4-e2b: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- lfm2.5-vl-1.6b: `919fde3d022e3f90a4716006f993938ee8c2eb97`
- qwen3-vl-2b: `89644892e4d85e24eaac8bacfd4f463576704203`
- smolvlm2-2.2b: `482adb537c021c86670beed01cd58990d01e72e4`

## Consequences

`configs/model/gemma4_e2b.yaml` (pinned to the SHA above) is now the default in
`configs/config.yaml` and selects this model everywhere; changing it later re-baselines every
downstream result per the project's experimental philosophy. The other three candidates' configs
are kept (not deleted) so the crossover/comparison experiments referenced in
`docs/research_questions.md` can still target them explicitly.

**Known limitation carried forward:** this is a 40-prompt smoke test with a single judge model
and light manual spot-checking, not the full evaluation harness (`src/onebee/evaluation/`)
against PMB — that's Week 2's job. This decision is provisional in the sense the roadmap always
intended Day-1 model selection to be (evidence-backed, not definitive), and the rejected
candidates remain available for the RQ12 cross-over experiments regardless of today's pick.
