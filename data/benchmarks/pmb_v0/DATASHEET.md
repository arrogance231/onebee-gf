# PMB v0 — Personalised Memory Benchmark

## Overview
The PMB-v0 benchmark evaluates a system's ability to recall and reason over personalised information across multi-session conversations.

## Generation

- **Teacher client**: `fixture`
- **Personas**: 2
- **Facts per persona**: 8
- **Total probes**: 36

Probe categories: factual, preference, episodic, temporal, unanswerable, outdated_fact, distractor, continuity.

## Limitations

- This corpus was generated with a **fixture teacher** (deterministic templates, no live LLM). Conversations and probes are synthetic approximations, not naturally generated.
- A real teacher model (OpenAI-compatible endpoint) is required for production-quality data. Pass `--teacher real --teacher-endpoint ...` once wired up.
- Distractor and continuity probes are lightweight approximations; a real teacher would produce richer variants.
- Per-category probe counts are derived proportionally from `--facts-per-persona` rather than targeting the absolute reference distribution from the paper.
- This fixture-scale run (2 personas) is a smoke-test example, not the full v0 benchmark (target: 8 personas).
