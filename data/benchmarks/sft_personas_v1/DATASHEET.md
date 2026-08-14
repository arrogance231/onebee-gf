# PMB v0 — Personalised Memory Benchmark

## Overview
The PMB-v0 benchmark evaluates a system's ability to recall and reason over personalised information across multi-session conversations.

## Generation

- **Teacher client**: `openai`
- **Personas**: 40
- **Facts per persona**: 40
- **Total probes**: 3437

Probe categories: factual, preference, episodic, temporal, unanswerable, outdated_fact, distractor, continuity.

## Limitations

- This corpus was generated with a **live teacher model** (OpenAI-compatible endpoint), not a deterministic fixture. It is **not yet human-reviewed** — conversations and probes should be spot-checked by a human before being treated as production-quality reference data.
