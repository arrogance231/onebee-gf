# DPO v1_scale preference dataset (proper scale)

Persona-contrastive preference pairs: chosen = real teacher (gpt-5.6-luna) response using
injected memory context (same pipeline as data/sft/v1), rejected = a generic persona-breaking
AI-assistant disclaimer, matching the exact failure mode observed in Day 3's raw-model
evaluation ("I am a large language model, trained by Google."). Built from the same 40 SFT
personas as data/sft/v1/ (disjoint from the PMB-v0 eval set).

- Total pairs: 2277 (2049 train / 228 val)
- Only turns where memory was actually retrieved are included (a meaningful preference signal
  requires there to be a "right" character-consistent answer to prefer).
- rejected responses are a fixed pool of 5 hand-written disclaimer sentences, not
  teacher-generated — a deliberate, cheap way to encode "always prefer being in-character over
  a generic disclaimer" as the DPO signal, not a nuanced quality preference.
- Not human-reviewed.
