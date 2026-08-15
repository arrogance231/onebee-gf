# Distillation v1 prompt set (H23)

Prompt-only (system+user) examples extracted from `data/sft/v1/train.jsonl` for on-policy
distillation via `trl.DistillationTrainer` (student generates its own completions during
training, scored against the teacher's token distribution -- no assistant turns needed or
used from the source data).

- Total: 2232 (2008 train / 224 val)
- Source: `data/sft/v1/train.jsonl` (already contamination-checked clean against
  `pmb_v0_full`, see `data/sft/v1/DATASHEET.md` -- since this is a strict subset with no new
  content, no new contamination check was run).
