import json
import random
import sys
from pathlib import Path

# Extract prompt-only (system+user) examples from the existing SFT v1 dataset for H23's
# on-policy distillation run -- trl.DistillationTrainer needs a "prompt" column and the
# student generates its own completions, so we deliberately do NOT reuse the assistant
# turns here (unlike sft.py/dpo.py, which train on them directly).
SRC = Path("data/sft/v1/train.jsonl")
OUT_DIR = Path("data/distill/v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(271828)

examples = []
with open(SRC) as f:
    for line in f:
        ex = json.loads(line)
        messages = ex["messages"]
        prompt = [m for m in messages if m["role"] in ("system", "user")]
        if len(prompt) == 2:
            examples.append({"prompt": prompt})

rng.shuffle(examples)
split_idx = int(len(examples) * 0.9)
train = examples[:split_idx]
val = examples[split_idx:]

with open(OUT_DIR / "train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")
with open(OUT_DIR / "val.jsonl", "w") as f:
    for ex in val:
        f.write(json.dumps(ex) + "\n")

datasheet = f"""# Distillation v1 prompt set (H23)

Prompt-only (system+user) examples extracted from `data/sft/v1/train.jsonl` for on-policy
distillation via `trl.DistillationTrainer` (student generates its own completions during
training, scored against the teacher's token distribution -- no assistant turns needed or
used from the source data).

- Total: {len(examples)} ({len(train)} train / {len(val)} val)
- Source: `data/sft/v1/train.jsonl` (already contamination-checked clean against
  `pmb_v0_full`, see `data/sft/v1/DATASHEET.md` -- since this is a strict subset with no new
  content, no new contamination check was run).
"""
(OUT_DIR / "DATASHEET.md").write_text(datasheet)

print(f"DONE. train={len(train)} val={len(val)}", file=sys.stderr)
