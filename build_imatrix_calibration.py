import json
import sys
from pathlib import Path

# Build an imatrix calibration corpus from this project's OWN real data (companion
# conversations, memory-grounded responses, persona-consistency preference pairs) rather than
# a generic corpus like wikitext -- the importance matrix should reflect the actual task
# distribution (companion chat, memory recall/abstention, in-character responses), not generic
# language modeling, since that's what we actually want the quantizer to preserve precision on.
OUT = Path("data/imatrix_calibration.txt")

lines: list[str] = []

# SFT v1 train: full real conversations (system + user + assistant), the core companion-chat
# distribution this model is actually used for.
sft_path = Path("data/sft/v1/train.jsonl")
with open(sft_path) as f:
    for line in f:
        ex = json.loads(line)
        for m in ex["messages"]:
            role = m["role"].upper()
            lines.append(f"{role}: {m['content']}")
        lines.append("")

# DPO v1_scale train: chosen (preferred, persona-consistent) responses only -- the rejected
# generic-disclaimer responses are deliberately NOT included, since we don't want the
# quantizer preserving precision on behavior we're actively training the model away from.
dpo_path = Path("data/dpo/v1_scale/train.jsonl")
with open(dpo_path) as f:
    for line in f:
        ex = json.loads(line)
        lines.append(f"PROMPT: {ex['prompt']}")
        lines.append(f"ASSISTANT: {ex['chosen']}")
        lines.append("")

OUT.write_text("\n".join(lines))
print(f"DONE. {len(lines)} lines, {OUT.stat().st_size} bytes -> {OUT}", file=sys.stderr)
