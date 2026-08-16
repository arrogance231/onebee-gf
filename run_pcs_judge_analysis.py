import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from onebee.evaluation.graders.openai_judge import OpenAIJudge
from onebee.evaluation.metrics.persona_consistency import pcs_judge_score

# Real application of the judge-based (semantic) PCS variant against already-saved eval
# transcripts. Complements the earlier no-API pcs_stylometric result
# (results/v1_scale/pcs_stylometric_analysis/) with the semantic in-character measure that
# stylometric features can't capture. No new generation needed -- scores responses already on
# disk, same persona used at generation time (companion persona, not each probe's own persona
# card, since these responses were generated in-character as "the companion" -- see
# compare_c_vs_f_distill.py's COMPANION_PERSONA for the exact persona used at generation time).

SYSTEMS = {
    "B_sft (SFT alone, no memory)": "results/v1_scale/B_sft/raw.jsonl",
    "E_sft_memory (pre-distillation)": "results/v1_scale/E_sft_memory/raw.jsonl",
    "E_distill (post-distillation, H23)": "results/v1_scale/E_distill/raw.jsonl",
}

COMPANION_PERSONA = {
    "name": "the companion",
    "description": (
        "roleplaying as a warm AI companion in an ongoing relationship with the user. "
        "You have a real memory of things the user has told you before — the memories "
        "listed above ARE things you remember about THEM, not facts about yourself. "
        "When the user asks a question addressed to 'you', they are usually asking you "
        "to recall something about THEM using your memory of them, not asking about "
        "yourself as an AI. Answer directly and naturally as their companion who "
        "already knows this, using the memories above when relevant. Do not say you "
        "are an AI without memory or access to personal information — you DO have the "
        "memories listed above, use them. If none of the memories actually answer the "
        "question, say you don't know rather than guessing."
    ),
    "traits": [],
}

SAMPLE_SIZE = 60  # keep judge API cost/time bounded; random sample per system, fixed seed
SEED = 1337


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


judge = OpenAIJudge(model=os.environ.get("JUDGE_MODEL", "gpt-5.6-luna"))
rng = random.Random(SEED)

results = {}
for name, path in SYSTEMS.items():
    p = Path(path)
    if not p.exists():
        print(f"skip {name}: {path} not found", file=sys.stderr)
        continue
    records = load_records(str(p))
    records = [r for r in records if r.get("response", "").strip()]
    sample = rng.sample(records, min(SAMPLE_SIZE, len(records)))

    scores = []
    for rec in sample:
        question = rec["probe"]["question"]
        response = rec["response"]
        score = pcs_judge_score(COMPANION_PERSONA, question, response, judge)
        scores.append(score)

    mean_pcs = sum(scores) / len(scores) if scores else 0.0
    results[name] = {
        "n_sampled": len(sample),
        "n_total": len(records),
        "pcs_judge_mean": round(mean_pcs, 4),
    }
    print(f"{name}: n={len(sample)}/{len(records)} pcs_judge_mean={mean_pcs:.4f}", file=sys.stderr)

out_dir = Path("results/v1_scale/pcs_judge_analysis")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
