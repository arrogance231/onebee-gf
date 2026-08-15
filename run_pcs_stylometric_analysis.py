import json
import sys
from pathlib import Path

from onebee.evaluation.metrics.persona_consistency import (
    pcs_stylometric,
    stylometric_drift,
    word_frequency_profile,
)

# Real application of the new PCS-stylometric metric against already-saved eval
# transcripts -- no GPU/API needed, this is pure text analysis over data already on
# disk. Answers a real question left open by docs/distillation_results.md: did
# on-policy distillation (which pulls the student toward a non-persona-tuned 8B
# teacher) measurably change writing STYLE consistency, separate from the semantic
# quality/persona-consistency questions already answered by the judge-based evals?

SYSTEMS = {
    "B_sft_v1 (SFT alone)": "results/v1_scale/B_sft/raw.jsonl",
    "E_sft_memory (pre-distillation)": "results/v1_scale/E_sft_memory/raw.jsonl",
    "E_distill (post-distillation, H23)": "results/v1_scale/E_distill/raw.jsonl",
}


def load_responses(path: str) -> list[str]:
    responses = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            resp = rec.get("response", "")
            if resp.strip():
                responses.append(resp)
    return responses


results = {}
for name, path in SYSTEMS.items():
    p = Path(path)
    if not p.exists():
        print(f"skip {name}: {path} not found", file=sys.stderr)
        continue
    responses = load_responses(str(p))
    self_consistency = pcs_stylometric(responses)
    top_words = word_frequency_profile(responses, top_n=10)
    results[name] = {
        "n_responses": len(responses),
        "pcs_stylometric_self_consistency": round(self_consistency, 4),
        "top_10_words": top_words,
    }
    print(f"{name}: n={len(responses)} self_consistency={self_consistency:.4f}", file=sys.stderr)

# Cross-system drift: does the post-distillation system write in a measurably
# different STYLE than the pre-distillation system? (Separate question from the
# semantic pairwise-judge comparison already in docs/distillation_results.md.)
if "E_sft_memory (pre-distillation)" in [k for k in SYSTEMS if Path(SYSTEMS[k]).exists()]:
    pre = load_responses(SYSTEMS["E_sft_memory (pre-distillation)"])
    post = load_responses(SYSTEMS["E_distill (post-distillation, H23)"])
    drift = stylometric_drift(pre, post)
    results["stylometric_drift_pre_vs_post_distillation"] = round(drift, 4)
    print(f"stylometric_drift(pre-distill, post-distill) = {drift:.4f}", file=sys.stderr)

out_dir = Path("results/v1_scale/pcs_stylometric_analysis")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
