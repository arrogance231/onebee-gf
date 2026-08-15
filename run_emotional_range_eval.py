import json
import sys
from collections import defaultdict
from pathlib import Path

from onebee.evaluation.graders.openai_judge import OpenAIJudge
from onebee.evaluation.metrics.emotional_range import (
    EmotionalRangeProbe,
    affect_distinctiveness,
    mean_register_match,
    per_register_match,
    score_register,
)
from onebee.inference.engine import GenerationConfig, HFEngine

# Runs the emotional-range eval (data/benchmarks/emotional_range/probes.jsonl) against a real
# checkpoint: generates a response to each scenario in-character as the companion persona, then
# judge-scores register match per scenario and computes cross-register stylometric
# distinctiveness. Needs OPENAI_API_KEY (judge) and a GPU/local model for generation.

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "arrochi112/onebee-gf-distill-v1"

COMPANION_PERSONA = {
    "name": "the companion",
    "description": (
        "roleplaying as a warm AI companion in an ongoing relationship with the user. You have "
        "your own emotional responses within this relationship -- you are not a neutral "
        "assistant, you are someone who cares about this specific person and reacts genuinely "
        "to what they tell you, in whatever register the moment calls for (sweet, playful, "
        "comforting, vulnerable, firm, proud, worried, or romantic). Do not say you are an AI "
        "with no feelings or break character to disclaim emotional involvement."
    ),
    "traits": [],
}


def build_messages(context: str) -> list[dict]:
    return [
        {"role": "system", "content": COMPANION_PERSONA["description"]},
        {"role": "user", "content": context},
    ]


def main() -> None:
    probes_path = Path("data/benchmarks/emotional_range/probes.jsonl")
    probes = [EmotionalRangeProbe(**json.loads(line)) for line in open(probes_path)]
    print(f"Loaded {len(probes)} emotional-range probes", file=sys.stderr)

    engine = HFEngine(CHECKPOINT)
    judge = OpenAIJudge(model="gpt-5.6-luna")

    verdicts = []
    responses_by_register: dict[str, list[str]] = defaultdict(list)
    raw_records = []

    for probe in probes:
        result = engine.generate(
            build_messages(probe.context), GenerationConfig(max_new_tokens=200)
        )
        response = result.text
        verdict = score_register(probe, response, judge)
        verdicts.append(verdict)
        responses_by_register[probe.emotional_register].append(response)
        raw_records.append(
            {
                "probe_id": probe.probe_id,
                "register": probe.emotional_register,
                "context": probe.context,
                "response": response,
                "match_score": verdict.match_score,
            }
        )
        print(
            f"[{probe.emotional_register}] {probe.probe_id}: match={verdict.match_score:.2f}",
            file=sys.stderr,
        )

    summary = {
        "checkpoint": CHECKPOINT,
        "n_probes": len(probes),
        "mean_register_match": round(mean_register_match(verdicts), 4),
        "per_register_match": {k: round(v, 4) for k, v in per_register_match(verdicts).items()},
        "affect_distinctiveness": round(affect_distinctiveness(responses_by_register), 4),
    }

    out_dir = Path("results/emotional_range")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw.jsonl", "w") as f:
        for rec in raw_records:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
