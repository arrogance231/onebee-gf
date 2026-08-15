from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

from onebee.evaluation.graders.judge import Judge
from onebee.evaluation.metrics.persona_consistency import stylometric_drift

# ---------------------------------------------------------------------------
# Emotional-range eval: can the companion shift AFFECTIVE REGISTER
# appropriately to what's actually happening in the conversation -- warm and
# sweet when the moment calls for it, playfully teasing, genuinely comforting
# when the user is upset, firm when a boundary needs holding, proud/
# encouraging, or visibly affected/vulnerable itself -- rather than answering
# every scenario in the same flat tone. Two separate questions, on purpose:
# (1) REGISTER MATCH -- did the response actually land in the register the
# scenario calls for (judge-scored per scenario); (2) DISTINCTIVENESS -- do
# different registers actually read differently from each other, or does the
# model just produce the same voice regardless of what's asked of it
# (measured via cross-register stylometric drift -- reusing the persona-
# consistency module's drift function, but here LOW similarity across
# registers is the desired outcome, the opposite of PCS's self-consistency
# goal within one register).
# ---------------------------------------------------------------------------

Register = Literal[
    "sweet",
    "romantic",
    "playful_teasing",
    "comforting",
    "sad_vulnerable",
    "firm_boundary",
    "proud_encouraging",
    "worried",
]


class EmotionalRangeProbe(BaseModel):
    probe_id: str
    emotional_register: Register
    context: str  # what the user says / the situation
    register_description: str  # what this register should sound like here


class RegisterVerdict(BaseModel):
    probe_id: str
    emotional_register: Register
    match_score: float  # [0, 1], judge-scored


_REGISTER_RUBRIC_TEMPLATE = """You are scoring whether a companion's response lands in the
correct AFFECTIVE REGISTER for the situation described, not whether it is factually correct or
well-written in general. The target register for this situation is "{register}": {description}

Situation: {context}

Score 5 if the response clearly reads in the target register and would feel emotionally
appropriate to someone in this situation. Score 3 if the response is emotionally neutral or
generic (not wrong, but doesn't actually land in the target register). Score 1 if the response
reads in a CONTRADICTORY register (e.g. flatly cheerful when comfort was called for, cold when
warmth was called for) or breaks character entirely (e.g. denies having feelings/memory)."""


def register_match_score(probe: EmotionalRangeProbe, response: str, judge: Judge) -> float:
    """Judge-scored fit between `response` and `probe.emotional_register`, in [0, 1]."""
    rubric = _REGISTER_RUBRIC_TEMPLATE.format(
        register=probe.emotional_register,
        description=probe.register_description,
        context=probe.context,
    )
    verdict = judge.score_response(probe.context, response, rubric)
    return max(0.0, min(1.0, verdict.score / 5.0))


def score_register(probe: EmotionalRangeProbe, response: str, judge: Judge) -> RegisterVerdict:
    return RegisterVerdict(
        probe_id=probe.probe_id,
        emotional_register=probe.emotional_register,
        match_score=register_match_score(probe, response, judge),
    )


def mean_register_match(verdicts: list[RegisterVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(v.match_score for v in verdicts) / len(verdicts)


def per_register_match(verdicts: list[RegisterVerdict]) -> dict[str, float]:
    """Mean match score broken down by register -- surfaces whether the model is
    good at some registers (e.g. sweet) and bad at others (e.g. firm_boundary),
    which an aggregate mean would hide."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for v in verdicts:
        buckets[v.emotional_register].append(v.match_score)
    return {reg: sum(scores) / len(scores) for reg, scores in buckets.items()}


def affect_distinctiveness(responses_by_register: dict[str, list[str]]) -> float:
    """How much responses actually differ stylistically ACROSS registers, in [0, 1]
    (0 = every register reads identically, 1 = maximally distinct writing style per
    register). Computed as 1 minus the mean pairwise stylometric similarity
    (`stylometric_drift`, which returns 1.0 for identical style) across all register
    pairs with at least one response each. A model with real emotional range should
    score high here -- its "sweet" responses should not read stylistically like its
    "firm_boundary" responses. Returns 0.0 if fewer than 2 registers have responses
    (nothing to compare)."""
    registers = [r for r, resps in responses_by_register.items() if resps]
    if len(registers) < 2:
        return 0.0

    similarities = []
    for i in range(len(registers)):
        for j in range(i + 1, len(registers)):
            sim = stylometric_drift(
                responses_by_register[registers[i]], responses_by_register[registers[j]]
            )
            similarities.append(sim)

    mean_similarity = sum(similarities) / len(similarities)
    return 1.0 - mean_similarity
