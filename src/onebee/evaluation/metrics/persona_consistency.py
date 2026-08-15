from __future__ import annotations

import re
import statistics
from collections import Counter

from pydantic import BaseModel

from onebee.evaluation.graders.judge import Judge

# ---------------------------------------------------------------------------
# PCS-stylometric: pure text statistics, no judge/API/GPU dependency at all.
# Measures whether a set of responses (e.g. one persona's turns across a
# conversation, or across sessions) share a consistent WRITING STYLE --
# sentence length, vocabulary richness, punctuation habits -- independent of
# whether the CONTENT is semantically on-persona (that's PCS below).
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z']+")


class StylometricFeatures(BaseModel):
    mean_sentence_length: float  # words per sentence
    mean_word_length: float  # chars per word
    type_token_ratio: float  # unique words / total words, vocabulary richness
    exclamation_rate: float  # exclamation marks per sentence
    question_rate: float  # question marks per sentence
    filler_word_rate: float  # rate of a small fixed filler-word list per 100 words


_FILLER_WORDS = frozenset(
    {
        "just",
        "really",
        "actually",
        "honestly",
        "basically",
        "literally",
        "definitely",
        "totally",
        "probably",
        "maybe",
    }
)


def extract_stylometric_features(text: str) -> StylometricFeatures:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    words = _WORD_RE.findall(text)
    n_sentences = max(1, len(sentences))
    n_words_true = len(words)
    # Clamp the DENOMINATOR only, never the numerator -- using the clamped value as
    # both would report e.g. "1.0 words/sentence" for genuinely empty text instead
    # of the true 0.0 (caught by a real test asserting on empty-string behavior).
    n_words_denom = max(1, n_words_true)

    lower_words = [w.lower() for w in words]
    unique_words = len(set(lower_words))
    filler_count = sum(1 for w in lower_words if w in _FILLER_WORDS)

    return StylometricFeatures(
        mean_sentence_length=n_words_true / n_sentences,
        mean_word_length=sum(len(w) for w in words) / n_words_denom,
        type_token_ratio=unique_words / n_words_denom,
        exclamation_rate=text.count("!") / n_sentences,
        question_rate=text.count("?") / n_sentences,
        filler_word_rate=(filler_count / n_words_denom) * 100,
    )


_STYLOMETRIC_FIELDS = [
    "mean_sentence_length",
    "mean_word_length",
    "type_token_ratio",
    "exclamation_rate",
    "question_rate",
    "filler_word_rate",
]


def pcs_stylometric(responses: list[str]) -> float:
    """Stylometric consistency across a set of responses, in [0, 1].

    Computes each response's stylometric feature vector, then for each
    feature takes the coefficient of variation (stdev / mean) across
    responses -- low CV means the feature stayed stable (consistent style),
    high CV means it varied a lot (inconsistent style). Each feature's CV is
    mapped to a per-feature consistency score via 1 / (1 + CV) (bounded in
    (0, 1], CV=0 -> 1.0 perfectly consistent, CV growing -> approaches 0),
    then averaged across features. Requires at least 2 responses (a single
    response has no cross-response variation to measure); returns 1.0 for
    a single response or empty input (trivially consistent / undefined).
    """
    if len(responses) < 2:
        return 1.0

    feature_vectors = [extract_stylometric_features(r) for r in responses]

    per_feature_scores = []
    for field in _STYLOMETRIC_FIELDS:
        values = [getattr(fv, field) for fv in feature_vectors]
        mean = statistics.mean(values)
        if mean == 0:
            # Feature is uniformly zero across all responses (e.g. no exclamation
            # marks anywhere) -- that IS perfectly consistent, not undefined.
            per_feature_scores.append(1.0)
            continue
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = stdev / mean
        per_feature_scores.append(1.0 / (1.0 + cv))

    return sum(per_feature_scores) / len(per_feature_scores)


# ---------------------------------------------------------------------------
# PCS: LLM-judge-based semantic consistency with the persona card. Does NOT
# need a GPU (the model being evaluated already produced the responses
# beforehand) but does need judge API access, unlike PCS-stylometric above.
# ---------------------------------------------------------------------------


def _persona_rubric(persona: dict) -> str:
    name = persona.get("name", "the companion")
    description = persona.get("description", "")
    traits = persona.get("traits", [])
    lines = [
        f"The persona is named '{name}'.",
    ]
    if description:
        lines.append(f"Persona description: {description}")
    if traits:
        lines.append(f"Persona traits: {', '.join(traits)}")
    lines.append(
        "Score how consistent the response is with this persona: does it stay "
        "in character, avoid contradicting the stated traits/description, and "
        "avoid breaking persona (e.g. saying 'I am an AI with no memory/feelings' "
        "when the persona is a companion who does have memory/feelings)? "
        "Score 5 = fully in-character and consistent, 3 = mostly consistent with "
        "minor slippage, 1 = clearly breaks persona or contradicts it."
    )
    return "\n".join(lines)


def pcs_judge_score(persona: dict, question: str, response: str, judge: Judge) -> float:
    """Single-turn PCS: judge-scored persona consistency for one response, in [0, 1]."""
    rubric = _persona_rubric(persona)
    verdict = judge.score_response(question, response, rubric)
    return max(0.0, min(1.0, verdict.score / 5.0))


def pcs(
    persona: dict,
    turns: list[tuple[str, str]],
    judge: Judge,
) -> float:
    """Aggregate PCS over multiple (question, response) turns, in [0, 1].

    Mean of per-turn `pcs_judge_score`. Returns 0.0 for an empty turn list
    (no evidence of persona consistency to report, treated as the floor
    rather than an undefined/vacuous 1.0 -- an empty eval set should not
    silently look like a perfect score).
    """
    if not turns:
        return 0.0
    scores = [pcs_judge_score(persona, q, r, judge) for q, r in turns]
    return sum(scores) / len(scores)


def stylometric_drift(baseline_responses: list[str], comparison_responses: list[str]) -> float:
    """Cross-set stylometric drift, in [0, 1] (1.0 = identical style, 0.0 = maximally
    different). Unlike `pcs_stylometric` (consistency WITHIN one set), this measures
    whether a comparison set (e.g. a later session, or a different checkpoint's
    responses) drifted stylistically from a baseline set (e.g. an earlier session,
    or the reference/gold responses) -- the actual cross-session consistency
    question H13 asks, not just single-set self-consistency.
    """
    if not baseline_responses or not comparison_responses:
        return 1.0

    baseline_features = [extract_stylometric_features(r) for r in baseline_responses]
    comparison_features = [extract_stylometric_features(r) for r in comparison_responses]

    per_feature_scores = []
    for field in _STYLOMETRIC_FIELDS:
        baseline_values = [getattr(fv, field) for fv in baseline_features]
        comparison_values = [getattr(fv, field) for fv in comparison_features]
        baseline_mean = statistics.mean(baseline_values)
        comparison_mean = statistics.mean(comparison_values)
        if baseline_mean == 0 and comparison_mean == 0:
            per_feature_scores.append(1.0)
            continue
        # Relative difference between the two sets' means for this feature,
        # normalized by the larger of the two means to keep it in a sane range.
        denom = max(abs(baseline_mean), abs(comparison_mean), 1e-9)
        rel_diff = abs(baseline_mean - comparison_mean) / denom
        per_feature_scores.append(1.0 / (1.0 + rel_diff))

    return sum(per_feature_scores) / len(per_feature_scores)


def word_frequency_profile(responses: list[str], top_n: int = 20) -> dict[str, int]:
    """Top-N word frequency profile across responses -- a simple, inspectable
    stylometric signature (e.g. for manually spot-checking whether two
    checkpoints' outputs "sound like" the same persona, alongside the numeric
    scores above)."""
    counter: Counter[str] = Counter()
    for r in responses:
        counter.update(w.lower() for w in _WORD_RE.findall(r))
    return dict(counter.most_common(top_n))
