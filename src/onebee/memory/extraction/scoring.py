from __future__ import annotations

from onebee.memory.extraction.schema import (
    ASSERTION_STRENGTH_MULTIPLIER,
    SOURCE_RELIABILITY_MULTIPLIER,
    AssertionStrength,
    SourceReliability,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_importance(
    affect_arousal: float,
    novelty: float,
    entity_salience: float,
    user_emphasis: float,
    consequence: float,
) -> float:
    affect_arousal = _clamp(affect_arousal)
    novelty = _clamp(novelty)
    entity_salience = _clamp(entity_salience)
    user_emphasis = _clamp(user_emphasis)
    consequence = _clamp(consequence)

    raw = (
        0.30 * affect_arousal
        + 0.20 * novelty
        + 0.20 * entity_salience
        + 0.15 * user_emphasis
        + 0.15 * consequence
    )
    return _clamp(raw)


def compute_confidence(
    extractor_confidence: float,
    assertion_strength: AssertionStrength,
    source_reliability: SourceReliability,
) -> float:
    raw = (
        extractor_confidence
        * ASSERTION_STRENGTH_MULTIPLIER[assertion_strength]
        * SOURCE_RELIABILITY_MULTIPLIER[source_reliability]
    )
    return _clamp(raw)


_STRENGTH_CUES: list[tuple[list[str], AssertionStrength]] = [
    (["definitely", "certainly", "always"], "definite"),
    (["maybe", "perhaps", "might"], "uncertain"),
    (["i think", "i guess", "probably"], "moderate"),
]


def detect_assertion_strength(text: str) -> AssertionStrength:
    text_lower = text.lower()
    for cue_phrases, strength in _STRENGTH_CUES:
        for phrase in cue_phrases:
            if phrase in text_lower:
                return strength
    return "moderate"
