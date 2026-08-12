from __future__ import annotations

import re
from typing import Dict, List, Protocol

from pydantic import BaseModel

from onebee.memory.extraction.schema import ExtractedClaim
from onebee.memory.extraction.scoring import compute_confidence
from onebee.memory.extraction.validators import (
    GroundingChecker,
    NullGroundingChecker,
    validate_claim,
)


class TeacherExtractor(Protocol):
    def extract(self, turn_text: str, context: dict) -> list[ExtractedClaim]: ...


class ExtractionResult(BaseModel):
    claim: ExtractedClaim
    confidence: float
    rejected: bool
    rejection_reason: str | None = None


_PREFERENCE_VERBS = r"\b(?:like|love|hate|prefer|enjoy|dislike)\b"

_ASSERTION_FIRST_PERSON = r"\b(?:I am|I have|I like|I hate|I work|I live|I feel|my )"

_TEMPORAL_WEEKDAY = (
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
)
_TEMPORAL_RELATIVE = r"\b(?:yesterday|last week|tomorrow|ago)\b"
_TEMPORAL_DATE = r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"


class SalienceGate:
    def _has_entity_heuristic(self, text: str) -> bool:
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            words = sentence.strip().split()
            for i, word in enumerate(words):
                cleaned = word.strip("'\",;:()[]{}")
                if i == 0:
                    continue
                if cleaned and cleaned[0].isupper() and len(cleaned) > 1 and cleaned.upper() != cleaned:
                    return True
        return False

    def _has_first_person_assertion(self, text: str) -> bool:
        return bool(re.search(_ASSERTION_FIRST_PERSON, text, re.IGNORECASE))

    def _has_temporal(self, text: str) -> bool:
        if re.search(_TEMPORAL_WEEKDAY, text, re.IGNORECASE):
            return True
        if re.search(_TEMPORAL_RELATIVE, text, re.IGNORECASE):
            return True
        if re.search(_TEMPORAL_DATE, text):
            return True
        return False

    def _has_preference_verb(self, text: str) -> bool:
        return bool(re.search(_PREFERENCE_VERBS, text, re.IGNORECASE))

    def trigger_reason(self, turn_text: str) -> str | None:
        if self._has_entity_heuristic(turn_text):
            return "entity_heuristic"
        if self._has_first_person_assertion(turn_text):
            return "first_person_assertion"
        if self._has_temporal(turn_text):
            return "temporal"
        if self._has_preference_verb(turn_text):
            return "preference_verb"
        return None

    def should_trigger(self, turn_text: str) -> bool:
        return self.trigger_reason(turn_text) is not None


class ExtractionPipeline:
    def __init__(
        self,
        extractor: TeacherExtractor,
        grounding_checker: GroundingChecker | None = None,
        grounding_threshold: float = 0.5,
    ) -> None:
        self._gate = SalienceGate()
        self._extractor = extractor
        self._grounding_checker = grounding_checker or NullGroundingChecker()
        self._grounding_threshold = grounding_threshold

    def process_turn(
        self, turn_text: str, context: dict | None = None
    ) -> list[ExtractionResult]:
        if not self._gate.should_trigger(turn_text):
            return []

        if context is None:
            context = {}

        claims = self._extractor.extract(turn_text, context)
        results: list[ExtractionResult] = []

        for claim in claims:
            passed, reason = validate_claim(claim, turn_text)

            if passed:
                grounding_score = self._grounding_checker.check(
                    claim.content, turn_text
                )
                if grounding_score < self._grounding_threshold:
                    results.append(
                        ExtractionResult(
                            claim=claim,
                            confidence=0.0,
                            rejected=True,
                            rejection_reason="grounding_low",
                        )
                    )
                    continue

            confidence = (
                compute_confidence(
                    claim.extractor_confidence,
                    claim.assertion_strength,
                    claim.source_reliability,
                )
                if passed
                else 0.0
            )

            results.append(
                ExtractionResult(
                    claim=claim,
                    confidence=confidence,
                    rejected=not passed,
                    rejection_reason=reason if not passed else None,
                )
            )

        return results
