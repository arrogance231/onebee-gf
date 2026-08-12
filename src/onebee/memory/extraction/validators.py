from __future__ import annotations

import re
from typing import Protocol

from onebee.memory.extraction.schema import ExtractedClaim

TRIVIAL_STOPLIST: list[str] = [
    "said hello",
    "said hi",
    "greeted the user",
    "asked how",
    "user said goodbye",
    "acknowledged",
    "thanked the assistant",
]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def verify_span(claim: ExtractedClaim, source_text: str) -> bool:
    return _normalize_ws(claim.verbatim_span) in _normalize_ws(source_text)


def is_trivial(claim: ExtractedClaim) -> bool:
    content_lower = claim.content.lower()
    word_count = len(content_lower.split())
    if word_count >= 8:
        return False
    return any(phrase in content_lower for phrase in TRIVIAL_STOPLIST)


def validate_claim(claim: ExtractedClaim, source_text: str) -> tuple[bool, str | None]:
    if not verify_span(claim, source_text):
        return (False, "no_span_match")

    if is_trivial(claim):
        return (False, "trivial")

    if (
        claim.attribution == "third_party"
        and claim.tier == "semantic"
        and (claim.subject is None or claim.subject == "user")
    ):
        return (False, "attribution_mismatch")

    return (True, None)


class GroundingChecker(Protocol):
    def check(self, claim_content: str, source_text: str) -> float: ...


class NullGroundingChecker:
    def check(self, claim_content: str, source_text: str) -> float:
        return 1.0
