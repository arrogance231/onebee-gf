from __future__ import annotations

from typing import Literal, Protocol

NLILabel = Literal["entailment", "contradiction", "neutral"]

_NEGATION_MARKERS: set[str] = {"not", "never", "no", "n't", "no longer", "neither", "nor"}


def _content_words(text: str) -> set[str]:
    words = text.lower().split()
    stops = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "also",
        "that",
        "this",
        "and",
        "but",
        "or",
        "so",
        "if",
        "than",
        "too",
        "very",
        "just",
        "then",
        "now",
        "not",
        "no",
        "n't",
        "never",
    }
    return {w for w in words if w not in stops and len(w) > 1}


def _has_negation_difference(premise: str, hypothesis: str) -> bool:
    prem_neg = _NEGATION_MARKERS & set(premise.lower().split())
    hyp_neg = _NEGATION_MARKERS & set(hypothesis.lower().split())
    return bool(prem_neg ^ hyp_neg)


class NLIChecker(Protocol):
    def check(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]: ...


class FakeNLI:
    def check(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        prem_words = _content_words(premise)
        hyp_words = _content_words(hypothesis)
        if not prem_words or not hyp_words:
            return "neutral", 1.0
        shared = prem_words & hyp_words
        share_ratio = len(shared) / max(len(prem_words), len(hyp_words))
        if share_ratio >= 0.5 and not _has_negation_difference(premise, hypothesis):
            return "entailment", 1.0
        if _has_negation_difference(premise, hypothesis) and len(shared) >= 1:
            return "contradiction", 1.0
        return "neutral", 1.0
