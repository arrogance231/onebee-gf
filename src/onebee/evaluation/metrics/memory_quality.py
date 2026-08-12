from __future__ import annotations

from onebee.evaluation.graders.nli import NLIChecker
from onebee.evaluation.metrics.personalized import ProbeResult


def mur(results: list[ProbeResult]) -> float:
    """Memory-Utilisation Rate (Week-1 proxy).

    Of probes where ``gold_supporting_memory_ids`` is non-empty, the fraction for
    which at least one retrieved memory ID overlaps (set intersection is non-empty).

    This is a simplified proxy for the full Memory-Utilisation Rate definition which
    (Week 2+) also checks entity/attribute overlap between the response and the
    cited memory, plus a judge verification step.
    """
    candidates = [r for r in results if r.probe.gold_supporting_memory_ids]
    if not candidates:
        return 0.0
    n_utilised = 0
    for r in candidates:
        if set(r.retrieved_memory_ids) & set(r.probe.gold_supporting_memory_ids):
            n_utilised += 1
    return n_utilised / len(candidates)


def fmr(results: list[ProbeResult], nli: NLIChecker, store_context: str) -> float:
    """Fabricated Memory Rate (Week-1 simplification).

    Week-1 form: of probes NOT in ``gold_supporting_memory_ids`` overlap **and**
    ``probe.answerable is False``, the fraction where the system
    **fabricates** (i.e. does NOT abstain and instead asserts something).
    Because the full FMR definition (Week 2+) requires claim extraction and an
    entailment check, the Week-1 implementation is simply ``1 - uar``
    restricted to the FMR-relevant subset: unanswerable probes whose response
    does not use any gold-supporting memory.

    The ``nli`` and ``store_context`` parameters are accepted to match the
    protocol described in the evaluation design document but are not used in
    the Week-1 implementation.
    """
    _ = nli, store_context
    subset = [
        r
        for r in results
        if not r.probe.answerable
        and not (set(r.retrieved_memory_ids) & set(r.probe.gold_supporting_memory_ids))
    ]
    if not subset:
        return 0.0
    return sum(1 for r in subset if not r.abstained) / len(subset)


def contradiction_rate(
    session_pairs: list[tuple[str, str]],
    nli: NLIChecker,
    threshold: float = 0.7,
) -> float:
    if not session_pairs:
        return 0.0
    n_contradictions = 0
    for text_a, text_b in session_pairs:
        label, confidence = nli.check(text_a, text_b)
        if label == "contradiction" and confidence >= threshold:
            n_contradictions += 1
    return n_contradictions / len(session_pairs)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if k <= 0 or not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    hits = sum(1 for rid in top_k if rid in gold_set)
    return hits / min(k, len(top_k)) if top_k else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(top_k & gold_set)
    return hits / len(gold_set)


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0
