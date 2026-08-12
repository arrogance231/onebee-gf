from __future__ import annotations

import math
from typing import Any, Callable

from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    record: dict[str, Any]
    memory_id: str
    source: str
    rank: int
    rrf_score: float = 0.0
    composite_score: float = 0.0


class FusionWeights(BaseModel):
    """
    Weights for composite scoring.  Should sum to roughly 1.0 but this is NOT
    enforced (intentional – experiments zero out individual terms deliberately).

    ``w_bm`` is present for a future per-source-score design but is NOT
    currently used as a separate additive term: both dense and BM25 feeds
    enter the same RRF pool, so the BM25 contribution is already captured by
    the fused *rrf_score*.
    """

    w_sim: float = 0.30
    w_bm: float = 0.20
    w_rec: float = 0.15
    w_imp: float = 0.15
    w_conf: float = 0.10
    w_ent: float = 0.10


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def reciprocal_rank_fusion(
    candidate_lists: list[list[RetrievalCandidate]],
    k_constant: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidates in candidate_lists:
        for cand in candidates:
            score = 1.0 / (k_constant + cand.rank)
            scores[cand.memory_id] = scores.get(cand.memory_id, 0.0) + score
    return scores


def compute_composite_score(
    record: dict[str, Any],
    rrf_score: float,
    query_entities: set[str],
    weights: FusionWeights,
    now_ms: int,
    tau_ms: float = 7 * 24 * 3600 * 1000,
) -> float:
    # RRF score is already a small bounded-ish value; clamp to [0, 1] as an
    # approximation of true cosine/BM25 fusion since raw scores aren't retained.
    norm_rrf = min(1.0, rrf_score)

    ts = record.get("event_time") or record.get("created_at", 0)
    recency = math.exp(-(now_ms - ts) / tau_ms)

    importance = record.get("importance", 0.0)

    confidence = record.get("confidence", 0.0)

    record_entities = set(record.get("entities") or [])
    entity_overlap = _jaccard(query_entities, record_entities)

    score = (
        weights.w_sim * norm_rrf
        + weights.w_rec * recency
        + weights.w_imp * importance
        + weights.w_conf * confidence
        + weights.w_ent * entity_overlap
    )

    return max(0.0, min(1.0, score))


def mmr_select(
    candidates: list[tuple[str, float]],
    similarity_fn: Callable[[str, str], float],
    k: int,
    lambda_param: float = 0.7,
) -> list[str]:
    if not candidates:
        return []

    remaining = set(candidates)
    selected: list[str] = []

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            best = max(
                remaining,
                key=lambda x: (
                    lambda_param * x[1]
                    - (1 - lambda_param) * max(similarity_fn(x[0], s) for s in selected)
                ),
            )
        selected.append(best[0])
        remaining.remove(best)

    return selected[: min(k, len(selected))]


def apply_tier_quota(
    ranked_ids: list[str],
    record_lookup: dict[str, dict[str, Any]],
    quotas: dict[str, int] | None = None,
) -> list[str]:
    if quotas is None:
        quotas = {"episodic": 4, "semantic": 3, "short_term": 6}

    tier_counts: dict[str, int] = {}
    result: list[str] = []

    for mid in ranked_ids:
        record = record_lookup.get(mid)
        if record is None:
            continue
        tier = record.get("tier", "episodic")
        current = tier_counts.get(tier, 0)
        limit = quotas.get(tier)
        if limit is not None and current >= limit:
            continue
        tier_counts[tier] = current + 1
        result.append(mid)

    return result
