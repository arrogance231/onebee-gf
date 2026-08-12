from __future__ import annotations

import time
from typing import Any

from onebee.memory.store import MemoryStore
from onebee.retrieval.fusion import (
    FusionWeights,
    RetrievalCandidate,
    _jaccard,
    apply_tier_quota,
    compute_composite_score,
    mmr_select,
    reciprocal_rank_fusion,
)
from onebee.retrieval.strategies.bm25 import SparseRetriever
from onebee.retrieval.strategies.dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        weights: FusionWeights | None = None,
    ) -> None:
        self.store = store
        self.weights = weights or FusionWeights()
        self.sparse = SparseRetriever(store)
        self.dense = DenseRetriever(store)

    def retrieve(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        query_entities: set[str] | None = None,
        tier: str | None = None,
        k: int = 8,
        k_candidates: int = 20,
        tier_quotas: dict[str, int] | None = None,
    ) -> list[RetrievalCandidate]:
        candidate_lists: list[list[RetrievalCandidate]] = []

        sparse_results = self.sparse.retrieve(query_text, tier=tier, k=k_candidates)
        candidate_lists.append(sparse_results)

        if query_embedding is not None:
            dense_results = self.dense.retrieve(
                query_embedding, tier=tier, k=k_candidates
            )
            candidate_lists.append(dense_results)

        rrf_scores = reciprocal_rank_fusion(candidate_lists)

        now_ms = int(time.time() * 1000)
        entities = query_entities or set()

        scored: list[tuple[str, float]] = []
        record_lookup: dict[str, dict[str, Any]] = {}
        source_lookup: dict[str, str] = {}

        for clist in candidate_lists:
            for cand in clist:
                if cand.memory_id not in source_lookup:
                    record_lookup[cand.memory_id] = cand.record
                    source_lookup[cand.memory_id] = cand.source

        for memory_id in rrf_scores:
            record_obj = self.store.get_by_id(memory_id)
            if record_obj is None:
                # Fall back to the record we already have from the candidate list
                record_d = record_lookup.get(memory_id, {})
            else:
                record_d = record_obj.model_dump()
                record_lookup[memory_id] = record_d

            composite = compute_composite_score(
                record=record_d,
                rrf_score=rrf_scores[memory_id],
                query_entities=entities,
                weights=self.weights,
                now_ms=now_ms,
            )
            scored.append((memory_id, composite))

        scored.sort(key=lambda x: x[1], reverse=True)

        def entity_similarity(a: str, b: str) -> float:
            ents_a = set(record_lookup.get(a, {}).get("entities") or [])
            ents_b = set(record_lookup.get(b, {}).get("entities") or [])
            return _jaccard(ents_a, ents_b)

        mmr_ordered = mmr_select(scored, entity_similarity, k=k_candidates)

        quota_ordered = apply_tier_quota(mmr_ordered, record_lookup, tier_quotas)

        final_ids = quota_ordered[:k]

        composite_map = {mid: comp for mid, comp in scored}

        result: list[RetrievalCandidate] = []
        for mid in final_ids:
            result.append(
                RetrievalCandidate(
                    record=record_lookup.get(mid, {}),
                    memory_id=mid,
                    source=source_lookup.get(mid, "unknown"),
                    rank=0,
                    rrf_score=rrf_scores.get(mid, 0.0),
                    composite_score=composite_map.get(mid, 0.0),
                )
            )

        return result
