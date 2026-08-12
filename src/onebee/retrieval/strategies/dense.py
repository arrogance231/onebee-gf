from __future__ import annotations

from onebee.memory.store import MemoryStore
from onebee.retrieval.fusion import RetrievalCandidate


class DenseRetriever:
    """
    Dense (vector-similarity) retrieval backed by the MemoryStore.

    The underlying store returns results ordered by vector distance but does
    **not** expose raw similarity scores.  Because RRF only needs rank position
    (not magnitude), we assign *rank* by the store's result order (1-indexed)
    and note that *rrf_score* is computed purely from this ordinal position.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        query_embedding: list[float],
        tier: str | None = None,
        k: int = 20,
    ) -> list[RetrievalCandidate]:
        records = self.store.search(query_embedding=query_embedding, tier=tier, k=k)
        return [
            RetrievalCandidate(
                record=r.model_dump(),
                memory_id=r.id,
                source="dense",
                rank=i + 1,
            )
            for i, r in enumerate(records)
        ]
