from __future__ import annotations

from onebee.memory.store import MemoryStore
from onebee.retrieval.fusion import RetrievalCandidate


class SparseRetriever:
    """Lexical (FTS5/BM25) retrieval backed by the MemoryStore's built-in
    full-text search.  No separate BM25 implementation – the store's FTS5
    already provides ranked lexical results."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        query_text: str,
        tier: str | None = None,
        k: int = 20,
    ) -> list[RetrievalCandidate]:
        records = self.store.search(query=query_text, tier=tier, k=k)
        return [
            RetrievalCandidate(
                record=r.model_dump(),
                memory_id=r.id,
                source="bm25",
                rank=i + 1,
            )
            for i, r in enumerate(records)
        ]
