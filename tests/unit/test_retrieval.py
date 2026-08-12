from __future__ import annotations

import time
from importlib.util import find_spec

import pytest

from onebee.memory.store import MemoryRecord, MemoryStore
from onebee.retrieval.fusion import (
    FusionWeights,
    RetrievalCandidate,
    apply_tier_quota,
    compute_composite_score,
    mmr_select,
    reciprocal_rank_fusion,
)
from onebee.retrieval.router import HybridRetriever
from onebee.retrieval.strategies.bm25 import SparseRetriever
from onebee.retrieval.strategies.dense import DenseRetriever

_HAS_SQLITE_VEC = find_spec("sqlite_vec") is not None

NOW_MS = int(time.time() * 1000)
TAU_MS = 7 * 24 * 3600 * 1000


def _make_record(
    tier: str = "episodic",
    content: str = "test content",
    importance: float = 0.5,
    confidence: float = 0.9,
    entities: list[str] | None = None,
    event_time: int | None = None,
    **kwargs,
) -> MemoryRecord:
    defaults: dict = {
        "tier": tier,
        "content": content,
        "created_at": NOW_MS,
        "importance": importance,
        "confidence": confidence,
        "decay_rate": 0.01,
        "provenance": {
            "source": "test",
            "session_id": "s1",
            "turn_ids": [],
            "extractor": "test",
            "extractor_version": "0.1",
            "verbatim_span": None,
        },
        "entities": entities,
        "event_time": event_time,
    }
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


def _record_to_dict(r: MemoryRecord) -> dict:
    return r.model_dump()


class TestReciprocalRankFusion:
    def test_single_list(self):
        c = RetrievalCandidate(record={}, memory_id="a", source="bm25", rank=1)
        result = reciprocal_rank_fusion([[c]])
        # score = 1/(60 + 1) = 1/61 ≈ 0.016393
        assert "a" in result
        assert abs(result["a"] - (1.0 / 61.0)) < 1e-9

    def test_two_lists_same_id_different_ranks(self):
        c1 = RetrievalCandidate(record={}, memory_id="x", source="bm25", rank=1)
        c2 = RetrievalCandidate(record={}, memory_id="x", source="dense", rank=5)
        # score = 1/61 + 1/65 ≈ 0.016393 + 0.015385 = 0.031778
        expected = 1.0 / 61.0 + 1.0 / 65.0
        result = reciprocal_rank_fusion([[c1], [c2]])
        assert abs(result["x"] - expected) < 1e-9

    def test_multiple_ids(self):
        c1 = RetrievalCandidate(record={}, memory_id="a", source="bm25", rank=1)
        c2 = RetrievalCandidate(record={}, memory_id="b", source="bm25", rank=2)
        c3 = RetrievalCandidate(record={}, memory_id="a", source="dense", rank=3)
        # a: 1/61 + 1/63 ≈ 0.016393 + 0.015873 = 0.032266
        # b: 1/62 ≈ 0.016129
        result = reciprocal_rank_fusion([[c1, c2], [c3]])
        expected_a = 1.0 / 61.0 + 1.0 / 63.0
        expected_b = 1.0 / 62.0
        assert abs(result["a"] - expected_a) < 1e-9
        assert abs(result["b"] - expected_b) < 1e-9


class TestComputeCompositeScore:
    def test_numeric_example(self):
        # Worked example with known inputs.
        # record has:
        #   event_time = now_ms (so recency = exp(0) = 1.0)
        #   importance = 0.8
        #   confidence = 0.9
        #   entities = ["cat", "dog"]
        # query_entities = {"cat", "rat"}   => Jaccard = 1/3 ≈ 0.3333
        # rrf_score = 0.05  (already < 1, so norm_rrf = 0.05)
        #
        # Default FusionWeights:
        #   w_sim=0.30, w_rec=0.15, w_imp=0.15, w_conf=0.10, w_ent=0.10
        #
        # score = 0.30*0.05 + 0.15*1.0 + 0.15*0.8 + 0.10*0.9 + 0.10*0.3333...
        #       = 0.015 + 0.15 + 0.12 + 0.09 + 0.03333...
        #       = 0.408333...
        record = _record_to_dict(
            _make_record(
                tier="episodic",
                importance=0.8,
                confidence=0.9,
                entities=["cat", "dog"],
                event_time=NOW_MS,
            )
        )
        weights = FusionWeights()
        query_entities = {"cat", "rat"}
        score = compute_composite_score(
            record=record,
            rrf_score=0.05,
            query_entities=query_entities,
            weights=weights,
            now_ms=NOW_MS,
        )
        # 0.30*0.05 + 0.15*1.0 + 0.15*0.8 + 0.10*0.9 + 0.10*(1/3)
        expected = 0.30 * 0.05 + 0.15 * 1.0 + 0.15 * 0.8 + 0.10 * 0.9 + 0.10 * (1.0 / 3.0)
        assert abs(score - expected) < 1e-6

    def test_recency_decay(self):
        # event_time = 7 days ago => recency = exp(-tau/tau) = exp(-1) ≈ 0.3679
        old_ts = NOW_MS - TAU_MS
        record = _record_to_dict(
            _make_record(
                tier="episodic",
                importance=0.0,
                confidence=0.0,
                entities=None,
                event_time=old_ts,
            )
        )
        weights = FusionWeights()
        score = compute_composite_score(
            record=record,
            rrf_score=0.0,
            query_entities=set(),
            weights=weights,
            now_ms=NOW_MS,
        )
        import math

        expected_rec = math.exp(-1.0)  # 0.3679
        expected = weights.w_rec * expected_rec
        assert abs(score - expected) < 1e-6

    def test_clamps_to_range(self):
        record = _record_to_dict(
            _make_record(
                tier="episodic",
                importance=10.0,
                confidence=10.0,
                entities=None,
                event_time=NOW_MS,
            )
        )
        # Weights total > 1.0 to force a high raw score
        weights = FusionWeights(w_sim=0.0, w_rec=0.0, w_imp=1.0, w_conf=1.0, w_ent=0.0)
        score = compute_composite_score(
            record=record,
            rrf_score=100.0,
            query_entities=set(),
            weights=weights,
            now_ms=NOW_MS,
        )
        assert 0.0 <= score <= 1.0


class TestMMRSelect:
    def test_highest_score_picked_first(self):
        candidates = [("a", 0.9), ("b", 0.5), ("c", 0.3)]
        result = mmr_select(candidates, lambda _a, _b: 0.0, k=3)
        assert result[0] == "a"

    def test_reduces_redundancy(self):
        # "dup1" and "dup2" are near-duplicate high-score candidates.
        # "distinct" is a lower-score distinct candidate.
        # With k=2 and lambda=0.7, MMR should pick one of the dups first
        # (highest score), then pick "distinct" over the other dup because
        # the redundancy penalty outweighs the score advantage.
        candidates = [("dup1", 0.9), ("dup2", 0.85), ("distinct", 0.5)]

        def sim(a: str, b: str) -> float:
            # dup1 and dup2 are 90% similar; everything else is 0%
            if "dup" in a and "dup" in b:
                return 0.9
            return 0.0

        result = mmr_select(candidates, sim, k=2)
        assert result[0] == "dup1"
        # Second pick: "dup2" scores: 0.7*0.85 - 0.3*0.9 = 0.595-0.27=0.325
        # "distinct" scores: 0.7*0.5 - 0.3*0.0 = 0.35
        # distinct wins because it has no redundancy penalty
        assert result[1] == "distinct"

    def test_respects_k(self):
        candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        result = mmr_select(candidates, lambda _a, _b: 0.0, k=2)
        assert len(result) == 2


class TestApplyTierQuota:
    def test_drops_overflow_preserves_order(self):
        ids = [
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",  # episodic: quota 4 -> e5 dropped
            "s1",
            "s2",
            "s3",
            "s4",  # semantic: quota 3 -> s4 dropped
            "st1",
            "st2",
        ]  # short_term: quota 6 -> none dropped
        lookup = {
            "e1": {"tier": "episodic"},
            "e2": {"tier": "episodic"},
            "e3": {"tier": "episodic"},
            "e4": {"tier": "episodic"},
            "e5": {"tier": "episodic"},
            "s1": {"tier": "semantic"},
            "s2": {"tier": "semantic"},
            "s3": {"tier": "semantic"},
            "s4": {"tier": "semantic"},
            "st1": {"tier": "short_term"},
            "st2": {"tier": "short_term"},
        }
        result = apply_tier_quota(ids, lookup)
        assert result == [
            "e1",
            "e2",
            "e3",
            "e4",
            "s1",
            "s2",
            "s3",
            "st1",
            "st2",
        ]

    def test_custom_quotas(self):
        ids = ["e1", "e2", "s1"]
        lookup = {
            "e1": {"tier": "episodic"},
            "e2": {"tier": "episodic"},
            "s1": {"tier": "semantic"},
        }
        result = apply_tier_quota(ids, lookup, {"episodic": 1, "semantic": 10})
        assert result == ["e1", "s1"]


class TestEmptyStore:
    def test_dense_empty_store(self, tmp_path):
        store = MemoryStore(str(tmp_path / "empty.db"))
        retriever = DenseRetriever(store)
        result = retriever.retrieve([0.1] * 384)
        assert result == []

    def test_sparse_empty_store(self, tmp_path):
        store = MemoryStore(str(tmp_path / "empty.db"))
        retriever = SparseRetriever(store)
        result = retriever.retrieve("test query")
        assert result == []


class TestHybridRetriever:
    @pytest.fixture
    def populated_store(self, tmp_path):
        store = MemoryStore(str(tmp_path / "test.db"))
        store.write(
            _make_record(
                tier="episodic",
                content="Python is a great programming language",
                importance=0.9,
                confidence=0.95,
                entities=["python", "programming"],
                event_time=NOW_MS - 1000,
            )
        )
        store.write(
            _make_record(
                tier="episodic",
                content="Javascript runs in the browser",
                importance=0.3,
                confidence=0.8,
                entities=["javascript", "browser"],
                event_time=NOW_MS - 2000,
            )
        )
        store.write(
            _make_record(
                tier="semantic",
                content="Docker containers isolate processes",
                importance=0.7,
                confidence=0.9,
                entities=["docker", "containers"],
                event_time=NOW_MS - 500000,
            )
        )
        store.write(
            _make_record(
                tier="short_term",
                content="The meeting is at 3pm tomorrow",
                importance=0.6,
                confidence=0.7,
                entities=["meeting", "schedule"],
                event_time=NOW_MS - 100,
            )
        )
        store.write(
            _make_record(
                tier="episodic",
                content="ChatGPT is a large language model",
                importance=0.5,
                confidence=0.85,
                entities=["chatgpt", "llm", "ai"],
                event_time=NOW_MS - 500,
            )
        )
        return store

    def test_text_only_returns_results(self, populated_store):
        retriever = HybridRetriever(populated_store)
        results = retriever.retrieve(
            query_text="python programming",
            query_entities={"python"},
            k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, RetrievalCandidate)
            assert r.memory_id
            assert r.composite_score >= 0.0
            assert r.rrf_score >= 0.0

    def test_text_only_does_not_crash(self, populated_store):
        retriever = HybridRetriever(populated_store)
        results = retriever.retrieve(query_text="python", query_embedding=None)
        assert isinstance(results, list)

    def test_with_fake_embedding(self, populated_store):
        retriever = HybridRetriever(populated_store)
        if _HAS_SQLITE_VEC:
            results = retriever.retrieve(
                query_text="programming",
                query_embedding=[0.1] * 384,
                query_entities={"python"},
                k=3,
            )
            assert len(results) > 0
        else:
            # Without sqlite-vec, the dense path is a no-op and
            # text-only should still work.
            results = retriever.retrieve(
                query_text="programming",
                query_embedding=[0.1] * 384,
                query_entities={"python"},
                k=3,
            )
            assert isinstance(results, list)

    def test_tier_filter(self, populated_store):
        retriever = HybridRetriever(populated_store)
        results = retriever.retrieve(
            query_text="containers",
            tier="semantic",
            k=5,
        )
        for r in results:
            assert r.record.get("tier") == "semantic"

    def test_k_truncation(self, populated_store):
        retriever = HybridRetriever(populated_store)
        results = retriever.retrieve(query_text="language", k=2, k_candidates=10)
        assert len(results) <= 2


class TestImportRoundtrip:
    def test_public_exports(self):
        from onebee.retrieval import (
            DenseRetriever,
            FusionWeights,
            HybridRetriever,
            RetrievalCandidate,
            SparseRetriever,
            apply_tier_quota,
            compute_composite_score,
            mmr_select,
            reciprocal_rank_fusion,
        )

        assert DenseRetriever is not None
        assert SparseRetriever is not None
        assert HybridRetriever is not None
        assert RetrievalCandidate is not None
        assert FusionWeights is not None
        assert callable(apply_tier_quota)
        assert callable(compute_composite_score)
        assert callable(mmr_select)
        assert callable(reciprocal_rank_fusion)
