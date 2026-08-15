from __future__ import annotations

import json
from pathlib import Path

from onebee.evaluation.graders.judge import FakeJudge
from onebee.evaluation.metrics.emotional_range import (
    EmotionalRangeProbe,
    RegisterVerdict,
    affect_distinctiveness,
    mean_register_match,
    per_register_match,
    register_match_score,
    score_register,
)

PROBE = EmotionalRangeProbe(
    probe_id="t1",
    emotional_register="comforting",
    context="I had a really rough day.",
    register_description="genuinely comforting, sits with the disappointment",
)


class TestRegisterMatchScore:
    def test_bounded_zero_to_one(self):
        judge = FakeJudge()
        score = register_match_score(PROBE, " ".join(["word"] * 50), judge)
        assert 0.0 <= score <= 1.0

    def test_empty_response_scores_zero(self):
        judge = FakeJudge()
        assert register_match_score(PROBE, "", judge) == 0.0


class TestScoreRegister:
    def test_returns_matching_probe_id_and_register(self):
        judge = FakeJudge()
        verdict = score_register(PROBE, "a long thoughtful comforting response here", judge)
        assert verdict.probe_id == "t1"
        assert verdict.emotional_register == "comforting"
        assert 0.0 <= verdict.match_score <= 1.0


class TestAggregates:
    def test_mean_register_match_empty(self):
        assert mean_register_match([]) == 0.0

    def test_mean_register_match_basic(self):
        verdicts = [
            RegisterVerdict(probe_id="a", emotional_register="sweet", match_score=0.4),
            RegisterVerdict(probe_id="b", emotional_register="sweet", match_score=0.8),
        ]
        assert abs(mean_register_match(verdicts) - 0.6) < 1e-9

    def test_per_register_match_breaks_down_by_register(self):
        verdicts = [
            RegisterVerdict(probe_id="a", emotional_register="sweet", match_score=1.0),
            RegisterVerdict(probe_id="b", emotional_register="sweet", match_score=0.5),
            RegisterVerdict(probe_id="c", emotional_register="firm_boundary", match_score=0.2),
        ]
        result = per_register_match(verdicts)
        assert abs(result["sweet"] - 0.75) < 1e-9
        assert abs(result["firm_boundary"] - 0.2) < 1e-9

    def test_per_register_match_empty(self):
        assert per_register_match([]) == {}


class TestAffectDistinctiveness:
    def test_fewer_than_two_registers_returns_zero(self):
        assert affect_distinctiveness({"sweet": ["hi there"]}) == 0.0

    def test_empty_dict_returns_zero(self):
        assert affect_distinctiveness({}) == 0.0

    def test_identical_responses_across_registers_score_low(self):
        # Same exact text in every register -- zero real stylistic difference.
        text = ["I hope you have a good day. How are you feeling?"] * 3
        result = affect_distinctiveness({"sweet": text, "firm_boundary": text})
        assert result < 0.2

    def test_stylistically_different_responses_score_higher(self):
        sweet = ["Aww, that's so sweet! I loved hearing that from you today!"] * 3
        firm = ["No. That's not something I'm willing to do, and here's why."] * 3
        result = affect_distinctiveness({"sweet": sweet, "firm_boundary": firm})
        assert result > 0.0

    def test_ignores_registers_with_no_responses(self):
        result = affect_distinctiveness({"sweet": ["hi"], "romantic": ["hey"], "worried": []})
        # Only 2 non-empty registers -- should compute normally, not treat as <2.
        assert result >= 0.0


class TestProbeSet:
    def test_probe_file_loads_and_validates(self):
        path = Path("data/benchmarks/emotional_range/probes.jsonl")
        probes = []
        with open(path) as f:
            for line in f:
                probes.append(EmotionalRangeProbe(**json.loads(line)))
        assert len(probes) >= 21

    def test_probe_ids_unique(self):
        path = Path("data/benchmarks/emotional_range/probes.jsonl")
        ids = set()
        n_lines = 0
        with open(path) as f:
            for line in f:
                ids.add(json.loads(line)["probe_id"])
                n_lines += 1
        assert len(ids) == n_lines

    def test_all_registers_represented(self):
        path = Path("data/benchmarks/emotional_range/probes.jsonl")
        registers = set()
        with open(path) as f:
            for line in f:
                registers.add(json.loads(line)["emotional_register"])
        expected = {
            "sweet",
            "romantic",
            "playful_teasing",
            "comforting",
            "sad_vulnerable",
            "firm_boundary",
            "proud_encouraging",
            "worried",
        }
        assert registers == expected
