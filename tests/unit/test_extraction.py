from __future__ import annotations

import pytest

from onebee.memory.extraction.extractor import (
    ExtractionPipeline,
    SalienceGate,
)
from onebee.memory.extraction.schema import (
    ASSERTION_STRENGTH_MULTIPLIER,
    SOURCE_RELIABILITY_MULTIPLIER,
    ExtractedClaim,
)
from onebee.memory.extraction.scoring import (
    compute_confidence,
    compute_importance,
    detect_assertion_strength,
)
from onebee.memory.extraction.validators import (
    is_trivial,
    validate_claim,
    verify_span,
)


class FakeTeacherExtractor:
    def __init__(self, claims: list[ExtractedClaim] | None = None):
        self._claims = claims or []

    def extract(self, turn_text: str, context: dict) -> list[ExtractedClaim]:
        return self._claims


def _make_claim(
    content: str = "test claim",
    tier: str = "short_term",
    verbatim_span: str = "test claim",
    assertion_strength: str = "moderate",
    source_reliability: str = "user_statement",
    attribution: str = "user",
    subject: str | None = None,
    extractor_confidence: float = 0.9,
    **kwargs,
) -> ExtractedClaim:
    defaults: dict = {
        "content": content,
        "tier": tier,
        "verbatim_span": verbatim_span,
        "assertion_strength": assertion_strength,
        "source_reliability": source_reliability,
        "attribution": attribution,
        "subject": subject,
        "extractor_confidence": extractor_confidence,
    }
    defaults.update(kwargs)
    return ExtractedClaim(**defaults)


class TestVerifySpan:
    def test_exact_match(self):
        claim = _make_claim(content="hello world", verbatim_span="hello world")
        assert verify_span(claim, "hello world")

    def test_case_insensitive(self):
        claim = _make_claim(content="Hello", verbatim_span="HELLO")
        assert verify_span(claim, "hello")

    def test_whitespace_normalized(self):
        claim = _make_claim(content="a b", verbatim_span="a  b")
        assert verify_span(claim, "a   b")

    def test_no_match(self):
        claim = _make_claim(verbatim_span="xyz")
        assert not verify_span(claim, "abc def")


class TestIsTrivial:
    def test_trivial_said_hello(self):
        claim = _make_claim(content="said hello")
        assert is_trivial(claim)

    def test_trivial_acknowledged(self):
        claim = _make_claim(content="acknowledged")
        assert is_trivial(claim)

    def test_trivial_too_long(self):
        claim = _make_claim(content="said hello and then talked about many other things too")
        assert not is_trivial(claim)

    def test_nontrivial_substantive(self):
        claim = _make_claim(content="I work at a hospital as a surgeon in Boston")
        assert not is_trivial(claim)

    def test_trivial_greeted(self):
        claim = _make_claim(content="greeted the user")
        assert is_trivial(claim)


class TestValidateClaim:
    def test_passes_all_checks(self):
        claim = _make_claim(
            content="I live in New York",
            verbatim_span="I live in New York",
        )
        passed, reason = validate_claim(claim, "I live in New York")
        assert passed
        assert reason is None

    def test_no_span_match_rejected(self):
        claim = _make_claim(
            content="I live in New York",
            verbatim_span="I live in London",
        )
        passed, reason = validate_claim(claim, "I live in New York")
        assert not passed
        assert reason == "no_span_match"

    def test_trivial_rejected(self):
        claim = _make_claim(
            content="said hello",
            verbatim_span="said hello",
        )
        passed, reason = validate_claim(claim, "said hello")
        assert not passed
        assert reason == "trivial"

    def test_attribution_mismatch(self):
        claim = _make_claim(
            content="some fact",
            verbatim_span="some fact",
            attribution="third_party",
            tier="semantic",
            subject=None,
        )
        passed, reason = validate_claim(claim, "some fact")
        assert not passed
        assert reason == "attribution_mismatch"

    def test_attribution_mismatch_user_subject(self):
        claim = _make_claim(
            content="some fact",
            verbatim_span="some fact",
            attribution="third_party",
            tier="semantic",
            subject="user",
        )
        passed, reason = validate_claim(claim, "some fact")
        assert not passed
        assert reason == "attribution_mismatch"

    def test_third_party_semantic_with_subject_passes(self):
        claim = _make_claim(
            content="some fact",
            verbatim_span="some fact",
            attribution="third_party",
            tier="semantic",
            subject="Dr. Smith",
        )
        passed, reason = validate_claim(claim, "some fact")
        assert passed
        assert reason is None

    def test_span_fails_before_triviality(self):
        claim = _make_claim(
            content="said hello",
            verbatim_span="not in text",
        )
        passed, reason = validate_claim(claim, "said hello")
        assert not passed
        assert reason == "no_span_match"


class TestComputeImportance:
    def test_all_mid(self):
        result = compute_importance(0.5, 0.5, 0.5, 0.5, 0.5)
        expected = 0.30 * 0.5 + 0.20 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5 + 0.15 * 0.5
        assert result == pytest.approx(expected)

    def test_all_high(self):
        result = compute_importance(1.0, 1.0, 1.0, 1.0, 1.0)
        assert result == pytest.approx(1.0)

    def test_clamps_inputs(self):
        result = compute_importance(2.0, -0.5, 1.5, 0.5, 0.5)
        expected = 0.30 * 1.0 + 0.20 * 0.0 + 0.20 * 1.0 + 0.15 * 0.5 + 0.15 * 0.5
        assert result == pytest.approx(expected)

    def test_worked_example_1(self):
        result = compute_importance(0.8, 0.4, 0.6, 0.3, 0.9)
        expected = 0.30 * 0.8 + 0.20 * 0.4 + 0.20 * 0.6 + 0.15 * 0.3 + 0.15 * 0.9
        assert result == pytest.approx(expected)

    def test_worked_example_2(self):
        result = compute_importance(0.1, 0.2, 0.9, 0.1, 0.1)
        expected = 0.30 * 0.1 + 0.20 * 0.2 + 0.20 * 0.9 + 0.15 * 0.1 + 0.15 * 0.1
        assert result == pytest.approx(expected)


class TestComputeConfidence:
    def test_definite_user_statement(self):
        result = compute_confidence(0.9, "definite", "user_statement")
        expected = (
            0.9
            * ASSERTION_STRENGTH_MULTIPLIER["definite"]
            * SOURCE_RELIABILITY_MULTIPLIER["user_statement"]
        )
        assert result == pytest.approx(expected)

    def test_uncertain_reflection(self):
        result = compute_confidence(0.8, "uncertain", "reflection_derived")
        expected = (
            0.8
            * ASSERTION_STRENGTH_MULTIPLIER["uncertain"]
            * SOURCE_RELIABILITY_MULTIPLIER["reflection_derived"]
        )
        assert result == pytest.approx(expected)

    def test_worked_example_1(self):
        result = compute_confidence(0.95, "definite", "user_statement")
        expected = 0.95 * 1.0 * 1.0
        assert result == pytest.approx(expected)

    def test_worked_example_2(self):
        result = compute_confidence(0.7, "moderate", "agent_inferred")
        expected = 0.7 * 0.6 * 0.7
        assert result == pytest.approx(expected)

    def test_clamps_to_1(self):
        result = compute_confidence(1.0, "definite", "user_statement")
        assert result == pytest.approx(1.0)


class TestDetectAssertionStrength:
    def test_definite_cues(self):
        assert detect_assertion_strength("I definitely want that") == "definite"
        assert detect_assertion_strength("certainly, yes") == "definite"
        assert detect_assertion_strength("I always do that") == "definite"

    def test_uncertain_cues(self):
        assert detect_assertion_strength("maybe later") == "uncertain"
        assert detect_assertion_strength("perhaps we could") == "uncertain"
        assert detect_assertion_strength("it might work") == "uncertain"

    def test_moderate_cues(self):
        assert detect_assertion_strength("I think so") == "moderate"
        assert detect_assertion_strength("I guess that's right") == "moderate"
        assert detect_assertion_strength("probably true") == "moderate"

    def test_default_moderate(self):
        assert detect_assertion_strength("no cue words here") == "moderate"

    def test_first_match_wins(self):
        assert detect_assertion_strength("maybe I definitely think so") == "definite"


class TestSalienceGate:
    def test_entity_heuristic(self):
        gate = SalienceGate()
        assert gate.should_trigger("I talked to Alice yesterday")
        assert gate.trigger_reason("I talked to Alice yesterday") == "entity_heuristic"

    def test_first_person_assertion(self):
        gate = SalienceGate()
        assert gate.should_trigger("I feel great today")
        assert gate.trigger_reason("I feel great today") == "first_person_assertion"

    def test_temporal_weekday(self):
        gate = SalienceGate()
        assert gate.should_trigger("On Monday I started a new project")
        assert gate.trigger_reason("On Monday I started a new project") == "entity_heuristic"

    def test_temporal_yesterday(self):
        gate = SalienceGate()
        assert gate.should_trigger("yesterday was busy")
        assert gate.trigger_reason("yesterday was busy") == "temporal"

    def test_temporal_ago(self):
        gate = SalienceGate()
        assert gate.should_trigger("two days ago")
        assert gate.trigger_reason("two days ago") == "temporal"

    def test_temporal_date_pattern(self):
        gate = SalienceGate()
        assert gate.should_trigger("on 12/05/2020")
        assert gate.trigger_reason("on 12/05/2020") == "temporal"

    def test_preference_verb(self):
        gate = SalienceGate()
        assert gate.should_trigger("I like pizza")
        assert gate.trigger_reason("I like pizza") == "first_person_assertion"

    def test_preference_verb_no_preference(self):
        gate = SalienceGate()
        assert gate.should_trigger("I enjoy hiking")
        assert gate.trigger_reason("I enjoy hiking") == "preference_verb"

    def test_no_trigger_just_hello(self):
        gate = SalienceGate()
        assert not gate.should_trigger("hello")
        assert gate.trigger_reason("hello") is None

    def test_no_trigger_simple_greeting(self):
        gate = SalienceGate()
        assert not gate.should_trigger("hi how are you")
        assert gate.trigger_reason("hi how are you") is None

    def test_no_trigger_said_hello(self):
        gate = SalienceGate()
        assert not gate.should_trigger("said hello to the user")
        assert gate.trigger_reason("said hello to the user") is None


class TestExtractionPipeline:
    def test_gate_skip_returns_empty(self):
        pipeline = ExtractionPipeline(extractor=FakeTeacherExtractor())
        results = pipeline.process_turn("hello")
        assert results == []

    def test_valid_claims_accepted(self):
        claim = _make_claim(
            content="I work at Google",
            verbatim_span="I work at Google",
            extractor_confidence=0.9,
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([claim]),
        )
        results = pipeline.process_turn("I work at Google")
        assert len(results) == 1
        r = results[0]
        assert not r.rejected
        assert r.rejection_reason is None
        assert r.confidence > 0
        assert r.claim == claim

    def test_invalid_claim_no_span_match(self):
        claim = _make_claim(
            content="I work at Google",
            verbatim_span="I work at Amazon",
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([claim]),
        )
        results = pipeline.process_turn("I work at Google")
        assert len(results) == 1
        r = results[0]
        assert r.rejected
        assert r.rejection_reason == "no_span_match"
        assert r.confidence == 0.0

    def test_invalid_claim_trivial(self):
        claim = _make_claim(
            content="said hello",
            verbatim_span="said hello",
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([claim]),
        )
        results = pipeline.process_turn("I work at Google and said hello")
        assert len(results) == 1
        r = results[0]
        assert r.rejected
        assert r.rejection_reason == "trivial"

    def test_mixed_valid_invalid_split(self):
        valid = _make_claim(
            content="I work at Google",
            verbatim_span="I work at Google",
            extractor_confidence=0.9,
        )
        invalid_span = _make_claim(
            content="bad span claim",
            verbatim_span="not in source",
        )
        trivial = _make_claim(
            content="said hello",
            verbatim_span="said hello",
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([valid, invalid_span, trivial]),
        )
        results = pipeline.process_turn("I work at Google and said hello")
        assert len(results) == 3

        accepted = [r for r in results if not r.rejected]
        rejected = [r for r in results if r.rejected]

        assert len(accepted) == 1
        assert accepted[0].claim == valid

        rejected_reasons = {r.rejection_reason for r in rejected}
        assert rejected_reasons == {"no_span_match", "trivial"}

    def test_grounding_threshold_rejection(self):
        class LowScoreGroundingChecker:
            def check(self, claim_content: str, source_text: str) -> float:
                return 0.2

        claim = _make_claim(
            content="I work at Google",
            verbatim_span="I work at Google",
            extractor_confidence=0.9,
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([claim]),
            grounding_checker=LowScoreGroundingChecker(),
            grounding_threshold=0.5,
        )
        results = pipeline.process_turn("I work at Google")
        assert len(results) == 1
        r = results[0]
        assert r.rejected
        assert r.rejection_reason == "grounding_low"
        assert r.confidence == 0.0

    def test_grounding_passes_threshold(self):
        class HighScoreGroundingChecker:
            def check(self, claim_content: str, source_text: str) -> float:
                return 0.9

        claim = _make_claim(
            content="I work at Google",
            verbatim_span="I work at Google",
            extractor_confidence=0.9,
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([claim]),
            grounding_checker=HighScoreGroundingChecker(),
            grounding_threshold=0.5,
        )
        results = pipeline.process_turn("I work at Google")
        assert len(results) == 1
        r = results[0]
        assert not r.rejected
        assert r.confidence > 0

    def test_context_default_none(self):
        claim = _make_claim(
            content="I work at Google",
            verbatim_span="I work at Google",
        )
        pipeline = ExtractionPipeline(
            extractor=FakeTeacherExtractor([claim]),
        )
        results = pipeline.process_turn("I work at Google")
        assert len(results) == 1
        assert not results[0].rejected


class TestSchemaValidation:
    def test_empty_verbatim_span_rejected(self):
        with pytest.raises(ValueError, match="verbatim_span must be non-empty"):
            ExtractedClaim(
                content="test",
                tier="short_term",
                verbatim_span="",
                extractor_confidence=0.5,
            )

    def test_whitespace_only_verbatim_span_rejected(self):
        with pytest.raises(ValueError, match="verbatim_span must be non-empty"):
            ExtractedClaim(
                content="test",
                tier="short_term",
                verbatim_span="   ",
                extractor_confidence=0.5,
            )

    def test_extra_field_forbidden(self):
        with pytest.raises(ValueError):
            ExtractedClaim(
                content="test",
                tier="short_term",
                verbatim_span="test",
                extractor_confidence=0.5,
                extra_field="should not be here",
            )

    def test_extractor_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            ExtractedClaim(
                content="test",
                tier="short_term",
                verbatim_span="test",
                extractor_confidence=1.5,
            )

    def test_extractor_confidence_negative(self):
        with pytest.raises(ValueError):
            ExtractedClaim(
                content="test",
                tier="short_term",
                verbatim_span="test",
                extractor_confidence=-0.1,
            )

    def test_default_values(self):
        claim = ExtractedClaim(
            content="test",
            tier="short_term",
            verbatim_span="test",
            extractor_confidence=0.5,
        )
        assert claim.assertion_strength == "moderate"
        assert claim.source_reliability == "user_statement"
        assert claim.attribution == "user"
        assert claim.sensitive is False
        assert claim.entities == []
        assert claim.topics == []
