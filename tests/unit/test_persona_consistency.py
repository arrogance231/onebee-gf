from __future__ import annotations

from onebee.evaluation.graders.judge import FakeJudge
from onebee.evaluation.metrics.persona_consistency import (
    extract_stylometric_features,
    pcs,
    pcs_judge_score,
    pcs_stylometric,
    stylometric_drift,
    word_frequency_profile,
)


class TestExtractStylometricFeatures:
    def test_basic_sentence(self):
        f = extract_stylometric_features("Hello there. How are you?")
        assert f.mean_sentence_length > 0
        assert f.question_rate > 0

    def test_empty_string_does_not_crash(self):
        f = extract_stylometric_features("")
        assert f.mean_sentence_length == 0.0
        assert f.mean_word_length == 0.0

    def test_exclamations_counted(self):
        f = extract_stylometric_features("Wow! Amazing! Great!")
        assert f.exclamation_rate > 0

    def test_filler_words_detected(self):
        f = extract_stylometric_features("I just really honestly think so.")
        assert f.filler_word_rate > 0

    def test_type_token_ratio_bounds(self):
        f = extract_stylometric_features("the the the the cat")
        assert 0.0 < f.type_token_ratio <= 1.0


class TestPcsStylometric:
    def test_identical_responses_are_perfectly_consistent(self):
        responses = [
            "I really love spending time with you.",
            "I really love spending time with you.",
            "I really love spending time with you.",
        ]
        score = pcs_stylometric(responses)
        assert score == 1.0

    def test_single_response_returns_one(self):
        assert pcs_stylometric(["Hello there."]) == 1.0

    def test_empty_list_returns_one(self):
        assert pcs_stylometric([]) == 1.0

    def test_wildly_different_styles_score_lower_than_similar_styles(self):
        similar = [
            "I love talking with you about your day.",
            "I love hearing about your day and your plans.",
            "I love our conversations about your day-to-day life.",
        ]
        wildly_different = [
            "Hi.",
            (
                "Well, honestly, I have been thinking about this for quite a "
                "long, long time now, and I really do think that, all things "
                "considered, it is a genuinely fascinating and multifaceted "
                "topic worth exploring together at great length!"
            ),
            "OK?!?!",
        ]
        similar_score = pcs_stylometric(similar)
        different_score = pcs_stylometric(wildly_different)
        assert similar_score > different_score

    def test_score_bounded_zero_to_one(self):
        responses = ["Short.", "This is a considerably longer sentence with more words in it."]
        score = pcs_stylometric(responses)
        assert 0.0 <= score <= 1.0


class TestStylometricDrift:
    def test_identical_sets_have_no_drift(self):
        a = ["I love you.", "You are wonderful."]
        b = ["I love you.", "You are wonderful."]
        assert stylometric_drift(a, b) == 1.0

    def test_empty_input_returns_one(self):
        assert stylometric_drift([], ["hi"]) == 1.0
        assert stylometric_drift(["hi"], []) == 1.0

    def test_very_different_sets_drift_more_than_similar_sets(self):
        baseline = ["I really love spending time with you today."] * 3
        similar = ["I really love spending time with you tomorrow."] * 3
        very_different = ["Hi."] * 3
        similar_drift = stylometric_drift(baseline, similar)
        different_drift = stylometric_drift(baseline, very_different)
        assert similar_drift > different_drift


class TestPcsJudgeScore:
    def test_uses_fake_judge_deterministically(self):
        persona = {"name": "Robin", "description": "warm and playful", "traits": ["kind"]}
        judge = FakeJudge()
        # FakeJudge scores by word count / 10, capped at 5.0 -- a 50-word response
        # should score at the cap.
        long_response = " ".join(["word"] * 60)
        score = pcs_judge_score(persona, "How are you?", long_response, judge)
        assert score == 1.0

    def test_score_bounded_zero_to_one(self):
        persona = {"name": "Robin", "description": "warm", "traits": []}
        judge = FakeJudge()
        score = pcs_judge_score(persona, "Hi", "ok", judge)
        assert 0.0 <= score <= 1.0

    def test_no_traits_or_description_does_not_crash(self):
        judge = FakeJudge()
        score = pcs_judge_score({"name": "X"}, "Hi", "hello there", judge)
        assert 0.0 <= score <= 1.0


class TestPcs:
    def test_aggregates_multiple_turns(self):
        persona = {"name": "Robin", "description": "warm", "traits": ["kind"]}
        judge = FakeJudge()
        turns = [
            ("How are you?", " ".join(["word"] * 60)),  # caps at 5.0 -> 1.0
            ("What's up?", ""),  # 0 words -> 0.0
        ]
        score = pcs(persona, turns, judge)
        assert score == 0.5

    def test_empty_turns_returns_zero_not_one(self):
        # An empty eval set should not silently look like a perfect score.
        persona = {"name": "Robin", "description": "warm", "traits": []}
        judge = FakeJudge()
        assert pcs(persona, [], judge) == 0.0


class TestWordFrequencyProfile:
    def test_counts_words_case_insensitively(self):
        profile = word_frequency_profile(["Cat cat CAT dog"])
        assert profile["cat"] == 3
        assert profile["dog"] == 1

    def test_respects_top_n(self):
        responses = ["a a a b b c"]
        profile = word_frequency_profile(responses, top_n=2)
        assert len(profile) == 2
        assert profile["a"] == 3

    def test_empty_input(self):
        assert word_frequency_profile([]) == {}
