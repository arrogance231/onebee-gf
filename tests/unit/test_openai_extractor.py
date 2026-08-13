from __future__ import annotations

import json
import sys
import types

import pytest

from onebee.memory.extraction import OpenAITeacherExtractor
from onebee.memory.extraction.schema import ExtractedClaim


class FakeResponse:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def _install_fake_openai(monkeypatch, completions):
    class FakeBadRequestError(Exception):
        pass

    class FakeChat:
        pass

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    FakeChat.completions = completions

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAIClient
    fake_openai.BadRequestError = FakeBadRequestError
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return FakeBadRequestError


def _claim_dict(**overrides):
    base = {
        "content": "I live in NYC.",
        "tier": "semantic",
        "verbatim_span": "I live in NYC",
        "entities": ["NYC"],
        "topics": ["home"],
        "assertion_strength": "definite",
        "source_reliability": "user_statement",
        "attribution": "user",
        "sensitive": False,
        "extractor_confidence": 0.9,
    }
    base.update(overrides)
    return base


def _two_claims_payload():
    return json.dumps(
        {
            "claims": [
                _claim_dict(),
                _claim_dict(
                    content="My favorite food is pizza.",
                    verbatim_span="my favorite food is pizza",
                    entities=["pizza"],
                    topics=["food"],
                ),
            ]
        }
    )


class TestOpenAITeacherExtractorConstruction:
    def test_construction_needs_no_openai_package_or_env(self):
        assert "openai" not in sys.modules
        extractor = OpenAITeacherExtractor(model="gpt-4o")
        assert extractor.model == "gpt-4o"
        assert extractor.api_key is None
        assert extractor.base_url is None
        assert extractor.temperature == 0.2
        assert "openai" not in sys.modules

    def test_api_key_and_base_url_are_stored(self):
        extractor = OpenAITeacherExtractor(
            model="gpt-4o", api_key="sk-test", base_url="http://x", temperature=0.1
        )
        assert extractor.api_key == "sk-test"
        assert extractor.base_url == "http://x"
        assert extractor.temperature == 0.1


class TestExtract:
    def test_parses_two_valid_claims(self, monkeypatch):
        turn_text = "I live in NYC and my favorite food is pizza."
        completions = FakeCompletions([_two_claims_payload()])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        claims = extractor.extract(turn_text, {})

        assert len(claims) == 2
        assert all(isinstance(c, ExtractedClaim) for c in claims)
        assert claims[0].content == "I live in NYC."
        assert claims[0].verbatim_span == "I live in NYC"
        assert claims[0].tier == "semantic"
        assert claims[0].assertion_strength == "definite"
        assert claims[0].attribution == "user"
        assert claims[0].extractor_confidence == 0.9
        assert claims[1].content == "My favorite food is pizza."
        assert claims[1].verbatim_span == "my favorite food is pizza"
        assert completions.calls[0]["temperature"] == 0.2

    def test_handles_empty_context_dict(self, monkeypatch):
        turn_text = "I live in NYC."
        completions = FakeCompletions([json.dumps({"claims": [_claim_dict()]})])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        claims = extractor.extract(turn_text, {})

        assert len(claims) == 1
        assert "persona_name" not in completions.calls[0]["messages"][1]["content"]

    def test_passes_context_through_when_present(self, monkeypatch):
        turn_text = "I moved here last week."
        completions = FakeCompletions([json.dumps({"claims": []})])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        extractor.extract(turn_text, {"persona_name": "Alice", "recent_turns": "Some context."})

        user_content = completions.calls[0]["messages"][1]["content"]
        assert "Alice" in user_content
        assert "Some context." in user_content

    def test_drops_single_malformed_claim_keeps_valid_one(self, monkeypatch):
        turn_text = "I live in NYC and my favorite food is pizza."
        payload = json.dumps(
            {
                "claims": [
                    _claim_dict(),
                    {
                        "content": "My favorite food is pizza.",
                        "tier": "semantic",
                    },
                ]
            }
        )
        completions = FakeCompletions([payload])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        claims = extractor.extract(turn_text, {})

        assert len(claims) == 1
        assert claims[0].content == "I live in NYC."

    def test_empty_claims_returns_empty_list(self, monkeypatch):
        completions = FakeCompletions([json.dumps({"claims": []})])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        claims = extractor.extract("Hi, how are you?", {})

        assert claims == []

    def test_malformed_json_retries_with_repair_then_succeeds(self, monkeypatch):
        valid = json.dumps({"claims": [_claim_dict()]})
        completions = FakeCompletions(["this is not json", valid])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        claims = extractor.extract("I live in NYC.", {})

        assert len(claims) == 1
        assert len(completions.calls) == 2
        repair_messages = completions.calls[1]["messages"]
        assert repair_messages[-2]["role"] == "assistant"
        assert repair_messages[-2]["content"] == "this is not json"
        assert repair_messages[-1]["role"] == "user"
        assert "not valid JSON" in repair_messages[-1]["content"]

    def test_still_malformed_json_raises_runtime_error(self, monkeypatch):
        completions = FakeCompletions(["not json", "still not json"])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        with pytest.raises(RuntimeError, match="unparseable response"):
            extractor.extract("I live in NYC.", {})

    def test_retries_without_temperature_when_unsupported(self, monkeypatch):
        completions = FakeCompletions([])
        err_cls = _install_fake_openai(monkeypatch, completions)
        valid = json.dumps({"claims": [_claim_dict()]})
        completions._responses = [
            err_cls("Unsupported value: 'temperature' does not support 0 with this model."),
            valid,
        ]

        extractor = OpenAITeacherExtractor(model="gpt-5.6-luna", api_key="sk-test")
        claims = extractor.extract("I live in NYC.", {})

        assert len(claims) == 1
        assert len(completions.calls) == 2
        assert "temperature" in completions.calls[0]
        assert "temperature" not in completions.calls[1]
        assert extractor._temperature_unsupported is True

        # Second call for the same instance should skip the failing attempt entirely.
        completions._responses = [valid]
        completions.calls.clear()
        extractor.extract("I love hiking.", {})
        assert len(completions.calls) == 1
        assert "temperature" not in completions.calls[0]

    def test_missing_claims_array_raises_runtime_error(self, monkeypatch):
        completions = FakeCompletions([json.dumps({"foo": "bar"})])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        with pytest.raises(RuntimeError, match="missing 'claims' array"):
            extractor.extract("I live in NYC.", {})


class TestTeacherExtractorProtocol:
    def test_satisfies_protocol(self, monkeypatch):
        turn_text = "I live in NYC."
        completions = FakeCompletions([json.dumps({"claims": [_claim_dict()]})])
        _install_fake_openai(monkeypatch, completions)

        extractor = OpenAITeacherExtractor(model="gpt-4o")
        assert hasattr(extractor, "extract")
        result = extractor.extract(turn_text, {})
        assert isinstance(result, list)
        assert all(isinstance(c, ExtractedClaim) for c in result)
