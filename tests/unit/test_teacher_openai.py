from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from onebee.data.personas import ConversationTurn, FactSheetEntry, Persona
from onebee.data.teacher import OpenAITeacherClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def _mk_persona() -> Persona:
    return Persona(
        persona_id="p000",
        name="Alice",
        description="Alice is a friendly and organized person.",
        traits=["friendly", "organized"],
        fact_sheet=[
            FactSheetEntry(
                fact_id="p000_f000",
                subject="p000",
                predicate="lives in",
                object="NYC",
                category="factual",
            ),
            FactSheetEntry(
                fact_id="p000_f001",
                subject="p000",
                predicate="favorite food",
                object="pizza",
                category="preference",
            ),
        ],
    )


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


class TestOpenAITeacherConstruction:
    def test_construction_needs_no_openai_package_or_env(self):
        assert "openai" not in sys.modules
        teacher = OpenAITeacherClient(model="gpt-4o")
        assert teacher.model == "gpt-4o"
        assert teacher.api_key is None
        assert teacher.base_url is None
        assert teacher.temperature == 0.9
        assert "openai" not in sys.modules

    def test_api_key_and_base_url_are_stored(self):
        teacher = OpenAITeacherClient(
            model="gpt-4o", api_key="sk-test", base_url="http://x", temperature=0.7
        )
        assert teacher.api_key == "sk-test"
        assert teacher.base_url == "http://x"
        assert teacher.temperature == 0.7


class TestGenerateConversation:
    def test_builds_turns_from_well_formed_json(self, monkeypatch):
        payload = {
            "turns": [
                {
                    "role": "user",
                    "text": "Hi! I just moved to NYC.",
                    "revealed_fact_ids": ["p000_f000"],
                },
                {
                    "role": "assistant",
                    "text": "Welcome! That is exciting.",
                    "revealed_fact_ids": [],
                },
                {
                    "role": "user",
                    "text": "And my favorite food is pizza.",
                    "revealed_fact_ids": ["p000_f001"],
                },
            ]
        }
        completions = FakeCompletions([json.dumps(payload)])
        _install_fake_openai(monkeypatch, completions)

        teacher = OpenAITeacherClient(model="gpt-4o")
        persona = _mk_persona()
        turns = teacher.generate_conversation(persona, 2, 6, persona.fact_sheet)

        assert len(turns) == 3
        assert all(isinstance(t, ConversationTurn) for t in turns)
        assert [t.turn_id for t in turns] == ["p000_s002_t000", "p000_s002_t001", "p000_s002_t002"]
        assert all(t.session_id == "p000_s002" for t in turns)
        assert [t.role for t in turns] == ["user", "assistant", "user"]
        assert turns[0].text == "Hi! I just moved to NYC."
        assert turns[0].revealed_fact_ids == ["p000_f000"]
        assert turns[1].revealed_fact_ids == []
        assert turns[0].ts == 200000
        assert turns[1].ts == 200060
        assert turns[2].ts == 200120
        assert completions.calls[0]["temperature"] == 0.9

    def test_malformed_json_retries_with_repair_then_succeeds(self, monkeypatch):
        valid = json.dumps(
            {
                "turns": [
                    {
                        "role": "user",
                        "text": "I live in NYC.",
                        "revealed_fact_ids": ["p000_f000"],
                    }
                ]
            }
        )
        completions = FakeCompletions(["this is not json", valid])
        _install_fake_openai(monkeypatch, completions)

        teacher = OpenAITeacherClient(model="gpt-4o")
        turns = teacher.generate_conversation(_mk_persona(), 0, 4, _mk_persona().fact_sheet)

        assert len(turns) == 1
        assert turns[0].text == "I live in NYC."
        assert len(completions.calls) == 2
        repair_messages = completions.calls[1]["messages"]
        assert repair_messages[-2]["role"] == "assistant"
        assert repair_messages[-2]["content"] == "this is not json"
        assert repair_messages[-1]["role"] == "user"
        assert "not valid JSON" in repair_messages[-1]["content"]

    def test_still_malformed_json_raises_runtime_error(self, monkeypatch):
        completions = FakeCompletions(["not json", "still not json"])
        _install_fake_openai(monkeypatch, completions)

        teacher = OpenAITeacherClient(model="gpt-4o")
        with pytest.raises(RuntimeError, match="unparseable response"):
            teacher.generate_conversation(_mk_persona(), 0, 4, _mk_persona().fact_sheet)

    def test_retries_without_temperature_when_unsupported(self, monkeypatch):
        completions = FakeCompletions([])
        err_cls = _install_fake_openai(monkeypatch, completions)
        valid = json.dumps(
            {
                "turns": [
                    {
                        "role": "user",
                        "text": "I live in NYC.",
                        "revealed_fact_ids": ["p000_f000"],
                    }
                ]
            }
        )
        completions._responses = [
            err_cls("Unsupported value: 'temperature' does not support 0 with this model."),
            valid,
        ]

        teacher = OpenAITeacherClient(model="gpt-5.6-luna", api_key="sk-test")
        turns = teacher.generate_conversation(_mk_persona(), 0, 4, _mk_persona().fact_sheet)

        assert len(turns) == 1
        assert len(completions.calls) == 2
        assert "temperature" in completions.calls[0]
        assert "temperature" not in completions.calls[1]
        assert teacher._temperature_unsupported is True

        # Second call for the same instance should skip the failing attempt entirely.
        completions._responses = [valid]
        completions.calls.clear()
        teacher.generate_conversation(_mk_persona(), 0, 4, _mk_persona().fact_sheet)
        assert len(completions.calls) == 1
        assert "temperature" not in completions.calls[0]

    def test_invalid_turn_structure_raises_runtime_error(self, monkeypatch):
        payload = json.dumps(
            {"turns": [{"role": "system", "text": "nope", "revealed_fact_ids": []}]}
        )
        completions = FakeCompletions([payload])
        _install_fake_openai(monkeypatch, completions)

        teacher = OpenAITeacherClient(model="gpt-4o")
        with pytest.raises(RuntimeError, match="malformed turn"):
            teacher.generate_conversation(_mk_persona(), 0, 4, _mk_persona().fact_sheet)


class TestGenerateProbe:
    def test_returns_stripped_text(self, monkeypatch):
        completions = FakeCompletions(['  "What city does Alice live in?"  '])
        _install_fake_openai(monkeypatch, completions)

        teacher = OpenAITeacherClient(model="gpt-4o")
        persona = _mk_persona()
        question = teacher.generate_probe(persona, persona.fact_sheet[0], "factual")

        assert question == "What city does Alice live in?"
        assert completions.calls[0]["temperature"] == 0.9

    def test_retries_without_temperature_when_unsupported(self, monkeypatch):
        completions = FakeCompletions([])
        err_cls = _install_fake_openai(monkeypatch, completions)
        completions._responses = [
            err_cls("Unsupported value: 'temperature' does not support 0 with this model."),
            "What city does Alice live in?",
        ]

        teacher = OpenAITeacherClient(model="gpt-5.6-luna", api_key="sk-test")
        persona = _mk_persona()
        question = teacher.generate_probe(persona, persona.fact_sheet[0], "factual")

        assert question == "What city does Alice live in?"
        assert len(completions.calls) == 2
        assert "temperature" in completions.calls[0]
        assert "temperature" not in completions.calls[1]
        assert teacher._temperature_unsupported is True


class TestBuildPmbCliHelp:
    def test_openai_teacher_help_works_without_openai_installed(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/build_pmb.py"),
                "--teacher",
                "openai",
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--teacher" in result.stdout
        assert "--teacher-model" in result.stdout
        assert "--teacher-temperature" in result.stdout
        assert "openai" in result.stdout
