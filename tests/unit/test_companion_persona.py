from __future__ import annotations

from onebee.data.companion_persona import CompanionPersona


class TestCompanionPersona:
    def test_minimal_construction(self):
        p = CompanionPersona(name="Alex")
        assert p.name == "Alex"
        assert p.description == ""
        assert p.traits == []
        assert p.age is None

    def test_full_construction(self):
        p = CompanionPersona(
            name="Alex",
            description="warm and curious",
            traits=["optimistic", "playful"],
            age=27,
            appearance="curly dark hair, always wearing something green",
            favorite_color="amber",
            hobbies=["pottery", "night runs"],
            personality_quirks=["hums when thinking", "always finishes other people's sentences"],
            speech_style="uses a lot of em-dashes, rarely uses emoji",
            backstory="grew up moving between three countries as a kid",
            key_relationships=["her sister Mia, who she talks to every Sunday"],
            values=["honesty", "showing up for people"],
            boundaries=["won't pretend to agree just to keep the peace"],
        )
        assert p.age == 27
        assert "pottery" in p.hobbies
        assert "her sister Mia, who she talks to every Sunday" in p.key_relationships

    def test_to_dict_excludes_defaults(self):
        p = CompanionPersona(name="Alex", age=27)
        d = p.to_dict()
        assert d == {"name": "Alex", "age": 27}

    def test_to_dict_includes_set_fields(self):
        p = CompanionPersona(name="Alex", hobbies=["reading"])
        d = p.to_dict()
        assert d["hobbies"] == ["reading"]
        assert "appearance" not in d
