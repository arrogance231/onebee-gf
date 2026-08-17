from __future__ import annotations

from pydantic import BaseModel

# Distinct from onebee.data.personas.Persona, which represents the USER's synthetic identity
# for PMB benchmark construction. This is the COMPANION's own identity -- always-injected
# context (like Tier 5 user-profile memory), not retrieved, and not something the benchmark
# probes are testing recall of. The point of this schema is to give the companion enough of an
# actual human-identity surface (not just name/description/traits) that persona-consistency
# evaluation (PCS, PCS-stylometric, the register-match eval in emotional_range.py) has
# something real to hold the model to -- see docs/research_questions.md's persona-card design
# note for the full rationale.


class CompanionPersona(BaseModel):
    name: str
    description: str = ""
    traits: list[str] = []

    # Identity surface -- the fields a real person would have, not a feature list.
    age: int | None = None
    appearance: str = ""
    favorite_color: str = ""
    hobbies: list[str] = []
    personality_quirks: list[str] = []
    speech_style: str = ""  # e.g. "uses a lot of em-dashes, rarely uses emoji, calls you 'hey'"
    backstory: str = ""
    key_relationships: list[str] = []  # e.g. "her sister Mia, who she talks to every Sunday"
    values: list[str] = []
    boundaries: list[str] = []  # things she won't do/say, held even under pressure

    def to_dict(self) -> dict:
        """Plain-dict form for render_persona_card, which stays dict-based for backward
        compatibility with existing callers (context builder, probes) that pass plain dicts."""
        return self.model_dump(exclude_defaults=True)
