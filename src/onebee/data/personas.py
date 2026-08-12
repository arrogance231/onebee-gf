from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FactSheetEntry(BaseModel):
    fact_id: str
    subject: str
    predicate: str
    object: str
    category: Literal["factual", "preference", "episodic", "temporal"]


class Persona(BaseModel):
    persona_id: str
    name: str
    description: str
    traits: list[str]
    fact_sheet: list[FactSheetEntry]


class ConversationTurn(BaseModel):
    turn_id: str
    session_id: str
    role: Literal["user", "assistant"]
    text: str
    ts: int
    revealed_fact_ids: list[str] = []


class PersonaSession(BaseModel):
    session_id: str
    persona_id: str
    turns: list[ConversationTurn]


class PersonaCorpus(BaseModel):
    persona: Persona
    sessions: list[PersonaSession]
