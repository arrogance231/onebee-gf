from __future__ import annotations

import random as stdlib_random
from typing import Protocol, runtime_checkable

from onebee.data.personas import ConversationTurn, FactSheetEntry, Persona


@runtime_checkable
class TeacherClient(Protocol):
    def generate_conversation(
        self,
        persona: Persona,
        session_index: int,
        target_turns: int,
        facts_to_reveal: list[FactSheetEntry],
    ) -> list[ConversationTurn]:
        ...

    def generate_probe(
        self,
        persona: Persona,
        fact: FactSheetEntry,
        category: str,
    ) -> str:
        ...


FACTUAL_TEMPLATES = [
    "What is {subject}'s {predicate}?",
    "What {predicate} does {subject} have?",
    "Tell me {subject}'s {predicate}.",
]

PREFERENCE_TEMPLATES = [
    "What is {subject}'s favorite {predicate}?",
    "What does {subject} prefer for {predicate}?",
    "Which {predicate} does {subject} like best?",
]

EPISODIC_TEMPLATES = [
    "What happened when {subject} {predicate}?",
    "Describe {subject}'s experience with {predicate}.",
    "What did {subject} do regarding {predicate}?",
]

TEMPORAL_TEMPLATES = [
    "When did {subject} {predicate}?",
    "When is {subject}'s {predicate}?",
    "At what time did {subject} {predicate}?",
]

UNABANSWERABLE_PREDICATES = [
    "pet's name",
    "sibling count",
    "middle name",
    "shoe size",
    "childhood nickname",
    "favorite movie",
    "first car model",
    "last vacation destination",
    "phone number",
    "email address",
]

ASSISTANT_REPLIES = [
    "That is interesting, {name}. Thanks for sharing!",
    "I see! Good to know, {name}.",
    "Got it, {name}. I will keep that in mind.",
    "Thanks for telling me, {name}.",
    "Noted, {name}!",
    "That makes sense, {name}.",
]

USER_FACT_STATEMENTS = {
    "factual": [
        "My {predicate} is {object}.",
        "I {predicate} {object}.",
        "I have {predicate} {object}.",
    ],
    "preference": [
        "My favorite {predicate} is {object}.",
        "I prefer {object} when it comes to {predicate}.",
        "I like {object} for {predicate}.",
    ],
    "episodic": [
        "Recently, I {predicate} {object}.",
        "I remember when I {predicate} {object}.",
        "Last time, I {predicate} {object}.",
    ],
    "temporal": [
        "My {predicate} is {object}.",
        "{predicate} happens on {object} for me.",
        "I have {predicate} on {object}.",
    ],
}


class FixtureTeacherClient:
    def __init__(self, seed: int = 1337):
        self._rng = stdlib_random.Random(seed)

    def generate_conversation(
        self,
        persona: Persona,
        session_index: int,
        target_turns: int,
        facts_to_reveal: list[FactSheetEntry],
    ) -> list[ConversationTurn]:
        session_id = f"{persona.persona_id}_s{session_index:03d}"
        base_ts = session_index * 100000
        turns: list[ConversationTurn] = []

        facts_per_user_turn = max(1, len(facts_to_reveal) // max(1, target_turns // 2))
        fact_idx = 0
        turn_idx = 0

        while fact_idx < len(facts_to_reveal) and turn_idx + 1 < target_turns:
            batch = facts_to_reveal[fact_idx : fact_idx + facts_per_user_turn]
            if not batch:
                break
            fact_idx += len(batch)

            statements = []
            for fact in batch:
                templates = USER_FACT_STATEMENTS.get(fact.category, USER_FACT_STATEMENTS["factual"])
                tmpl = self._rng.choice(templates)
                statements.append(tmpl.format(predicate=fact.predicate, object=fact.object))

            user_text = " ".join(statements)
            turn_id_user = f"{session_id}_t{turn_idx:03d}"
            ts = base_ts + turn_idx * 60
            turns.append(
                ConversationTurn(
                    turn_id=turn_id_user,
                    session_id=session_id,
                    role="user",
                    text=user_text,
                    ts=ts,
                    revealed_fact_ids=[f.fact_id for f in batch],
                )
            )
            turn_idx += 1

            reply_tmpl = self._rng.choice(ASSISTANT_REPLIES)
            assistant_text = reply_tmpl.format(name=persona.name)
            turn_id_asst = f"{session_id}_t{turn_idx:03d}"
            ts = base_ts + turn_idx * 60
            turns.append(
                ConversationTurn(
                    turn_id=turn_id_asst,
                    session_id=session_id,
                    role="assistant",
                    text=assistant_text,
                    ts=ts,
                    revealed_fact_ids=[],
                )
            )
            turn_idx += 1

        while turn_idx < target_turns:
            if turn_idx % 2 == 0:
                turn_id = f"{session_id}_t{turn_idx:03d}"
                ts = base_ts + turn_idx * 60
                turns.append(
                    ConversationTurn(
                        turn_id=turn_id,
                        session_id=session_id,
                        role="user",
                        text="How is the weather today?",
                        ts=ts,
                        revealed_fact_ids=[],
                    )
                )
            else:
                turn_id = f"{session_id}_t{turn_idx:03d}"
                ts = base_ts + turn_idx * 60
                turns.append(
                    ConversationTurn(
                        turn_id=turn_id,
                        session_id=session_id,
                        role="assistant",
                        text="The weather looks nice today!",
                        ts=ts,
                        revealed_fact_ids=[],
                    )
                )
            turn_idx += 1

        return turns

    def generate_probe(
        self,
        persona: Persona,
        fact: FactSheetEntry,
        category: str,
    ) -> str:
        if category == "unanswerable":
            predicates = self._rng.sample(
                UNABANSWERABLE_PREDICATES,
                k=min(len(UNABANSWERABLE_PREDICATES), 3),
            )
            predicate = self._rng.choice(predicates)
            return f"What is {persona.name}'s {predicate}?"

        cat_templates: dict[str, list[str]] = {
            "factual": FACTUAL_TEMPLATES,
            "preference": PREFERENCE_TEMPLATES,
            "episodic": EPISODIC_TEMPLATES,
            "temporal": TEMPORAL_TEMPLATES,
            "outdated_fact": FACTUAL_TEMPLATES,
            "distractor": FACTUAL_TEMPLATES,
            "continuity": EPISODIC_TEMPLATES,
        }

        templates = cat_templates.get(category, FACTUAL_TEMPLATES)
        tmpl = self._rng.choice(templates)
        return tmpl.format(subject=persona.name, predicate=fact.predicate, object=fact.object)
