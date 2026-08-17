from __future__ import annotations

from datetime import datetime, timezone


def render_persona_card(persona: dict) -> str:
    name = persona.get("name", "")
    description = persona.get("description", "")
    traits = persona.get("traits", [])

    lines = []
    if name:
        lines.append(f"Persona: {name}")
    if traits:
        lines.append(f"Traits: {', '.join(traits)}")
    if description:
        lines.append(f"Description: {description}")

    # Extended companion-identity fields (CompanionPersona schema) -- all optional, rendered
    # only when present, so this stays a strict superset of the original name/traits/description
    # card and every existing caller passing a plain {name, description, traits} dict is
    # unaffected.
    age = persona.get("age")
    if age is not None:
        lines.append(f"Age: {age}")
    appearance = persona.get("appearance", "")
    if appearance:
        lines.append(f"Appearance: {appearance}")
    favorite_color = persona.get("favorite_color", "")
    if favorite_color:
        lines.append(f"Favorite color: {favorite_color}")
    hobbies = persona.get("hobbies", [])
    if hobbies:
        lines.append(f"Hobbies: {', '.join(hobbies)}")
    personality_quirks = persona.get("personality_quirks", [])
    if personality_quirks:
        lines.append(f"Personality quirks: {', '.join(personality_quirks)}")
    speech_style = persona.get("speech_style", "")
    if speech_style:
        lines.append(f"Speech style: {speech_style}")
    backstory = persona.get("backstory", "")
    if backstory:
        lines.append(f"Backstory: {backstory}")
    key_relationships = persona.get("key_relationships", [])
    if key_relationships:
        lines.append(f"Key relationships: {', '.join(key_relationships)}")
    values = persona.get("values", [])
    if values:
        lines.append(f"Values: {', '.join(values)}")
    boundaries = persona.get("boundaries", [])
    if boundaries:
        lines.append(f"Boundaries (held even under pressure): {', '.join(boundaries)}")

    return "\n".join(lines)


def render_profile(profile: dict) -> str:
    fields = [
        "name",
        "pronouns",
        "timezone",
        "occupation",
        "key_people",
        "core_interests",
    ]
    lines = []
    for field in fields:
        value = profile.get(field)
        if value is None or value == "" or value == []:
            continue
        display_name = field.replace("_", " ").title()
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"{display_name}: {value}")
    return "\n".join(lines)


def _date_str_from_record(record: dict) -> str:
    ts = record.get("event_time") or record.get("created_at", 0)
    if ts is None:
        ts = 0
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def render_memory_structured_kv(record: dict) -> str:
    tier = record.get("tier", "?")
    date_str = _date_str_from_record(record)
    content = record.get("content", "")
    confidence = record.get("confidence", 0.0)
    return f"{tier} | {date_str} | {content} | conf {confidence:.2f}"


def render_memories_block(records: list[dict]) -> str:
    if not records:
        return ""
    return "\n".join(render_memory_structured_kv(r) for r in records)


def render_recent_turns(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = t.get("role", "unknown")
        text = t.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def render_boundaries(boundaries: list[str]) -> str:
    if not boundaries:
        return ""
    return "\n".join(f"- {b}" for b in boundaries)
