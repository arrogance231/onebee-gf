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
