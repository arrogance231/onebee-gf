from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AssertionStrength = Literal["definite", "moderate", "uncertain"]

ASSERTION_STRENGTH_MULTIPLIER: dict[AssertionStrength, float] = {
    "definite": 1.0,
    "moderate": 0.6,
    "uncertain": 0.4,
}

SourceReliability = Literal["user_statement", "agent_inferred", "reflection_derived"]

SOURCE_RELIABILITY_MULTIPLIER: dict[SourceReliability, float] = {
    "user_statement": 1.0,
    "agent_inferred": 0.7,
    "reflection_derived": 0.5,
}


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    tier: Literal["short_term", "episodic", "semantic"]
    verbatim_span: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    event_time_raw: str | None = None
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    assertion_strength: AssertionStrength = "moderate"
    source_reliability: SourceReliability = "user_statement"
    attribution: Literal["user", "agent", "third_party"] = "user"
    sensitive: bool = False
    extractor_confidence: float = Field(ge=0, le=1)

    @field_validator("verbatim_span")
    @classmethod
    def _check_verbatim_span_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("verbatim_span must be non-empty")
        return v
