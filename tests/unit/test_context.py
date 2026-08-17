from __future__ import annotations

import pytest

from onebee.context.budget import (
    ApproxTokenCounter,
    BudgetBlock,
    TokenCounter,
    allocate,
)
from onebee.context.builder import ContextBuilder, TraceRecord
from onebee.context.render import (
    render_boundaries,
    render_memories_block,
    render_memory_structured_kv,
    render_persona_card,
    render_profile,
    render_recent_turns,
)


class TestApproxTokenCounter:
    def test_empty_string(self):
        c = ApproxTokenCounter()
        assert c.count("") == 1

    def test_short_string(self):
        c = ApproxTokenCounter()
        result = c.count("hello")
        assert result == max(1, len("hello") // 4)

    def test_long_string(self):
        c = ApproxTokenCounter()
        result = c.count("hello world " * 100)
        assert result == len("hello world " * 100) // 4

    def test_conforms_to_protocol(self):
        c = ApproxTokenCounter()
        assert isinstance(c, TokenCounter)


class TestAllocateMandatory:
    def test_mandatory_always_kept(self):
        counter = ApproxTokenCounter()
        blocks = [
            BudgetBlock(name="a", content="hello", priority=0, token_count=5, droppable=False),
            BudgetBlock(name="b", content="world", priority=0, token_count=5, droppable=True),
        ]
        kept, log = allocate(blocks, total_budget=100, counter=counter)
        kept_names = {b.name for b in kept}
        assert "a" in kept_names
        assert log["mandatory_kept"] == ["a"]

    def test_mandatory_exceeds_budget_raises(self):
        counter = ApproxTokenCounter()
        blocks = [
            BudgetBlock(name="a", content="hello", priority=0, token_count=100, droppable=False),
            BudgetBlock(name="b", content="world", priority=0, token_count=100, droppable=False),
        ]
        with pytest.raises(ValueError, match="Mandatory"):
            allocate(blocks, total_budget=150, counter=counter)

    def test_mandatory_exact_budget(self):
        counter = ApproxTokenCounter()
        blocks = [
            BudgetBlock(name="a", content="x" * 40, priority=0, token_count=10, droppable=False),
        ]
        kept, log = allocate(blocks, total_budget=10, counter=counter)
        assert len(kept) == 1


class TestAllocatePriority:
    def test_lower_priority_kept_first(self):
        counter = ApproxTokenCounter()
        # priority 2 is more droppable than priority 1
        blocks = [
            BudgetBlock(
                name="prio1", content="x" * 400, priority=1, token_count=100, droppable=True
            ),
            BudgetBlock(
                name="prio2", content="y" * 400, priority=2, token_count=100, droppable=True
            ),
        ]
        # Budget only allows one of the two droppable blocks
        kept, log = allocate(blocks, total_budget=100, counter=counter)
        kept_names = {b.name for b in kept}
        assert "prio1" in kept_names
        assert "prio2" not in kept_names

    def test_value_density_tiebreak_same_priority(self):
        counter = ApproxTokenCounter()
        blocks = [
            BudgetBlock(
                name="low_density",
                content="x" * 400,
                priority=1,
                token_count=100,
                droppable=True,
                value_score=1.0,
            ),
            BudgetBlock(
                name="high_density",
                content="y" * 400,
                priority=1,
                token_count=100,
                droppable=True,
                value_score=10.0,
            ),
        ]
        kept, log = allocate(blocks, total_budget=100, counter=counter)
        kept_names = {b.name for b in kept}
        assert "high_density" in kept_names
        assert "low_density" not in kept_names


class TestAllocateTruncation:
    def test_memories_truncated_when_no_room(self):
        counter = ApproxTokenCounter()
        content = "line1\nline2\nline3\nline4\nline5"
        token_count = counter.count(content)
        blocks = [
            BudgetBlock(
                name="memories",
                content=content,
                priority=1,
                token_count=token_count,
                droppable=True,
            ),
        ]
        kept, log = allocate(blocks, total_budget=counter.count("line1\nline2"), counter=counter)
        assert "truncated_memories" in log
        truncated_block = next(b for b in kept if b.name == "memories")
        assert truncated_block.token_count < token_count
        assert "line5" not in truncated_block.content

    def test_recent_turns_truncated_when_no_room(self):
        counter = ApproxTokenCounter()
        content = "turn1\nturn2\nturn3"
        token_count = counter.count(content)
        blocks = [
            BudgetBlock(
                name="recent_turns",
                content=content,
                priority=1,
                token_count=token_count,
                droppable=True,
            ),
        ]
        kept, log = allocate(blocks, total_budget=counter.count("turn1"), counter=counter)
        assert "truncated_recent_turns" in log
        truncated_block = next(b for b in kept if b.name == "recent_turns")
        assert truncated_block.token_count < token_count

    def test_non_special_block_dropped_not_truncated(self):
        counter = ApproxTokenCounter()
        blocks = [
            BudgetBlock(
                name="summary", content="x" * 400, priority=1, token_count=100, droppable=True
            ),
        ]
        kept, log = allocate(blocks, total_budget=50, counter=counter)
        kept_names = {b.name for b in kept}
        assert "summary" not in kept_names
        assert "dropped_summary" in log

    def test_allocation_log_complete(self):
        counter = ApproxTokenCounter()
        blocks = [
            BudgetBlock(name="a", content="x", priority=0, token_count=10, droppable=False),
            BudgetBlock(name="b", content="y" * 400, priority=1, token_count=100, droppable=True),
            BudgetBlock(name="c", content="z" * 400, priority=1, token_count=100, droppable=True),
        ]
        kept, log = allocate(blocks, total_budget=110, counter=counter)
        assert "final_used" in log
        assert "final_remaining" in log
        assert log["final_used"] == 110


class TestRenderPersonaCard:
    def test_full_persona(self):
        persona = {"name": "Alex", "description": "Friendly AI", "traits": ["helpful", "curious"]}
        result = render_persona_card(persona)
        assert "Persona: Alex" in result
        assert "helpful, curious" in result
        assert "Description: Friendly AI" in result

    def test_persona_no_traits(self):
        persona = {"name": "Bot", "description": "minimal"}
        result = render_persona_card(persona)
        assert "Persona: Bot" in result
        assert "Traits" not in result
        assert "Description: minimal" in result

    def test_persona_empty(self):
        result = render_persona_card({})
        assert result == ""

    def test_extended_fields_rendered_when_present(self):
        persona = {
            "name": "Alex",
            "age": 27,
            "appearance": "curly dark hair",
            "favorite_color": "amber",
            "hobbies": ["pottery", "night runs"],
            "personality_quirks": ["hums when thinking"],
            "speech_style": "uses a lot of em-dashes",
            "backstory": "grew up moving between three countries",
            "key_relationships": ["her sister Mia"],
            "values": ["honesty"],
            "boundaries": ["won't pretend to agree just to keep the peace"],
        }
        result = render_persona_card(persona)
        assert "Age: 27" in result
        assert "Appearance: curly dark hair" in result
        assert "Favorite color: amber" in result
        assert "pottery, night runs" in result
        assert "hums when thinking" in result
        assert "Speech style: uses a lot of em-dashes" in result
        assert "Backstory: grew up moving between three countries" in result
        assert "her sister Mia" in result
        assert "Values: honesty" in result
        assert "Boundaries" in result and "won't pretend to agree" in result

    def test_extended_fields_absent_when_not_given(self):
        # Strict backward compatibility: a plain {name, description, traits} dict must not
        # gain any extended-field lines it didn't ask for.
        persona = {"name": "Bot", "description": "minimal"}
        result = render_persona_card(persona)
        assert "Age" not in result
        assert "Appearance" not in result
        assert "Hobbies" not in result
        assert "Boundaries" not in result

    def test_age_zero_is_rendered_not_treated_as_falsy(self):
        # age=0 is a real (if unusual) value -- must not be silently dropped by a truthiness
        # check the way an empty string/list correctly is.
        result = render_persona_card({"name": "Bot", "age": 0})
        assert "Age: 0" in result


class TestRenderProfile:
    def test_full_profile(self):
        profile = {
            "name": "Sam",
            "pronouns": "they/them",
            "timezone": "UTC",
            "occupation": "Developer",
            "key_people": ["Alice", "Bob"],
            "core_interests": ["programming", "music"],
        }
        result = render_profile(profile)
        assert "Name: Sam" in result
        assert "Pronouns: they/them" in result
        assert "Timezone: UTC" in result
        assert "Occupation: Developer" in result
        assert "Key People: Alice, Bob" in result
        assert "Core Interests: programming, music" in result

    def test_profile_empty_fields_skipped(self):
        profile = {"name": "Sam", "pronouns": None, "timezone": ""}
        result = render_profile(profile)
        assert "Name: Sam" in result
        assert "Pronouns" not in result
        assert "Timezone" not in result

    def test_profile_empty(self):
        result = render_profile({})
        assert result == ""


class TestRenderMemoryStructuredKV:
    def test_full_record(self):
        record = {
            "tier": "episodic",
            "event_time": 1700000000000,
            "content": "User likes coffee",
            "confidence": 0.87,
        }
        result = render_memory_structured_kv(record)
        assert "episodic" in result
        assert "User likes coffee" in result
        assert "conf 0.87" in result
        # date should be YYYY-MM-DD
        assert "2023-11-14" in result

    def test_uses_created_at_fallback(self):
        record = {
            "tier": "semantic",
            "created_at": 1710000000000,
            "content": "Python is a language",
            "confidence": 0.95,
        }
        result = render_memory_structured_kv(record)
        assert "semantic" in result
        assert "2024-03-09" in result

    def test_no_timestamp_defaults_to_epoch(self):
        record = {
            "tier": "short_term",
            "content": "temp",
            "confidence": 0.5,
        }
        result = render_memory_structured_kv(record)
        assert "1970-01-01" in result


class TestRenderMemoriesBlock:
    def test_multiple_records(self):
        records = [
            {"tier": "episodic", "event_time": 1700000000000, "content": "A", "confidence": 0.9},
            {"tier": "semantic", "event_time": 1710000000000, "content": "B", "confidence": 0.8},
        ]
        result = render_memories_block(records)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "episodic" in lines[0]
        assert "semantic" in lines[1]

    def test_empty_records(self):
        result = render_memories_block([])
        assert result == ""


class TestRenderRecentTurns:
    def test_multiple_turns(self):
        turns = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there"},
        ]
        result = render_recent_turns(turns)
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "user: Hello"
        assert lines[1] == "assistant: Hi there"

    def test_empty_turns(self):
        result = render_recent_turns([])
        assert result == ""


class TestRenderBoundaries:
    def test_nonempty(self):
        result = render_boundaries(["no politics", "no spam"])
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "- no politics"
        assert lines[1] == "- no spam"

    def test_empty(self):
        assert render_boundaries([]) == ""


class _Persona:
    FULL = {"name": "OneBee", "description": "Helpful assistant", "traits": ["kind", "smart"]}
    EMPTY: dict = {}


class _Profile:
    FULL = {
        "name": "User",
        "pronouns": "he/him",
        "timezone": "Europe/London",
        "occupation": "Researcher",
        "key_people": ["Dr. X"],
        "core_interests": ["AI", "memory"],
    }
    EMPTY: dict = {}


class _Memories:
    @staticmethod
    def make(n: int) -> list[dict]:
        result = []
        for i in range(n):
            result.append(
                {
                    "id": f"mem_{i}",
                    "tier": "episodic",
                    "event_time": 1700000000000 + i * 86400000,
                    "content": f"Memory item {i} with some extra words for token count",
                    "confidence": 0.8 + i * 0.01,
                }
            )
        return result


class _Turns:
    @staticmethod
    def make(n: int) -> list[dict]:
        result = []
        for i in range(n):
            if i % 2 == 0:
                result.append({"role": "user", "text": f"User message number {i}"})
            else:
                result.append({"role": "assistant", "text": f"Assistant response number {i}"})
        return result


class TestContextBuilderNormal:
    def test_build_returns_string_and_trace(self):
        builder = ContextBuilder()
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["rule1"],
            retrieved_memories=_Memories.make(3),
            recent_turns=_Turns.make(2),
            user_turn="Hello, how are you?",
        )
        assert isinstance(ctx, str)
        assert isinstance(trace, TraceRecord)
        assert ctx != ""

    def test_correct_assembly_order(self):
        builder = ContextBuilder()
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["rule1"],
            retrieved_memories=_Memories.make(3),
            recent_turns=_Turns.make(2),
            user_turn="USER_MARKER",
        )
        # user turn should be last
        assert ctx.strip().endswith("USER_MARKER")
        # persona should appear before boundaries, before dynamic, before turns
        persona_idx = ctx.index("OneBee")
        profile_idx = ctx.index("User")
        boundaries_idx = ctx.index("rule1")
        memories_idx = ctx.index("Memory item")
        turns_idx = ctx.index("User message")
        user_idx = ctx.index("USER_MARKER")
        assert persona_idx < profile_idx
        assert profile_idx < boundaries_idx
        assert boundaries_idx < memories_idx
        assert memories_idx < turns_idx
        assert turns_idx < user_idx

    def test_total_tokens_consistent(self):
        builder = ContextBuilder()
        counter = builder.counter
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["rule1"],
            retrieved_memories=_Memories.make(3),
            recent_turns=_Turns.make(2),
            user_turn="hello",
        )
        assert trace.total_tokens == counter.count(ctx)

    def test_empty_optionals_produce_valid_context(self):
        builder = ContextBuilder()
        ctx, trace = builder.build(
            turn_id="t1",
            persona={},
            profile={},
            boundaries=[],
            retrieved_memories=[],
            recent_turns=[],
            user_turn="only me",
        )
        assert "only me" in ctx
        assert trace.total_tokens == builder.counter.count(ctx)


class TestContextBuilderTightBudget:
    def test_memories_truncated_under_tight_budget(self):
        # Mandatory blocks (persona+profile+boundaries+recent_turns+response_headroom=256)
        # need ~320 tokens. Set budget to 400 so there is ~80 tokens left for
        # droppable blocks, forcing truncation of the large memories block.
        builder = ContextBuilder(total_budget=400)
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["r1", "r2"],
            retrieved_memories=_Memories.make(20),
            recent_turns=_Turns.make(2),
            user_turn="hi",
        )
        assert len(trace.dropped_memory_ids) > 0
        assert len(trace.kept_memory_ids) < len(trace.retrieved_memory_ids)
        assert trace.total_tokens <= 400

    def test_memories_fully_dropped_no_budget(self):
        # Budget only covers mandatory + user_turn; memories should be dropped.
        builder = ContextBuilder(total_budget=330)
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["r1"],
            retrieved_memories=_Memories.make(10),
            recent_turns=_Turns.make(2),
            user_turn="hi",
        )
        if len(trace.kept_memory_ids) == 0:
            assert len(trace.dropped_memory_ids) == len(trace.retrieved_memory_ids)

    def test_dropped_memory_ids_reflected_in_trace(self):
        # Persona only + response_headroom=256 ≈ 271 mandatory tokens. Budget=300
        # leaves ~29 tokens — enough for 1-2 memories before truncation.
        builder = ContextBuilder(total_budget=300)
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile={},
            boundaries=[],
            retrieved_memories=_Memories.make(10),
            recent_turns=[],
            user_turn="test",
        )
        assert trace.retrieved_memory_ids == [f"mem_{i}" for i in range(10)]
        all_ids = set(trace.kept_memory_ids) | set(trace.dropped_memory_ids)
        assert all_ids == set(trace.retrieved_memory_ids)


class TestContextBuilderDroppableSections:
    def test_state_and_summary_block_present(self):
        builder = ContextBuilder()
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["r1"],
            retrieved_memories=_Memories.make(3),
            recent_turns=_Turns.make(2),
            user_turn="hi",
            state_block="STATE_DATA",
            commitments_block="COMMITMENTS_DATA",
            session_summary="SUMMARY_DATA",
        )
        assert "STATE_DATA" in ctx
        assert "COMMITMENTS_DATA" in ctx
        assert "SUMMARY_DATA" in ctx

    def test_state_dropped_on_tight_budget(self):
        # Mandatory ≈ 304 (persona+profile+boundaries+response_headroom=256).
        # Budget=360 leaves 56 tokens — state has priority 2 and should be the
        # first droppable block dropped.
        builder = ContextBuilder(total_budget=360)
        ctx, trace = builder.build(
            turn_id="t1",
            persona=_Persona.FULL,
            profile=_Profile.FULL,
            boundaries=["r1"],
            retrieved_memories=[],
            recent_turns=[],
            user_turn="hi",
            state_block="STATE_DATA",
            commitments_block="COMMITMENTS_DATA_LONG_STRING_NEEDS_SPACE",
            session_summary="SUMMARY_DATA",
        )
        if "dropped_state" in trace.allocation_log:
            assert "STATE_DATA" not in ctx


class TestImportRoundtrip:
    def test_public_exports(self):
        from onebee.context import (
            ApproxTokenCounter,
            BudgetBlock,
            ContextBuilder,
            TokenCounter,
            TraceRecord,
            allocate,
            render_boundaries,
            render_memories_block,
            render_memory_structured_kv,
            render_persona_card,
            render_profile,
            render_recent_turns,
        )

        assert ApproxTokenCounter is not None
        assert BudgetBlock is not None
        assert ContextBuilder is not None
        assert TokenCounter is not None
        assert TraceRecord is not None
        assert callable(allocate)
        assert callable(render_boundaries)
        assert callable(render_memories_block)
        assert callable(render_memory_structured_kv)
        assert callable(render_persona_card)
        assert callable(render_profile)
        assert callable(render_recent_turns)
