from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from onebee.context.budget import (
    DEFAULT_BUDGET_ALLOCATION,
    TOTAL_BUDGET_TOKENS,
    ApproxTokenCounter,
    BudgetBlock,
    TokenCounter,
    allocate,
)
from onebee.context.render import (
    render_boundaries,
    render_memories_block,
    render_persona_card,
    render_profile,
    render_recent_turns,
)


class TraceRecord(BaseModel):
    turn_id: str
    timestamp_ms: int
    persona_tokens: int
    profile_tokens: int
    boundaries_tokens: int
    memories_tokens: int
    recent_turns_tokens: int
    total_tokens: int
    retrieved_memory_ids: list[str]
    kept_memory_ids: list[str]
    dropped_memory_ids: list[str]
    allocation_log: dict
    assembled_context: str


class ContextBuilder:
    def __init__(
        self,
        counter: TokenCounter | None = None,
        total_budget: int = TOTAL_BUDGET_TOKENS,
        allocation: dict[str, int] | None = None,
    ) -> None:
        self.counter = counter if counter is not None else ApproxTokenCounter()
        self.total_budget = total_budget
        self.allocation = allocation if allocation is not None else DEFAULT_BUDGET_ALLOCATION

    def build(
        self,
        turn_id: str,
        persona: dict,
        profile: dict,
        boundaries: list[str],
        retrieved_memories: list[dict],
        recent_turns: list[dict],
        user_turn: str,
        state_block: str = "",
        commitments_block: str = "",
        session_summary: str = "",
    ) -> tuple[str, TraceRecord]:
        counter = self.counter
        total_budget = self.total_budget

        # Render each section
        persona_rendered = render_persona_card(persona)
        profile_rendered = render_profile(profile)
        boundaries_rendered = render_boundaries(boundaries)
        memories_rendered = render_memories_block(retrieved_memories)
        recent_turns_rendered = render_recent_turns(recent_turns)

        user_turn_token_count = counter.count(user_turn)

        # Build BudgetBlocks with priorities from the spec.
        # Lower priority number = higher importance.
        # Non-droppable blocks: persona, profile, boundaries, recent_turns,
        # response_headroom (priority 0)
        blocks: list[BudgetBlock] = [
            BudgetBlock(
                name="persona",
                content=persona_rendered,
                priority=0,
                token_count=counter.count(persona_rendered),
                droppable=False,
            ),
            BudgetBlock(
                name="profile",
                content=profile_rendered,
                priority=0,
                token_count=counter.count(profile_rendered),
                droppable=False,
            ),
            BudgetBlock(
                name="boundaries",
                content=boundaries_rendered,
                priority=0,
                token_count=counter.count(boundaries_rendered),
                droppable=False,
            ),
            BudgetBlock(
                name="state",
                content=state_block,
                priority=2,
                token_count=counter.count(state_block) if state_block else 0,
                droppable=True,
            ),
            BudgetBlock(
                name="commitments",
                content=commitments_block,
                priority=1,
                token_count=counter.count(commitments_block) if commitments_block else 0,
                droppable=True,
            ),
            BudgetBlock(
                name="memories",
                content=memories_rendered,
                priority=1,
                token_count=counter.count(memories_rendered),
                droppable=True,
            ),
            BudgetBlock(
                name="summary",
                content=session_summary,
                priority=1,
                token_count=counter.count(session_summary) if session_summary else 0,
                droppable=True,
            ),
            BudgetBlock(
                name="recent_turns",
                content=recent_turns_rendered,
                priority=0,
                token_count=counter.count(recent_turns_rendered),
                droppable=False,
            ),
            BudgetBlock(
                name="response_headroom",
                content="",
                priority=0,
                token_count=DEFAULT_BUDGET_ALLOCATION["response_headroom"],
                droppable=False,
            ),
        ]

        # First pass: allocate with full content
        kept_blocks, allocation_log = allocate(blocks, total_budget, counter)

        retrieved_ids = [m.get("id", m.get("memory_id", "")) for m in retrieved_memories]
        kept_ids: list[str] = []
        dropped_ids: list[str] = []

        # Check if memories block was truncated — if so, do item-level
        # truncation by dropping from the tail of the pre-sorted list, then
        # recompute rendered text.
        memories_truncated = "truncated_memories" in allocation_log
        if memories_truncated:
            truncated_tokens = allocation_log["truncated_memories"]["truncated_tokens"]
            truncated_memories_list = list(retrieved_memories)
            while truncated_memories_list:
                rendered = render_memories_block(truncated_memories_list)
                if counter.count(rendered) <= truncated_tokens:
                    break
                truncated_memories_list.pop()
            # Update the memories block in kept_blocks with re-rendered content
            for b in kept_blocks:
                if b.name == "memories":
                    b.content = rendered
                    b.token_count = counter.count(rendered)
                    break

            kept_ids = [m.get("id", m.get("memory_id", "")) for m in truncated_memories_list]
            dropped_ids = [m.get("id", m.get("memory_id", "")) for m in retrieved_memories if m not in truncated_memories_list]
        else:
            # Not truncated — check if fully kept or dropped
            if any(b.name == "memories" for b in kept_blocks):
                kept_ids = retrieved_ids
            else:
                dropped_ids = retrieved_ids

        # Check if recent_turns block was truncated
        turns_truncated = "truncated_recent_turns" in allocation_log
        if turns_truncated:
            truncated_tokens = allocation_log["truncated_recent_turns"]["truncated_tokens"]
            truncated_turns_list = list(recent_turns)
            while truncated_turns_list:
                rendered = render_recent_turns(truncated_turns_list)
                if counter.count(rendered) <= truncated_tokens:
                    break
                truncated_turns_list.pop()
            for b in kept_blocks:
                if b.name == "recent_turns":
                    b.content = rendered
                    b.token_count = counter.count(rendered)
                    break

        # Assemble final context in cache-friendly order:
        # static blocks first, then dynamic, then recent turns, then user turn
        assembly_order = [
            "persona",
            "profile",
            "boundaries",
            "state",
            "commitments",
            "memories",
            "summary",
            "recent_turns",
        ]

        kept_by_name = {b.name: b for b in kept_blocks}
        sections: list[str] = []
        for name in assembly_order:
            if name in kept_by_name:
                content = kept_by_name[name].content
                if content:
                    sections.append(content)

        # Append user turn
        sections.append(user_turn)

        assembled = "\n\n".join(sections)

        # Compute per-section token counts from kept blocks for TraceRecord
        persona_tokens = 0
        profile_tokens = 0
        boundaries_tokens = 0
        memories_tokens = 0
        recent_turns_tokens = 0

        for b in kept_blocks:
            if b.name == "persona":
                persona_tokens = b.token_count
            elif b.name == "profile":
                profile_tokens = b.token_count
            elif b.name == "boundaries":
                boundaries_tokens = b.token_count
            elif b.name == "memories":
                memories_tokens = b.token_count
            elif b.name == "recent_turns":
                recent_turns_tokens = b.token_count

        total_tokens = counter.count(assembled)

        trace = TraceRecord(
            turn_id=turn_id,
            timestamp_ms=int(time.time() * 1000),
            persona_tokens=persona_tokens,
            profile_tokens=profile_tokens,
            boundaries_tokens=boundaries_tokens,
            memories_tokens=memories_tokens,
            recent_turns_tokens=recent_turns_tokens,
            total_tokens=total_tokens,
            retrieved_memory_ids=retrieved_ids,
            kept_memory_ids=kept_ids,
            dropped_memory_ids=dropped_ids,
            allocation_log=allocation_log,
            assembled_context=assembled,
        )

        return assembled, trace
