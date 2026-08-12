from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

# Implementation uses len(text)//4 as an approximate token count.  This is a
# rough chars-per-token heuristic — swap in a real tokenizer for accuracy.
TOTAL_BUDGET_TOKENS = 2048

DEFAULT_BUDGET_ALLOCATION: dict[str, int] = {
    "persona": 180,
    "profile": 200,
    "boundaries": 80,
    "state": 60,
    "commitments": 60,
    "memories": 450,
    "summary": 120,
    "recent_turns": 600,
    "response_headroom": 256,
}


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class BudgetBlock(BaseModel):
    name: str
    content: str
    priority: int
    token_count: int
    droppable: bool = True
    value_score: float = 1.0


def _truncate_content_at_newlines(content: str, counter: TokenCounter, max_tokens: int) -> str:
    if not content:
        return ""
    lines = content.split("\n")
    kept_lines: list[str] = []
    kept_text = ""
    for line in lines:
        candidate = (kept_text + "\n" + line) if kept_lines else line
        if counter.count(candidate) <= max_tokens:
            kept_lines.append(line)
            kept_text = candidate
        else:
            break
    return kept_text


def allocate(
    blocks: list[BudgetBlock],
    total_budget: int,
    counter: TokenCounter,
) -> tuple[list[BudgetBlock], dict]:
    mandatory = [b for b in blocks if not b.droppable]
    mandatory_tokens = sum(b.token_count for b in mandatory)
    if mandatory_tokens > total_budget:
        raise ValueError(
            f"Mandatory (non-droppable) blocks require {mandatory_tokens} tokens "
            f"but total_budget is only {total_budget}"
        )

    remaining_budget = total_budget - mandatory_tokens
    droppable = sorted(
        [b for b in blocks if b.droppable],
        key=lambda b: (b.priority, -b.value_score / max(1, b.token_count)),
    )

    kept: list[BudgetBlock] = list(mandatory)
    log: dict = {
        "mandatory_kept": [b.name for b in mandatory],
        "mandatory_tokens": mandatory_tokens,
        "total_budget": total_budget,
        "remaining_start": remaining_budget,
    }

    budget_exhausted = False

    for block in droppable:
        if budget_exhausted:
            log[f"dropped_{block.name}"] = {
                "reason": "budget exhausted",
                "block_tokens": block.token_count,
            }
            continue

        if block.token_count <= remaining_budget:
            kept.append(block)
            remaining_budget -= block.token_count
            log[f"kept_{block.name}"] = block.token_count
        else:
            if block.name in ("memories", "recent_turns") and remaining_budget > 0:
                truncated = _truncate_content_at_newlines(block.content, counter, remaining_budget)
                new_token_count = counter.count(truncated) if truncated else 0
                truncated_block = block.model_copy(
                    update={
                        "content": truncated,
                        "token_count": new_token_count,
                    }
                )
                kept.append(truncated_block)
                log[f"truncated_{block.name}"] = {
                    "original_tokens": block.token_count,
                    "truncated_tokens": new_token_count,
                    "budget_remaining": remaining_budget,
                }
                remaining_budget = 0
                budget_exhausted = True
            else:
                log[f"dropped_{block.name}"] = {
                    "reason": "insufficient budget",
                    "block_tokens": block.token_count,
                    "budget_remaining": remaining_budget,
                }
                budget_exhausted = True

    kept_tokens = sum(b.token_count for b in kept)
    log["final_used"] = kept_tokens
    log["final_remaining"] = total_budget - kept_tokens

    return kept, log
