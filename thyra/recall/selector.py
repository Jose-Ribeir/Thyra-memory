"""Token-budget greedy selection with category saturation."""

from __future__ import annotations

from thyra.config import (
    GAMMA_DEFAULT,
    MAX_MEMORIES_INJECTED,
    PER_MEMORY_CAP,
    SCORE_FLOOR,
    TOKEN_BUDGET,
)
from thyra.models.memory import MemoryRecord

PROTECTED_CATEGORIES = frozenset({"constraints", "identity"})


def _estimate_tokens(content: str) -> int:
    return max(1, len(content) // 4)


def greedy_select(
    scored: list[tuple[MemoryRecord, float]],
    token_budget: int = TOKEN_BUDGET,
    gamma: float = GAMMA_DEFAULT,
    score_floor: float = SCORE_FLOOR,
    max_count: int = MAX_MEMORIES_INJECTED,
) -> list[MemoryRecord]:
    """Select memories greedily into token budget with category saturation.

    max_count: hard cap on number of memories returned (0 = unlimited).
    """
    """Select memories greedily into token budget with category saturation.

    Protected categories (constraints, identity) are given a saturation
    override: their gamma is 1.0 so they are never starved.
    """
    per_mem_cap = int(token_budget * PER_MEMORY_CAP)
    used_tokens = 0
    chosen: list[MemoryRecord] = []
    category_counts: dict[str, int] = {}
    skipped: list[tuple[MemoryRecord, float]] = []

    for rec, score in scored:
        if score < score_floor:
            break
        tokens = min(per_mem_cap, _estimate_tokens(rec.content))
        if used_tokens + tokens > token_budget:
            skipped.append((rec, score))
            continue

        # Category saturation discount
        n = category_counts.get(rec.category, 0)
        if rec.category in PROTECTED_CATEGORIES:
            effective_score = score  # no discount
        else:
            effective_score = score * (gamma**n)

        if effective_score < score_floor and n > 0:
            skipped.append((rec, score))
            continue

        chosen.append(rec)
        used_tokens += tokens
        category_counts[rec.category] = n + 1

    # Backfill with smallest skipped memories above floor
    skipped.sort(key=lambda x: _estimate_tokens(x[0].content))
    for rec, score in skipped:
        if score < score_floor:
            continue
        if max_count > 0 and len(chosen) >= max_count:
            break
        tokens = min(per_mem_cap, _estimate_tokens(rec.content))
        if used_tokens + tokens <= token_budget:
            chosen.append(rec)
            used_tokens += tokens

    if max_count > 0:
        chosen = chosen[:max_count]

    return chosen
