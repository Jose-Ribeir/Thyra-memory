"""Scoring: spreading activation + category presence."""

from __future__ import annotations

import math
from typing import Sequence

from thyra.config import (
    DISCRIMINABILITY_FLOOR,
    PRESENCE_FLOOR,
    SPREADING_ASSOC,
    SPREADING_DIRECT,
    SPREADING_SITUATION,
)
from thyra.models.memory import MemoryRecord, SituationEdge, compute_base_level


def score_memories(
    memories: Sequence[MemoryRecord],
    cues: Sequence[str],
    cue_edge_map: dict[str, list[tuple[str, float]]],
    assoc_edge_map: dict[str, list[tuple[str, float]]],
    situation_edges: list[SituationEdge],
    idf: dict[str, float],
    category_weights: dict[str, float],
    now_ms: int,
) -> list[tuple[MemoryRecord, float]]:
    """Return (memory, score) pairs sorted descending."""
    cue_set = set(cues)

    # Direct cue activation: sum(edge_weight * discrim) for each fired cue
    directly_activated: dict[str, float] = {}
    for cue in cues:
        discrim = idf.get(cue, DISCRIMINABILITY_FLOOR)
        for mem_id, ew in cue_edge_map.get(cue, []):
            directly_activated[mem_id] = (
                directly_activated.get(mem_id, 0.0) + ew * discrim
            )

    # Situation edge activation
    situation_activation: dict[str, float] = {}
    for sit in situation_edges:
        if all(c in cue_set for c in sit.cue_set):
            situation_activation[sit.memory_id] = (
                situation_activation.get(sit.memory_id, 0.0)
                + sit.weight * SPREADING_SITUATION
            )

    scored: list[tuple[MemoryRecord, float]] = []
    for rec in memories:
        base_level = compute_base_level(
            rec.base_strength, rec.decay_rate, rec.last_access, now_ms
        )

        # Direct spreading
        spreading = directly_activated.get(rec.id, 0.0) * SPREADING_DIRECT

        # One-hop association spread
        visited = {rec.id}
        for neighbor_id, aw in assoc_edge_map.get(rec.id, []):
            if neighbor_id not in visited and neighbor_id in directly_activated:
                spreading += aw * SPREADING_ASSOC
                visited.add(neighbor_id)

        # Situation contribution
        spreading += situation_activation.get(rec.id, 0.0)

        # Category presence
        cat_weights = [category_weights.get(c, 0.0) for c in [rec.category]]
        presence = PRESENCE_FLOOR + (1.0 - PRESENCE_FLOOR) * (
            1.0 - math.prod(1.0 - w for w in cat_weights)
        )

        score = (base_level + spreading) * presence
        scored.append((rec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
