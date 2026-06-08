"""Seed category definitions and constants."""

from __future__ import annotations

from dataclasses import dataclass

PROTECTED_CATEGORIES: frozenset[str] = frozenset(
    {"constraints", "identity", "preferences", "context"}
)

SLOW_DECAY_CATEGORIES: frozenset[str] = frozenset({"constraints", "identity"})

SEED_CATEGORIES: tuple[str, ...] = (
    "constraints",
    "identity",
    "preferences",
    "relationships",
    "tasks",
    "goals",
    "context",
    "skills",
    "habits",
    "knowledge",
    "events",
    "communication",
    "health",
    "finance",
    "routines",
)

# Default decay rate per category
CATEGORY_DECAY_RATES: dict[str, float] = {
    "constraints": 0.001,
    "identity": 0.001,
    "preferences": 0.02,
    "relationships": 0.02,
    "tasks": 0.02,
    "goals": 0.02,
    "context": 0.02,
    "skills": 0.02,
    "habits": 0.02,
    "knowledge": 0.02,
    "events": 0.02,
    "communication": 0.02,
    "health": 0.02,
    "finance": 0.02,
    "routines": 0.02,
}

# Static relevance floors (used as minimum weight in recall scoring)
CATEGORY_RELEVANCE_FLOORS: dict[str, float] = {
    "constraints": 0.80,
    "identity": 0.70,
    "preferences": 0.60,
    "context": 0.50,
    "relationships": 0.0,
    "tasks": 0.0,
    "goals": 0.0,
    "skills": 0.0,
    "habits": 0.0,
    "knowledge": 0.0,
    "events": 0.0,
    "communication": 0.0,
    "health": 0.0,
    "finance": 0.0,
    "routines": 0.0,
}
