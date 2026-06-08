"""CategoryManager: relevance weights, decay rates, soft cap enforcement."""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

from thyra.categories.taxonomy import (
    CATEGORY_DECAY_RATES,
    CATEGORY_RELEVANCE_FLOORS,
    PROTECTED_CATEGORIES,
    SEED_CATEGORIES,
)
from thyra.config import (
    CATEGORY_SOFT_CAP,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)


class CategoryManager:
    def __init__(
        self,
        conn: sqlite3.Connection,
        user_id: str = THYRA_USER_ID,
        agent_id: str = THYRA_AGENT_ID,
    ) -> None:
        self._conn = conn
        self._user_id = user_id
        self._agent_id = agent_id

    def get_weight(
        self,
        cat_id: str,
        turn_cues: Optional[list[str]] = None,
    ) -> float:
        """Return relevance weight for cat_id this turn.

        Priority: (1) DB floor, (2) cue-based inference (stub), (3) static floor.
        """
        row = self._conn.execute(
            "SELECT relevance_floor, activation_score FROM categories WHERE cat_id=? AND user_id=? AND agent_id=?",
            (cat_id, self._user_id, self._agent_id),
        ).fetchone()
        if row:
            floor = row["relevance_floor"]
            activation = row["activation_score"]
        else:
            floor = CATEGORY_RELEVANCE_FLOORS.get(cat_id, 0.0)
            activation = 1.0
        return max(floor, activation * 0.5)

    def get_all_weights(
        self, turn_cues: Optional[list[str]] = None
    ) -> dict[str, float]:
        """Return relevance weights for all active categories."""
        rows = self._conn.execute(
            "SELECT cat_id FROM categories WHERE user_id=? AND agent_id=?",
            (self._user_id, self._agent_id),
        ).fetchall()
        return {r["cat_id"]: self.get_weight(r["cat_id"], turn_cues) for r in rows}

    def get_decay_rate(self, cat_id: str) -> float:
        """Return the confirmed (post-graduation) decay rate for a category."""
        return CATEGORY_DECAY_RATES.get(cat_id, 0.02)

    def soft_cap_enforce(self) -> list[str]:
        """Dissolve the weakest non-protected emergent categories if count > CATEGORY_SOFT_CAP.

        Dissolved memories are re-homed to 'context'.
        Returns list of dissolved category IDs.
        """
        rows = self._conn.execute(
            "SELECT cat_id, is_protected, is_emergent FROM categories WHERE user_id=? AND agent_id=?",
            (self._user_id, self._agent_id),
        ).fetchall()
        if len(rows) <= CATEGORY_SOFT_CAP:
            return []

        dissolvable = [
            r["cat_id"] for r in rows if not r["is_protected"] and r["is_emergent"]
        ]
        # Score by avg base_strength of member memories
        scored = []
        for cat_id in dissolvable:
            row = self._conn.execute(
                "SELECT AVG(base_strength) FROM memories WHERE category=? AND user_id=? AND agent_id=? AND archived=0",
                (cat_id, self._user_id, self._agent_id),
            ).fetchone()
            avg = row[0] or 0.0
            scored.append((cat_id, avg))
        scored.sort(key=lambda x: x[1])

        to_dissolve_count = len(rows) - CATEGORY_SOFT_CAP
        dissolved = []
        for cat_id, _ in scored[:to_dissolve_count]:
            self._dissolve_category(cat_id)
            dissolved.append(cat_id)
        return dissolved

    def _dissolve_category(self, cat_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE memories SET category='context' WHERE category=? AND user_id=? AND agent_id=?",
                (cat_id, self._user_id, self._agent_id),
            )
            self._conn.execute(
                "DELETE FROM categories WHERE cat_id=? AND user_id=? AND agent_id=?",
                (cat_id, self._user_id, self._agent_id),
            )

    def add_emergent_category(self, cat_id: str, decay_rate: float = 0.02) -> None:
        """Register a newly crystallized emergent category."""
        now = int(time.time() * 1000)
        with self._conn:
            self._conn.execute(
                """INSERT OR IGNORE INTO categories
                   (cat_id, user_id, agent_id, is_protected, is_emergent, decay_rate, relevance_floor, activation_score, created_at)
                   VALUES (?,?,?,0,1,?,0.0,1.0,?)""",
                (cat_id, self._user_id, self._agent_id, decay_rate, now),
            )
