"""Emergent category detection via community detection on the co-use graph."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Optional

from thyra.config import (
    EMERGENT_MIN_EDGE_WEIGHT,
    EMERGENT_MIN_MEMBERS,
    EMERGENT_MIN_TURNS,
    HUB_CUE_FRACTION,
    THYRA_AGENT_ID,
    THYRA_USER_ID,
)


def _find_hub_cues(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
) -> set[str]:
    """Return cue_ids that point to > HUB_CUE_FRACTION of all active memories."""
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchone()
    M = row[0] if row else 0
    if M < 2:
        return set()
    threshold = int(M * HUB_CUE_FRACTION)
    rows = conn.execute(
        "SELECT cue_id FROM cue_nodes WHERE user_id=? AND agent_id=? AND df > ?",
        (user_id, agent_id, threshold),
    ).fetchall()
    return {r["cue_id"] for r in rows}


def _build_co_occurrence_graph(
    conn: sqlite3.Connection,
    user_id: str,
    agent_id: str,
    hub_cues: set[str],
) -> dict[str, set[str]]:
    """Build adjacency list from assoc_edges and shared cue edges (excluding hubs)."""
    adj: dict[str, set[str]] = defaultdict(set)

    # 1. Association edges above threshold
    rows = conn.execute(
        "SELECT memory_a, memory_b FROM association_edges WHERE user_id=? AND agent_id=? AND weight >= ?",
        (user_id, agent_id, EMERGENT_MIN_EDGE_WEIGHT),
    ).fetchall()
    for r in rows:
        adj[r["memory_a"]].add(r["memory_b"])
        adj[r["memory_b"]].add(r["memory_a"])

    # 2. Memories sharing ≥2 non-hub cue edges
    rows2 = conn.execute(
        "SELECT memory_id, cue_id FROM cue_edges WHERE user_id=? AND agent_id=? AND candidate=0",
        (user_id, agent_id),
    ).fetchall()
    # Build {cue_id: [memory_ids]} excluding hub cues
    cue_to_mems: dict[str, list[str]] = defaultdict(list)
    for r in rows2:
        if r["cue_id"] not in hub_cues:
            cue_to_mems[r["cue_id"]].append(r["memory_id"])

    # Count shared non-hub cues between pairs
    shared_count: dict[tuple[str, str], int] = defaultdict(int)
    for cue_id, mems in cue_to_mems.items():
        for i in range(len(mems)):
            for j in range(i + 1, len(mems)):
                a, b = (mems[i], mems[j]) if mems[i] < mems[j] else (mems[j], mems[i])
                shared_count[(a, b)] += 1

    for (a, b), count in shared_count.items():
        if count >= 2:
            adj[a].add(b)
            adj[b].add(a)

    return adj


def _connected_components(
    adj: dict[str, set[str]], all_nodes: set[str]
) -> list[set[str]]:
    """Deterministic connected-components on the adjacency list."""
    visited = set()
    components = []
    for node in sorted(all_nodes):
        if node in visited:
            continue
        component = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in sorted(adj.get(current, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(component) >= 2:
            components.append(component)
    return components


def _count_co_activation_turns(
    conn: sqlite3.Connection,
    memory_ids: set[str],
    user_id: str,
    agent_id: str,
) -> int:
    """Count turns where ≥2 memories from the set were used together."""
    import json

    rows = conn.execute(
        "SELECT memories_used FROM turn_log WHERE user_id=? AND agent_id=?",
        (user_id, agent_id),
    ).fetchall()
    count = 0
    for row in rows:
        used = set(json.loads(row["memories_used"]))
        if len(used & memory_ids) >= 2:
            count += 1
    return count


def _derive_label(memory_ids: set[str], conn: sqlite3.Connection) -> str:
    """Derive a category label from the most common existing category of the members."""
    from collections import Counter

    rows = conn.execute(
        f"SELECT category FROM memories WHERE id IN ({','.join('?' * len(memory_ids))})",
        list(memory_ids),
    ).fetchall()
    cats = [r["category"] for r in rows if r["category"] not in ("context",)]
    if not cats:
        return "context"
    most_common = Counter(cats).most_common(1)[0][0]
    # Strip all accumulated _cluster suffixes so repeated runs don't compound them.
    base = most_common
    while base.endswith("_cluster"):
        base = base[: -len("_cluster")]
    return base + "_cluster"


def detect_emergent_categories(
    conn: sqlite3.Connection,
    user_id: str = THYRA_USER_ID,
    agent_id: str = THYRA_AGENT_ID,
) -> list[str]:
    """Find memory communities and crystallize new categories.

    Returns list of new category IDs created.
    """
    hub_cues = _find_hub_cues(conn, user_id, agent_id)
    adj = _build_co_occurrence_graph(conn, user_id, agent_id, hub_cues)

    all_node_rows = conn.execute(
        "SELECT id FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchall()
    all_nodes = {r["id"] for r in all_node_rows}

    components = _connected_components(adj, all_nodes)

    # Fetch existing category IDs to avoid duplicates
    existing_cats = {
        r["cat_id"]
        for r in conn.execute(
            "SELECT cat_id FROM categories WHERE user_id=? AND agent_id=?",
            (user_id, agent_id),
        ).fetchall()
    }

    new_categories = []
    from thyra.categories.manager import CategoryManager

    manager = CategoryManager(conn, user_id, agent_id)

    for component in components:
        if len(component) < EMERGENT_MIN_MEMBERS:
            continue
        turns = _count_co_activation_turns(conn, component, user_id, agent_id)
        if turns < EMERGENT_MIN_TURNS:
            continue

        label = _derive_label(component, conn)
        if label in existing_cats:
            continue

        manager.add_emergent_category(label)
        # Re-home member memories to this new category
        with conn:
            conn.execute(
                f"UPDATE memories SET category=? WHERE id IN ({','.join('?' * len(component))})"
                f" AND user_id=? AND agent_id=?",
                [label, *component, user_id, agent_id],
            )
        existing_cats.add(label)
        new_categories.append(label)

    return new_categories
