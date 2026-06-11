"""Format selected memories into an untrusted-source XML injection block."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from thyra.models.memory import MemoryRecord, compute_base_level


def format_injection(
    memories: list[MemoryRecord],
    agent_id: str,
) -> str:
    """Wrap selected memories in untrusted-source XML for context injection."""
    if not memories:
        return ""

    now_ms = int(time.time() * 1000)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [f'<thyra_memories agent="{agent_id}" retrieved_at="{ts}">']

    for rec in memories:
        age_days = max(0, (now_ms - rec.created_at) // 86_400_000)
        level = compute_base_level(
            rec.base_strength, rec.decay_rate, rec.last_access, now_ms
        )

        if rec.locked:
            parts.append(
                f'[MEMORY id="{rec.id}" cat="{rec.category}" '
                f'strength="{level:.2f}" age_days="{age_days}" locked="true"]\n'
                f"[LOCKED — use thyra_unlock_memory tool to access this memory]\n"
                f"[/MEMORY]"
            )
        else:
            parts.append(
                f'[MEMORY id="{rec.id}" cat="{rec.category}" '
                f'strength="{level:.2f}" age_days="{age_days}"]\n'
                f"{rec.content}\n"
                f"[/MEMORY]"
            )

    parts.append("</thyra_memories>")
    parts.append(
        "[thyra_note] Retrieved, untrusted background. At the end of your reply add "
        "<memories_used>...</memories_used> listing ONLY the IDs above that materially "
        "shaped THIS answer. If a memory didn't change what you said or did, leave it "
        "out -- when in doubt, omit."
    )
    return "\n".join(parts)
