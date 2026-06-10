"""FastAPI dashboard server — memory browser and mode controls for Thyra."""

from __future__ import annotations

import math
import re
import time
from typing import Any, Optional
import pathlib

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from thyra.config import THYRA_AGENT_ID, THYRA_DB_PATH, THYRA_USER_ID
from thyra.db.connection import DBConnection
from thyra.models.memory import (
    archive_memory,
    count_active,
    delete_memory,
    get_flag,
    set_flag,
)
from thyra.recall.cache import HOT_CACHE

_STATIC = pathlib.Path(__file__).parent / "static"
_MEMORY_ID_RE = re.compile(r"^m_[0-9a-f]{12,}$", re.IGNORECASE)

app = FastAPI(title="Thyra Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def _conn():
    return DBConnection.get(THYRA_DB_PATH)


def _is_memory_id(s: str) -> bool:
    return bool(_MEMORY_ID_RE.match(s))


# ── Request bodies ─────────────────────────────────────────────────────────────


class ToggleBody(BaseModel):
    user_id: str = THYRA_USER_ID
    agent_id: str = THYRA_AGENT_ID
    enabled: bool


class MaxMemoriesBody(BaseModel):
    user_id: str = THYRA_USER_ID
    agent_id: str = THYRA_AGENT_ID
    count: int


class NsBody(BaseModel):
    user_id: str = THYRA_USER_ID
    agent_id: str = THYRA_AGENT_ID


# ── Static ──────────────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(str(_STATIC / "index.html"))


# ── Current namespace (active Claude Code session) ────────────────────────────


@app.get("/api/current")
async def get_current() -> dict[str, Any]:
    """Return the namespace for the active Claude Code session.

    Reads thyra_ctx_latest.json written by thyra_init_session / pre_turn hook.
    Falls back to the THYRA_USER_ID / THYRA_AGENT_ID env-var defaults so the
    dashboard always has something sensible to show.
    """
    import json
    import os
    import tempfile

    user_id = THYRA_USER_ID
    agent_id = THYRA_AGENT_ID
    try:
        ctx_path = os.path.join(
            os.environ.get("TEMP", tempfile.gettempdir()),
            "thyra_ctx_latest.json",
        )
        if os.path.exists(ctx_path):
            with open(ctx_path, encoding="utf-8") as f:
                ctx = json.load(f)
            agent_id = ctx.get("project_id") or agent_id
    except Exception:
        pass
    return {"user_id": user_id, "agent_id": agent_id}


# ── Namespaces ─────────────────────────────────────────────────────────────────


@app.get("/api/namespaces")
async def get_namespaces() -> dict[str, Any]:
    conn = _conn()
    rows = conn.execute(
        """SELECT user_id, agent_id, COUNT(*) AS mem_count
           FROM memories GROUP BY user_id, agent_id
           ORDER BY mem_count DESC"""
    ).fetchall()
    return {
        "namespaces": [
            {
                "user_id": r["user_id"],
                "agent_id": r["agent_id"],
                "mem_count": r["mem_count"],
            }
            for r in rows
        ]
    }


# ── Status ─────────────────────────────────────────────────────────────────────


@app.get("/api/status")
async def get_status(
    user_id: str = Query(default=THYRA_USER_ID),
    agent_id: str = Query(default=THYRA_AGENT_ID),
) -> dict[str, Any]:
    conn = _conn()
    archived = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE user_id=? AND agent_id=? AND archived=1",
        (user_id, agent_id),
    ).fetchone()[0]
    last_nightly_raw = get_flag(conn, "last_nightly", user_id, agent_id, default="0")
    import time as _t

    now_ms = int(_t.time() * 1000)
    last_ping = int(
        get_flag(conn, "last_monitor_ping", THYRA_USER_ID, THYRA_AGENT_ID, default="0")
        or "0"
    )
    monitor_ok = last_ping > 0 and (now_ms - last_ping) < 60_000
    return {
        "user_id": user_id,
        "agent_id": agent_id,
        "system_enabled": get_flag(conn, "system_enabled", user_id, agent_id) == "true",
        "formation_enabled": get_flag(conn, "formation_enabled", user_id, agent_id)
        == "true",
        "max_memories": int(
            get_flag(conn, "max_memories", user_id, agent_id, default="0") or "0"
        ),
        "active_memories": count_active(conn, user_id, agent_id),
        "archived_memories": archived,
        "last_nightly": int(last_nightly_raw or "0"),
        "monitor_ok": monitor_ok,
        "last_monitor_ping": last_ping,
    }


# ── Memories list ──────────────────────────────────────────────────────────────


@app.get("/api/memories")
async def list_memories(
    user_id: str = Query(default=THYRA_USER_ID),
    agent_id: str = Query(default=THYRA_AGENT_ID),
    category: Optional[str] = Query(default=None),
    memory_type: Optional[str] = Query(default=None),
    archived: bool = Query(default=False),
    probationary: Optional[bool] = Query(default=None),
    search: Optional[str] = Query(default=None),
    global_search: bool = Query(default=False),
    sort_by: str = Query(default="strength"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    conn = _conn()
    now_ms = int(time.time() * 1000)

    # Global search: drop namespace filter so all namespaces are searched.
    is_global = global_search and bool(search and search.strip())
    if is_global:
        conds: list[str] = ["m.archived=?"]
        params: list[Any] = [int(archived)]
    else:
        conds = ["m.user_id=?", "m.agent_id=?", "m.archived=?"]
        params = [user_id, agent_id, int(archived)]

    if category:
        conds.append("m.category=?")
        params.append(category)
    if memory_type:
        conds.append("m.memory_type=?")
        params.append(memory_type)
    if probationary is not None:
        conds.append("m.probationary=?")
        params.append(int(probationary))

    where = " AND ".join(conds)
    sort_map = {
        "strength": "m.base_strength DESC",
        "created": "m.created_at DESC",
        "last_access": "m.last_access DESC",
        "use_count": "m.use_count DESC",
    }
    order = sort_map.get(sort_by, "m.base_strength DESC")
    offset = (page - 1) * page_size

    if search and search.strip():
        s = search.strip()
        # Direct ID lookup: if the term looks like a memory ID skip FTS entirely
        if _is_memory_id(s):
            id_conds = conds + ["m.id=?"]
            id_where = " AND ".join(id_conds)
            total = conn.execute(
                f"SELECT COUNT(*) FROM memories m WHERE {id_where}",
                params + [s],
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT m.* FROM memories m WHERE {id_where} LIMIT ? OFFSET ?",
                params + [s, page_size, offset],
            ).fetchall()
        else:
            tokens = s.split()
            fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in tokens)
            count_sql = (
                f"SELECT COUNT(*) FROM memory_fts fts "
                f"JOIN memories m ON fts.rowid = m.memory_int_id "
                f"WHERE fts.content MATCH ? AND {where}"
            )
            list_sql = (
                f"SELECT m.* FROM memory_fts fts "
                f"JOIN memories m ON fts.rowid = m.memory_int_id "
                f"WHERE fts.content MATCH ? AND {where} "
                f"ORDER BY rank LIMIT ? OFFSET ?"
            )
            total = conn.execute(count_sql, [fts_query] + params).fetchone()[0]
            rows = conn.execute(
                list_sql, [fts_query] + params + [page_size, offset]
            ).fetchall()
    else:
        total = conn.execute(
            f"SELECT COUNT(*) FROM memories m WHERE {where}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"SELECT m.* FROM memories m WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()

    memories = []
    for r in rows:
        days = max(0.0, (now_ms - r["last_access"]) / 86_400_000)
        current_level = r["base_strength"] * math.exp(-r["decay_rate"] * days)
        age_days = max(0.0, (now_ms - r["created_at"]) / 86_400_000)
        locked = bool(r["locked"])
        memories.append(
            {
                "id": r["id"],
                "user_id": r["user_id"],
                "agent_id": r["agent_id"],
                "category": r["category"],
                "memory_type": r["memory_type"],
                "base_strength": round(r["base_strength"], 4),
                "decay_rate": r["decay_rate"],
                "current_level": round(current_level, 4),
                "age_days": round(age_days, 1),
                "last_access_ms": r["last_access"],
                "created_at_ms": r["created_at"],
                "use_count": r["use_count"],
                "probationary": bool(r["probationary"]),
                "archived": bool(r["archived"]),
                "locked": locked,
                "content": r["content"] if not locked else "[ENCRYPTED]",
                "snippet": (r["content"][:120] if not locked else "[ENCRYPTED]"),
            }
        )

    return {
        "memories": memories,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, math.ceil(total / page_size)),
        "global_search": is_global,
    }


# ── Memory detail ──────────────────────────────────────────────────────────────


@app.get("/api/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    user_id: str = Query(default=THYRA_USER_ID),
    agent_id: str = Query(default=THYRA_AGENT_ID),
) -> dict[str, Any]:
    conn = _conn()
    now_ms = int(time.time() * 1000)
    row = conn.execute(
        "SELECT * FROM memories WHERE id=? AND user_id=? AND agent_id=?",
        (memory_id, user_id, agent_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")

    cue_rows = conn.execute(
        """SELECT cue_id, weight, fire_count, use_count, candidate
           FROM cue_edges WHERE memory_id=? AND user_id=? AND agent_id=?
           ORDER BY weight DESC LIMIT 40""",
        (memory_id, user_id, agent_id),
    ).fetchall()

    assoc_rows = conn.execute(
        """SELECT CASE WHEN memory_a=? THEN memory_b ELSE memory_a END AS neighbor,
                  weight, co_use
           FROM association_edges
           WHERE (memory_a=? OR memory_b=?) AND user_id=? AND agent_id=?
           ORDER BY weight DESC LIMIT 20""",
        (memory_id, memory_id, memory_id, user_id, agent_id),
    ).fetchall()

    days = max(0.0, (now_ms - row["last_access"]) / 86_400_000)
    current_level = row["base_strength"] * math.exp(-row["decay_rate"] * days)
    age_days = max(0.0, (now_ms - row["created_at"]) / 86_400_000)
    locked = bool(row["locked"])

    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "agent_id": row["agent_id"],
        "category": row["category"],
        "memory_type": row["memory_type"],
        "base_strength": round(row["base_strength"], 4),
        "decay_rate": row["decay_rate"],
        "current_level": round(current_level, 4),
        "age_days": round(age_days, 2),
        "last_access_ms": row["last_access"],
        "created_at_ms": row["created_at"],
        "archived_at_ms": row["archived_at"],
        "use_count": row["use_count"],
        "probationary": bool(row["probationary"]),
        "archived": bool(row["archived"]),
        "locked": locked,
        "content": row["content"] if not locked else "[ENCRYPTED]",
        "cue_edges": [
            {
                "cue_id": c["cue_id"],
                "weight": round(c["weight"], 3),
                "fire_count": c["fire_count"],
                "use_count": c["use_count"],
                "candidate": bool(c["candidate"]),
            }
            for c in cue_rows
        ],
        "association_edges": [
            {
                "neighbor": a["neighbor"],
                "weight": round(a["weight"], 3),
                "co_use": a["co_use"],
            }
            for a in assoc_rows
        ],
    }


# ── Stats ──────────────────────────────────────────────────────────────────────


@app.get("/api/stats")
async def get_stats(
    user_id: str = Query(default=THYRA_USER_ID),
    agent_id: str = Query(default=THYRA_AGENT_ID),
) -> dict[str, Any]:
    conn = _conn()
    now_ms = int(time.time() * 1000)

    by_cat = conn.execute(
        """SELECT category, COUNT(*) AS cnt, AVG(base_strength) AS avg_str
           FROM memories WHERE user_id=? AND agent_id=? AND archived=0
           GROUP BY category ORDER BY cnt DESC""",
        (user_id, agent_id),
    ).fetchall()

    by_type = conn.execute(
        """SELECT memory_type, COUNT(*) AS cnt
           FROM memories WHERE user_id=? AND agent_id=? AND archived=0
           GROUP BY memory_type""",
        (user_id, agent_id),
    ).fetchall()

    strengths = conn.execute(
        "SELECT base_strength FROM memories WHERE user_id=? AND agent_id=? AND archived=0",
        (user_id, agent_id),
    ).fetchall()
    buckets = [0] * 10
    for r in strengths:
        idx = min(9, int(r["base_strength"]))
        buckets[idx] += 1

    seven_days_ago = now_ms - 7 * 86_400_000
    daily = conn.execute(
        """SELECT date(created_at / 1000, 'unixepoch') AS day, COUNT(*) AS cnt
           FROM memories WHERE user_id=? AND agent_id=? AND created_at >= ?
           GROUP BY day ORDER BY day""",
        (user_id, agent_id, seven_days_ago),
    ).fetchall()

    graph = {
        "cue_nodes": conn.execute(
            "SELECT COUNT(*) FROM cue_nodes WHERE user_id=? AND agent_id=?",
            (user_id, agent_id),
        ).fetchone()[0],
        "cue_edges": conn.execute(
            "SELECT COUNT(*) FROM cue_edges WHERE user_id=? AND agent_id=?",
            (user_id, agent_id),
        ).fetchone()[0],
        "assoc_edges": conn.execute(
            "SELECT COUNT(*) FROM association_edges WHERE user_id=? AND agent_id=?",
            (user_id, agent_id),
        ).fetchone()[0],
        "situation_edges": conn.execute(
            "SELECT COUNT(*) FROM situation_edges WHERE user_id=? AND agent_id=? AND candidate=0",
            (user_id, agent_id),
        ).fetchone()[0],
    }

    return {
        "by_category": [
            {
                "category": r["category"],
                "count": r["cnt"],
                "avg_strength": round(r["avg_str"] or 0, 3),
            }
            for r in by_cat
        ],
        "by_type": [{"type": r["memory_type"], "count": r["cnt"]} for r in by_type],
        "strength_histogram": {
            "buckets": buckets,
            "labels": [
                "0–1",
                "1–2",
                "2–3",
                "3–4",
                "4–5",
                "5–6",
                "6–7",
                "7–8",
                "8–9",
                "9–10",
            ],
        },
        "daily_formations": [{"day": r["day"], "count": r["cnt"]} for r in daily],
        "graph": graph,
    }


# ── Logs & activity ────────────────────────────────────────────────────────────


@app.get("/api/logs")
async def get_logs(
    since_ts: float = Query(default=0.0),
    limit: int = Query(default=100, le=500),
) -> dict[str, Any]:
    from thyra.dashboard.log_handler import get_handler

    handler = get_handler()
    if handler is None:
        return {"entries": [], "note": "log handler not installed"}
    return {"entries": handler.recent(since_ts=since_ts, limit=limit)}


@app.get("/api/activity")
async def get_activity(
    user_id: str = Query(default=THYRA_USER_ID),
    agent_id: str = Query(default=THYRA_AGENT_ID),
    limit: int = Query(default=50, le=200),
) -> dict[str, Any]:
    import json as _json

    conn = _conn()
    rows = conn.execute(
        """SELECT turn_id, session_id, memories_served, memories_used, cues_fired, created_at
           FROM turn_log WHERE user_id=? AND agent_id=?
           ORDER BY created_at DESC LIMIT ?""",
        (user_id, agent_id, limit),
    ).fetchall()
    return {
        "turns": [
            {
                "turn_id": r["turn_id"],
                "session_id": r["session_id"],
                "memories_served": _json.loads(r["memories_served"]),
                "memories_used": _json.loads(r["memories_used"]),
                "cues_fired": _json.loads(r["cues_fired"]),
                "created_at_ms": r["created_at"],
            }
            for r in rows
        ]
    }


# ── Toggle / settings ──────────────────────────────────────────────────────────


@app.post("/api/toggle/system")
async def toggle_system(body: ToggleBody) -> dict[str, Any]:
    conn = _conn()
    set_flag(
        conn,
        "system_enabled",
        "true" if body.enabled else "false",
        body.user_id,
        body.agent_id,
    )
    HOT_CACHE.invalidate(f"snapshot:{body.user_id}:{body.agent_id}")
    return {"success": True, "system_enabled": body.enabled}


@app.post("/api/toggle/formation")
async def toggle_formation(body: ToggleBody) -> dict[str, Any]:
    conn = _conn()
    set_flag(
        conn,
        "formation_enabled",
        "true" if body.enabled else "false",
        body.user_id,
        body.agent_id,
    )
    return {"success": True, "formation_enabled": body.enabled}


@app.post("/api/settings/max-memories")
async def set_max_memories(body: MaxMemoriesBody) -> dict[str, Any]:
    if body.count < 0:
        raise HTTPException(status_code=422, detail="count must be >= 0")
    conn = _conn()
    set_flag(conn, "max_memories", str(body.count), body.user_id, body.agent_id)
    return {"success": True, "max_memories": body.count}


# ── Memory actions ─────────────────────────────────────────────────────────────


@app.post("/api/memories/{memory_id}/delete")
async def delete_memory_endpoint(memory_id: str, body: NsBody) -> dict[str, Any]:
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM memories WHERE id=? AND user_id=? AND agent_id=?",
        (memory_id, body.user_id, body.agent_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    delete_memory(conn, memory_id, body.user_id, body.agent_id)
    HOT_CACHE.invalidate(f"snapshot:{body.user_id}:{body.agent_id}")
    return {"success": True, "deleted": memory_id}


@app.post("/api/memories/{memory_id}/archive")
async def archive_memory_endpoint(memory_id: str, body: NsBody) -> dict[str, Any]:
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM memories WHERE id=? AND user_id=? AND agent_id=?",
        (memory_id, body.user_id, body.agent_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    archive_memory(
        conn, memory_id, int(time.time() * 1000), body.user_id, body.agent_id
    )
    HOT_CACHE.invalidate(f"snapshot:{body.user_id}:{body.agent_id}")
    return {"success": True, "archived": memory_id}
