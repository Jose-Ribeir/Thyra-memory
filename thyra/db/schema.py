"""SQLite schema: all CREATE TABLE / INDEX / FTS5 SQL."""

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- ── Core memories ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    memory_int_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    user_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    content         TEXT NOT NULL,          -- plaintext or "ENCRYPTED:..." for locked
    locked          INTEGER NOT NULL DEFAULT 0,
    category        TEXT NOT NULL DEFAULT 'context',
    memory_type     TEXT NOT NULL DEFAULT 'explicit',  -- explicit | semantic | episodic
    base_strength   REAL NOT NULL DEFAULT 1.0,
    decay_rate      REAL NOT NULL DEFAULT 0.02,
    last_access     INTEGER NOT NULL,       -- ms since epoch
    created_at      INTEGER NOT NULL,
    probationary    INTEGER NOT NULL DEFAULT 0,
    archived        INTEGER NOT NULL DEFAULT 0,
    archived_at     INTEGER,
    use_count       INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT                    -- sha1(normalized content) for O(1) exact dedup
);

CREATE INDEX IF NOT EXISTS idx_mem_pair
    ON memories(user_id, agent_id, archived);

CREATE INDEX IF NOT EXISTS idx_mem_strength
    ON memories(user_id, agent_id, archived, base_strength DESC);

CREATE INDEX IF NOT EXISTS idx_mem_archived_at
    ON memories(archived, archived_at) WHERE archived = 1;

CREATE INDEX IF NOT EXISTS idx_mem_content_hash
    ON memories(content_hash, user_id, agent_id) WHERE content_hash IS NOT NULL;

-- ── Full-text search (FTS5) on content ───────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
    USING fts5(content, content='memories', content_rowid='memory_int_id');

-- Keep FTS in sync with memories
CREATE TRIGGER IF NOT EXISTS mem_fts_insert
    AFTER INSERT ON memories BEGIN
        INSERT INTO memory_fts(rowid, content) VALUES (new.memory_int_id, new.content);
    END;

CREATE TRIGGER IF NOT EXISTS mem_fts_delete
    AFTER DELETE ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.memory_int_id, old.content);
    END;

CREATE TRIGGER IF NOT EXISTS mem_fts_update
    AFTER UPDATE OF content ON memories BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, content)
            VALUES ('delete', old.memory_int_id, old.content);
        INSERT INTO memory_fts(rowid, content) VALUES (new.memory_int_id, new.content);
    END;

-- ── Cue nodes (virtual; exist by virtue of edges) ────────────────────────────
CREATE TABLE IF NOT EXISTS cue_nodes (
    cue_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    df          INTEGER NOT NULL DEFAULT 0,  -- document frequency (# memories linked)
    PRIMARY KEY (cue_id, user_id, agent_id)
);

-- ── Cue → memory edges ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cue_edges (
    cue_id          TEXT NOT NULL,
    memory_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 0.30,
    fire_count      INTEGER NOT NULL DEFAULT 0,
    use_count       INTEGER NOT NULL DEFAULT 0,
    candidate       INTEGER NOT NULL DEFAULT 0,  -- 1 = probationary edge
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (cue_id, memory_id, user_id, agent_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cue_edges_cue
    ON cue_edges(cue_id, user_id, agent_id, candidate);

CREATE INDEX IF NOT EXISTS idx_cue_edges_memory
    ON cue_edges(memory_id, user_id, agent_id);

-- ── Memory ↔ memory association edges ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS association_edges (
    memory_a    TEXT NOT NULL,
    memory_b    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.0,
    co_use      INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (memory_a, memory_b, user_id, agent_id),
    FOREIGN KEY (memory_a) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (memory_b) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_assoc_a
    ON association_edges(memory_a, user_id, agent_id);

CREATE INDEX IF NOT EXISTS idx_assoc_b
    ON association_edges(memory_b, user_id, agent_id);

-- ── Situation edges (cue conjunction → memory) ───────────────────────────────
CREATE TABLE IF NOT EXISTS situation_edges (
    sit_id      TEXT NOT NULL PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    cue_set     TEXT NOT NULL,  -- JSON array of cue_ids
    weight      REAL NOT NULL DEFAULT 0.0,
    fire_count  INTEGER NOT NULL DEFAULT 0,
    use_count   INTEGER NOT NULL DEFAULT 0,
    candidate   INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sit_memory
    ON situation_edges(memory_id, user_id, agent_id);

-- ── Category taxonomy ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS categories (
    cat_id          TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    is_protected    INTEGER NOT NULL DEFAULT 0,
    is_emergent     INTEGER NOT NULL DEFAULT 0,
    decay_rate      REAL NOT NULL DEFAULT 0.02,
    relevance_floor REAL NOT NULL DEFAULT 0.0,
    activation_score REAL NOT NULL DEFAULT 1.0,  -- decays each nightly cycle
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (cat_id, user_id, agent_id)
);

-- ── Turn log (for Hebbian / situation crystallization) ───────────────────────
CREATE TABLE IF NOT EXISTS turn_log (
    turn_id         TEXT NOT NULL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    memories_served TEXT NOT NULL DEFAULT '[]',  -- JSON array
    memories_used   TEXT NOT NULL DEFAULT '[]',  -- JSON array
    cues_fired      TEXT NOT NULL DEFAULT '[]',  -- JSON array
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turn_pair
    ON turn_log(user_id, agent_id, created_at DESC);

-- ── Processed turn ids (idempotency) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS processed_turns (
    turn_id     TEXT NOT NULL PRIMARY KEY,
    processed_at INTEGER NOT NULL
);

-- ── System flags (master switches, nightly timestamp) ────────────────────────
CREATE TABLE IF NOT EXISTS system_flags (
    flag_key    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    flag_value  TEXT NOT NULL,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (flag_key, user_id, agent_id)
);
"""

SEED_CATEGORIES_SQL = """
INSERT OR IGNORE INTO categories
    (cat_id, user_id, agent_id, is_protected, is_emergent, decay_rate, relevance_floor, activation_score, created_at)
VALUES
    ('constraints',   :u, :a, 1, 0, 0.001, 0.80, 1.0, :ts),
    ('identity',      :u, :a, 1, 0, 0.001, 0.70, 1.0, :ts),
    ('preferences',   :u, :a, 1, 0, 0.02,  0.60, 1.0, :ts),
    ('context',       :u, :a, 1, 0, 0.02,  0.50, 1.0, :ts),
    ('relationships', :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('tasks',         :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('goals',         :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('skills',        :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('habits',        :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('knowledge',     :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('events',        :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('communication', :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('health',        :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('finance',       :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts),
    ('routines',      :u, :a, 0, 0, 0.02,  0.0,  1.0, :ts);
"""

SEED_FLAGS_SQL = """
INSERT OR IGNORE INTO system_flags (flag_key, user_id, agent_id, flag_value, updated_at)
VALUES
    ('system_enabled',    :u, :a, 'true', :ts),
    ('formation_enabled', :u, :a, 'true', :ts),
    ('last_nightly',      :u, :a, '0',    :ts),
    ('max_memories',      :u, :a, '0',    :ts);
"""
