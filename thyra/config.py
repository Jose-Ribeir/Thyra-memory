"""Central configuration — all numeric constants live here. Tune from here only."""

import os

# ── Decay rates ────────────────────────────────────────────────────────────────
DECAY_EXPLICIT = 0.02  # ~35-day half-life for explicit saves
DECAY_CONSTRAINTS = 0.001  # near-permanent after graduation
DECAY_IDENTITY = 0.001
DECAY_SEMANTIC = 0.05  # ~14-day half-life (probationary semantic)
DECAY_EPISODIC = 0.15  # ~5-day half-life (probationary episodic)

# ── Lifecycle thresholds ───────────────────────────────────────────────────────
ARCHIVE_THRESHOLD = 0.05
HARD_DELETE_DAYS = 180
RESURRECTION_THRESHOLD = 0.15
RESURRECTION_STRENGTH = 0.10

# ── Strength bounds ────────────────────────────────────────────────────────────
BASE_STRENGTH_EXPLICIT = 1.0
BASE_STRENGTH_AUTOMATIC = 0.4
STRENGTH_CAP = 10.0

# ── Recall / scoring ──────────────────────────────────────────────────────────
SPREADING_DIRECT = 0.40
SPREADING_ASSOC = 0.15
SPREADING_SITUATION = 0.60
DISCRIMINABILITY_FLOOR = 0.15
PRESENCE_FLOOR = 0.20
SCORE_FLOOR = 0.05  # minimum score to include in injection

# ── Cue extraction ────────────────────────────────────────────────────────────
MAX_CUES_PER_TURN = 12
MIN_CUE_LENGTH = 3

# ── Token budgets (by tier) ───────────────────────────────────────────────────
TOKEN_BUDGETS: dict[str, int] = {
    "efficiency": 600,
    "balanced": 1500,
    "performance": 3000,
    "premium": 3000,
}
DEFAULT_TIER = "balanced"
TOKEN_BUDGET = TOKEN_BUDGETS[DEFAULT_TIER]
PER_MEMORY_CAP = 0.25  # max fraction of budget for one memory

# ── Category saturation (gamma) ───────────────────────────────────────────────
GAMMA_TIERS: dict[str, float] = {
    "efficiency": 0.55,
    "balanced": 0.70,
    "performance": 0.85,
    "premium": 0.85,
}
GAMMA_DEFAULT = GAMMA_TIERS[DEFAULT_TIER]

# ── Reinforcement ─────────────────────────────────────────────────────────────
REINFORCE_BASE = 0.30
SURFACED_BOOST = 0.05  # served but not declared used
OVERLAP_CONFIRMATORY_CAP = 0.10
# Overlap-as-primary reinforcement (API-level signal — no LLM declaration needed).
# When declared ∩ served is empty, term-overlap between the response and each served
# memory is the primary "was this used?" signal.
OVERLAP_PRIMARY_THRESHOLD = 0.35  # min fraction of memory words present in response
OVERLAP_INFER_MULT = 0.60  # inferred boost = REINFORCE_BASE × this

CATEGORY_MULTIPLIERS: dict[str, float] = {
    "constraints": 0.3,
    "identity": 0.5,
    "preferences": 1.0,
    "relationships": 1.2,
    "tasks": 0.8,
    "goals": 0.8,
    "context": 1.0,
    "skills": 1.1,
    "habits": 1.2,
    "communication": 1.0,
    "knowledge": 0.9,
    "events": 0.7,
    "health": 0.9,
    "finance": 0.9,
    "routines": 1.3,
}

# ── Cue graph dynamics ────────────────────────────────────────────────────────
CONTENT_SEED_CUES = 8
CONTENT_SEED_WEIGHT = 0.30
SYNONYM_CUE_WEIGHT = 0.10
SYNONYM_SIMILARITY_THRESHOLD = 0.82
SYNONYM_MAX_EXPANSIONS = 6
CUE_PROMOTE_THRESHOLD = 0.15  # candidate weight → real edge
CUE_PRUNE_MIN_FIRES = 5
CUE_PRUNE_MAX_RATE = 0.05  # (use_count / fire_count) < this → prune

# ── Association edges ─────────────────────────────────────────────────────────
HEBBIAN_WINDOW_TURNS = 3
HEBBIAN_MIN_CO_USE = 3
HEBBIAN_WEIGHT_DELTA = 0.1
ASSOC_WEIGHT_CAP = 1.0
ASSOC_NIGHTLY_DECAY = 0.99  # × each nightly cycle
ASSOC_PRUNE_THRESHOLD = 0.05

# ── Situation edges ───────────────────────────────────────────────────────────
SITUATION_MIN_FIRES = 5
SITUATION_MIN_RATE = 0.60

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORY_SOFT_CAP = 35
EMERGENT_MIN_MEMBERS = 4
EMERGENT_MIN_TURNS = 10
EMERGENT_MIN_EDGE_WEIGHT = 0.30
HUB_CUE_FRACTION = 0.50  # cue pointing to >50% of memories → hub

# ── Automatic formation ───────────────────────────────────────────────────────
SALIENCE_THRESHOLD = (
    0.32  # for user-sourced text — lowered so weak+weak signals sum to pass
)
AGENT_SALIENCE_THRESHOLD = 0.28  # for agent-sourced text (findings, fixes, facts)
NOVELTY_THRESHOLD = 0.40
DEDUP_SIMILARITY_THRESHOLD = 0.85
DEDUP_CANDIDATE_LIMIT = 10

# ── Formation precision (transience / distillation) ─────────────────────────────
# L2 distill-or-drop judge: advisory small-model pass that rewrites a clause into a
# clean atomic fact or drops it. Default OFF until validated (mirrors the MiniLM
# refiner's prewarm-or-skip discipline; never blocks the worker thread).
DISTILLER_ENABLED: bool = os.environ.get("THYRA_DISTILLER", "false").lower() == "true"
# L3 weak-admit horizon: admits that passed on weak signal alone (no directive, no
# confirmed reference, salience just over threshold) start on the steepest episodic
# decay regardless of category, so an unused borderline admit fades in days.
WEAK_ADMIT_DECAY = DECAY_EPISODIC  # 0.15, ~5-day half-life
# L3 A-Mem-style auto-purge: probationary + use_count==0 + older than this window is
# archived in the nightly sweep without waiting for the slow strength curve.
PROBATIONARY_AUTOPURGE_DAYS = 7

# ── Recall intent amplification ───────────────────────────────────────────────
RECALL_INTENT_BUDGET_MULT = 1.5
RECALL_INTENT_GAMMA_RELAX = 0.50
RECALL_INTENT_ARCHIVE_LIMIT = 5

# ── Hot cache ─────────────────────────────────────────────────────────────────
HOT_CACHE_TTL_SECONDS = 3600

# ── Background worker ─────────────────────────────────────────────────────────
WORKER_POLL_SECONDS = 2
NIGHTLY_INTERVAL_HOURS = 24
TURN_LOG_RETENTION_DAYS = 90

# ── Paths (overridable via env) ───────────────────────────────────────────────
THYRA_DB_PATH: str = os.environ.get(
    "THYRA_DB_PATH",
    r"J:\codigo\Memory_llm\data\thyra.db",
)
THYRA_USER_ID: str = os.environ.get("THYRA_USER_ID", "default")
THYRA_AGENT_ID: str = os.environ.get("THYRA_AGENT_ID", "claude-code-global")
THYRA_LOG_PATH: str = os.environ.get(
    "THYRA_LOG_PATH",
    r"J:\codigo\Memory_llm\data\thyra.log",
)

# ── Master switches ───────────────────────────────────────────────────────────
SYSTEM_ENABLED_DEFAULT = True
FORMATION_ENABLED_DEFAULT = True
SYNONYM_EXPANSION_ENABLED = True

# ── Injection limit ───────────────────────────────────────────────────────────
# 0 = unlimited (token budget governs). Any positive int caps the count first.
MAX_MEMORIES_INJECTED: int = 0

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_ENABLED: bool = os.environ.get("THYRA_DASHBOARD", "true").lower() != "false"
DASHBOARD_PORT: int = int(os.environ.get("THYRA_DASHBOARD_PORT", "7432"))
DASHBOARD_HOST: str = os.environ.get("THYRA_DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_AUTO_OPEN: bool = (
    os.environ.get("THYRA_DASHBOARD_AUTO_OPEN", "true").lower() != "false"
)
DASHBOARD_LOG_BUFFER_SIZE: int = 500

# ── System tray ────────────────────────────────────────────────────────────────
THYRA_TRAY_ENABLED: bool = os.environ.get("THYRA_TRAY", "true").lower() != "false"
