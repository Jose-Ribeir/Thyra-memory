import os, sys

sys.path.insert(0, r"J:\codigo\Memory_llm")
os.environ.setdefault("THYRA_DB_PATH", r"J:\codigo\Memory_llm\data\thyra.db")
os.environ.setdefault("THYRA_USER_ID", "default")
os.environ.setdefault("THYRA_AGENT_ID", "claude-code-global")

from thyra.db.connection import get_conn

conn = get_conn()
ver = conn.execute("SELECT version FROM schema_version").fetchone()[0]
cats = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
flags = conn.execute("SELECT flag_key, flag_value FROM system_flags").fetchall()
print(f"DB OK — schema v{ver}, {cats} categories seeded")
for f in flags:
    print(f"  {f[0]} = {f[1]}")
