"""Smoke test: simulate pre_turn.py hook call and stop_hook.py call."""

import json, os, subprocess, sys, tempfile

os.environ["THYRA_DB_PATH"] = r"J:\codigo\Memory_llm\data\thyra.db"
os.environ["THYRA_USER_ID"] = "default"
os.environ["THYRA_AGENT_ID"] = "claude-code-global"

python = r"C:\Users\josep\AppData\Local\Programs\Python\Python310\python.exe"

# ── pre_turn.py smoke test ────────────────────────────────────────────────────
event = {
    "session_id": "smoke_test_session",
    "transcript_path": "",
    "prompt": "I always prefer tabs over spaces in Python code",
    "cwd": r"J:\codigo\Memory_llm",
}
result = subprocess.run(
    [python, r"J:\codigo\Memory_llm\thyra\hooks\pre_turn.py"],
    input=json.dumps(event),
    capture_output=True,
    text=True,
    env=os.environ,
)
print(f"pre_turn exit code: {result.returncode}")
if result.stderr.strip():
    print(f"pre_turn stderr: {result.stderr.strip()[:500]}")

try:
    out = json.loads(result.stdout)
    assert "additionalContext" in out, f"Missing additionalContext key: {out}"
    print(
        f"pre_turn output OK — additionalContext length: {len(out['additionalContext'])}"
    )
except Exception as e:
    print(f"pre_turn output INVALID: {e}")
    print(f"  stdout was: {result.stdout[:300]}")
    sys.exit(1)

# ── stop_hook.py smoke test ───────────────────────────────────────────────────
# Write a fake transcript so stop_hook has something to parse
transcript = [
    {"role": "user", "content": "I always prefer tabs over spaces"},
    {"role": "assistant", "content": "Noted! <memories_used></memories_used>"},
]
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
    json.dump(transcript, tf)
    transcript_path = tf.name

stop_event = {
    "session_id": "smoke_test_session",
    "transcript_path": transcript_path,
}
result2 = subprocess.run(
    [python, r"J:\codigo\Memory_llm\thyra\hooks\stop_hook.py"],
    input=json.dumps(stop_event),
    capture_output=True,
    text=True,
    env=os.environ,
)
os.unlink(transcript_path)
print(f"stop_hook exit code: {result2.returncode}")
if result2.stderr.strip():
    print(f"stop_hook stderr: {result2.stderr.strip()[:500]}")
print("stop_hook OK" if result2.returncode == 0 else "stop_hook FAILED")

# ── check delta file was written ─────────────────────────────────────────────
import pathlib, time

queue = pathlib.Path(r"J:\codigo\Memory_llm\data\delta_queue")
deltas = sorted(queue.glob("*.json"))
if deltas:
    latest = deltas[-1]
    data = json.loads(latest.read_text())
    print(f"Delta file written: {latest.name}")
    print(f"  user_id={data['user_id']} agent_id={data['agent_id']}")
    print(f"  raw_user_text preview: {data.get('raw_user_text', '')[:60]}")
else:
    print("No delta files found in queue (may be normal if session state missing)")

print("\nSmoke test PASSED")
