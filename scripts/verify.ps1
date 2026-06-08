# Thyra Memory — smoke test / verification script
# Sends a fake UserPromptSubmit event to pre_turn.py and checks the output.

$ErrorActionPreference = "Stop"
$projectRoot = "J:\codigo\Memory_llm"
$dataDir     = "$projectRoot\data"

Write-Host "=== Thyra Smoke Test ===" -ForegroundColor Cyan

$env:THYRA_DB_PATH  = "$dataDir\thyra.db"
$env:THYRA_USER_ID  = "default"
$env:THYRA_AGENT_ID = "claude-code-global"

# Test 1: pre_turn hook returns valid JSON with additionalContext key
$fakeEvent = '{"session_id":"smoke-test","prompt":"Tell me about Python programming","cwd":"."}'
$output = $fakeEvent | python "$projectRoot\thyra\hooks\pre_turn.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] pre_turn.py exited non-zero" -ForegroundColor Red; exit 1
}
try {
    $parsed = $output | ConvertFrom-Json
    if ($null -eq $parsed.additionalContext) {
        Write-Host "[FAIL] pre_turn.py output missing 'additionalContext' key" -ForegroundColor Red; exit 1
    }
    Write-Host "[OK] pre_turn.py returned valid JSON with additionalContext"
} catch {
    Write-Host "[FAIL] pre_turn.py output is not valid JSON: $output" -ForegroundColor Red; exit 1
}

# Test 2: stop_hook exits 0 with empty input
$emptyEvent = '{}'
$emptyEvent | python "$projectRoot\thyra\hooks\stop_hook.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] stop_hook.py exited non-zero on empty input" -ForegroundColor Red; exit 1
}
Write-Host "[OK] stop_hook.py handles empty input gracefully"

# Test 3: DB is accessible
$dbCheck = python -c "
from thyra.db.connection import get_conn
conn = get_conn()
n = conn.execute('SELECT COUNT(*) FROM categories').fetchone()[0]
print(f'categories: {n}')
"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] DB check failed" -ForegroundColor Red; exit 1
}
Write-Host "[OK] DB check: $dbCheck"

# Test 4: Full recall pipeline with seeded memory
$pipelineCheck = python -c "
import os
os.environ['THYRA_DB_PATH'] = r'$dataDir\thyra.db'
from thyra.db.connection import get_conn
from thyra.models.memory import create_memory, upsert_cue_edge
from thyra.recall.intent import recall_pipeline
from thyra.recall.cache import HOT_CACHE

conn = get_conn()
mid = create_memory(conn, 'I prefer dark mode in all editors', 'preferences')
upsert_cue_edge(conn, 'dark', mid, 'default', 'claude-code-global', weight=0.8)
HOT_CACHE.clear()

xml, served = recall_pipeline(conn, 'default', 'claude-code-global',
    'What are my editor preferences?', 'smoke', 'smoke-turn-1')
print('served:', served)
print('xml_len:', len(xml))
"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Recall pipeline check failed" -ForegroundColor Red; exit 1
}
Write-Host "[OK] Recall pipeline: $pipelineCheck"

Write-Host ""
Write-Host "=== All smoke tests passed ===" -ForegroundColor Green
