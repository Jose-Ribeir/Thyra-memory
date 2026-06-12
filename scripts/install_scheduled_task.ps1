# Register a Windows Scheduled Task that runs run_worker_once.py every 15 minutes.
# Run once as Administrator; the task persists across reboots.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_scheduled_task.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$RunnerScript = Join-Path $ProjectDir "scripts\run_worker_once.py"

# Find the Python interpreter in the active venv (or fall back to PATH)
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Error "Python not found on PATH. Activate your venv before running this script."
    exit 1
}

Write-Host "Registering ThyraWorker scheduled task..."
Write-Host "  Python  : $PythonExe"
Write-Host "  Script  : $RunnerScript"
Write-Host "  Interval: every 15 minutes"

schtasks /create `
    /tn "ThyraWorker" `
    /tr "`"$PythonExe`" `"$RunnerScript`"" `
    /sc minute `
    /mo 15 `
    /f

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Task registered. Verify with:"
    Write-Host "  schtasks /query /tn ThyraWorker /fo LIST"
} else {
    Write-Error "schtasks failed (exit $LASTEXITCODE). Try running as Administrator."
}
