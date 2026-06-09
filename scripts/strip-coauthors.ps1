# Strip Co-Authored-By trailers from unpushed commits by replaying them.
# Safe to run repeatedly; skips when no unpushed commits or no co-author trailers.
param(
    [string]$Upstream = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot

if (-not $Upstream) {
    $Upstream = git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
    if (-not $Upstream) {
        $Upstream = "origin/master"
    }
}

$commits = @(git rev-list --reverse "$Upstream..HEAD" 2>$null)
if ($commits.Count -eq 0) {
    Write-Host "strip-coauthors: no unpushed commits ($Upstream..HEAD)."
    exit 0
}

$hasCoauthor = $false
foreach ($c in $commits) {
    $trailer = git log -1 --format="%(trailers:key=Co-Authored-By,valueonly)" $c
    if ($trailer) {
        $hasCoauthor = $true
        break
    }
}
if (-not $hasCoauthor) {
    Write-Host "strip-coauthors: no Co-Authored-By trailers in unpushed commits."
    exit 0
}

$branch = git rev-parse --abbrev-ref HEAD
$backup = "coauthor-strip-backup"
$cleanBranch = "coauthor-strip-temp"

Write-Host "strip-coauthors: removing Co-Authored-By from $($commits.Count) commit(s) on $branch..."

git branch -f $backup HEAD | Out-Null
git checkout -B $cleanBranch $Upstream | Out-Null

foreach ($c in $commits) {
    git cherry-pick $c | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "strip-coauthors: cherry-pick failed at $c"
        git cherry-pick --abort 2>$null
        git checkout $branch 2>$null
        git branch -D $cleanBranch 2>$null
        git branch -D $backup 2>$null
        exit 1
    }
    $body = git log -1 --format=%B
    $lines = $body -split "`r?`n" | Where-Object { $_ -notmatch '^\s*Co-Authored-By:' }
    $clean = ($lines -join [Environment]::NewLine).TrimEnd() + [Environment]::NewLine
    $msgFile = Join-Path $repoRoot ".git/coauthor-strip-msg"
    [System.IO.File]::WriteAllText($msgFile, $clean)
    git commit --amend -F .git/coauthor-strip-msg | Out-Null
}

git branch -f $branch $cleanBranch | Out-Null
git checkout $branch | Out-Null
git branch -D $cleanBranch | Out-Null
git branch -D $backup | Out-Null

Write-Host "strip-coauthors: done."
