param(
    [Parameter(Mandatory=$true)][string]$QuestRoot,
    [int]$Workers = 24,
    [int]$Level = 10,
    [double]$TimeoutSec = 86400,
    [string]$Output = ''
)
if (-not $Output) { $Output = Join-Path $PSScriptRoot "output\quest_test_l$Level" }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$pythonExe = if ($env:POLICY_PYTHON) { $env:POLICY_PYTHON } else { 'python' }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Host.UI.RawUI.WindowTitle = "Quest test average disc loss | Egaroucid L$Level | ETA"
"Run started: $(Get-Date -Format o) | level=$Level | workers=$Workers" | Tee-Object -FilePath (Join-Path $Output 'run.log') -Append
& $pythonExe -u (Join-Path $PSScriptRoot 'calculate_quest_avg_disc_loss.py') --quest-root $QuestRoot --level $Level --workers $Workers --timeout-sec $TimeoutSec --output $Output --progress-seconds 5 2>&1 | Tee-Object -FilePath (Join-Path $Output 'run.log') -Append
Write-Host "Python exit code: $LASTEXITCODE"
if ($LASTEXITCODE -eq 0) {
    & $pythonExe -u (Join-Path $PSScriptRoot 'verify_quest_avg_disc_loss.py') --output $Output 2>&1 | Tee-Object -FilePath (Join-Path $Output 'verification.log')
}
