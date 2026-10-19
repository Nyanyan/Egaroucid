param([int]$Workers = 24, [int]$MaxPositions = 0, [string]$Python = 'python')
$Host.UI.RawUI.WindowTitle = 'Quest test 1400-1600 | fixed blends | disc loss and Top-1'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$taskOutput = Join-Path $PSScriptRoot 'output'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskLog = Join-Path $taskOutput ('test_pairs_1400-1600_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host '[1400,1600) test: both validation-selected alpha values are fixed.' -ForegroundColor Cyan
Write-Host 'All original test positions; L10 analyze with warm TT; symmetry-aware human Top-1.'
Write-Host 'Each engine uses 1 search thread. Shared exact board/move caches are reused.'
Write-Host 'Progress, elapsed and ETA are logged. Ctrl+C or closing stops engines; rerun to resume.'
& $Python -u (Join-Path $PSScriptRoot 'evaluate_model_pairs_test.py') --workers $Workers --max-positions $MaxPositions 2>&1 | Tee-Object -FilePath $taskLog
$taskExit = $LASTEXITCODE
Write-Host ('Python exit code: ' + $taskExit)
Set-Content -LiteralPath ($taskLog + '.exit_code.txt') -Value $taskExit
Write-Host ('Output: ' + $taskOutput)
