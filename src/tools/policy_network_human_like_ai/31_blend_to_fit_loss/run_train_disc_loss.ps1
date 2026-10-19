param(
    [string]$Band = '1400-1600',
    [int]$Workers = 24,
    [int]$MaxPositions = 0,
    [int]$PreviewPositions = 100000,
    [string]$Python = 'python'
)
$Host.UI.RawUI.WindowTitle = 'Quest train | human and two FT policies | L10 disc loss'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$taskOutput = Join-Path $PSScriptRoot 'output'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskLog = Join-Path $taskOutput ('train_' + $Band + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host ('Train band ' + $Band + ': human, target FT, lower-band FT; L10, warm TT') -ForegroundColor Cyan
Write-Host ('Uniform random preview: ' + $PreviewPositions + ' positions, then full train; same positions for all policies.')
Write-Host 'Progress, elapsed and ETA are printed and logged. Ctrl+C or closing stops engines; rerun to resume.'
Write-Host ('Log: ' + $taskLog)
& $Python -u (Join-Path $PSScriptRoot 'compare_train_disc_loss.py') --band $Band --workers $Workers --max-positions $MaxPositions --preview-positions $PreviewPositions 2>&1 | Tee-Object -FilePath $taskLog
$taskExit = $LASTEXITCODE
Write-Host ('Python exit code: ' + $taskExit)
Set-Content -LiteralPath ($taskLog + '.exit_code.txt') -Value $taskExit
Write-Host ('Output: ' + $taskOutput)
