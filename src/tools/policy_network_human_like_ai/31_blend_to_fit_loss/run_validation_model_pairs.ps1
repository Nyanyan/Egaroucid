param([string]$Band = '1400-1600', [int]$Workers = 24, [int]$MaxPositions = 0,
      [int]$AlphaIntervals = 1000, [string]$Python = 'python')
$Host.UI.RawUI.WindowTitle = 'Quest validation | FT pairs with 1000-1200 | fit alpha'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$taskOutput = Join-Path $PSScriptRoot 'output'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskLog = Join-Path $taskOutput ('validation_pairs_' + $Band + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host ('Validation band ' + $Band + ' | 1200-1400 -> 1000-1200 and 1400-1600 -> 1000-1200') -ForegroundColor Cyan
Write-Host 'Alpha=0 higher-band FT, alpha=1 1000-1200 FT. L10, warm TT, 1 search thread per engine.'
Write-Host 'Saved exact board/move analyses are reused. Progress and ETA are printed and logged.'
Write-Host 'Ctrl+C or closing stops engines. Rerun to resume. Training evaluation stays stopped.'
& $Python -u (Join-Path $PSScriptRoot 'fit_validation_model_pairs.py') --band $Band --workers $Workers --max-positions $MaxPositions --alpha-intervals $AlphaIntervals 2>&1 | Tee-Object -FilePath $taskLog
$taskExit = $LASTEXITCODE
Write-Host ('Python exit code: ' + $taskExit)
Set-Content -LiteralPath ($taskLog + '.exit_code.txt') -Value $taskExit
Write-Host ('Output: ' + $taskOutput)
