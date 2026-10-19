param(
    [string]$Band = '1400-1600',
    [int]$Workers = 24,
    [int]$MaxPositions = 0,
    [int]$AlphaIntervals = 1000,
    [switch]$AllBands,
    [string]$Python = 'python'
)
$Host.UI.RawUI.WindowTitle = 'Quest FT blend | fit validation disc loss | L10'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$taskOutput = Join-Path $PSScriptRoot 'output'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskLog = Join-Path $taskOutput ('run_' + $Band + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$taskArgs = @('-u', (Join-Path $PSScriptRoot 'fit_blend_to_loss.py'), '--band', $Band,
              '--workers', $Workers, '--max-positions', $MaxPositions, '--alpha-intervals', $AlphaIntervals)
if ($AllBands) { $taskArgs += '--all-bands' }
Write-Host ('Target band: ' + $Band + ' | L10 | alpha=0 target FT, alpha=1 lower-band FT') -ForegroundColor Cyan
Write-Host 'Validation human loss -> fit alpha -> fixed-alpha test loss. Progress and ETA are printed.'
Write-Host 'Ctrl+C or closing this console stops the calculation and its engines. Rerun to resume.'
Write-Host ('Log: ' + $taskLog)
& $Python @taskArgs 2>&1 | Tee-Object -FilePath $taskLog
$taskExit = $LASTEXITCODE
Write-Host ('Python exit code: ' + $taskExit)
Set-Content -LiteralPath ($taskLog + '.exit_code.txt') -Value $taskExit
Write-Host ('Output: ' + $taskOutput)
