$Host.UI.RawUI.WindowTitle = 'Quest fine-tuned models | L10 average disc loss | running'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$taskOutput = Join-Path $PSScriptRoot 'output\quest_finetune_selected_own_band_l10'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskLog = Join-Path $taskOutput ('run_visible_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host 'Quest fine-tuned models: average disc loss' -ForegroundColor Cyan
Write-Host 'Reference: Egaroucid L10 | Each model on its own rating band | 9 models | 549,450 test positions'
Write-Host 'Resuming saved scores. Progress updates every 5 seconds.'
Write-Host 'Ctrl+C or closing this console stops the calculation and its engines. Rerun this script to resume.'
& 'C:\Users\yaman\AppData\Local\Programs\Python\Python39\python.exe' -u (Join-Path $PSScriptRoot 'calculate_quest_selected_move_loss.py') --workers 24 2>&1 | Tee-Object -FilePath $taskLog
Write-Host ('Python exit code: ' + $LASTEXITCODE)
Write-Host ('Output: ' + $taskOutput)
