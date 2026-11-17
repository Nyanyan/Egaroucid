param(
    [string]$Python = 'python',
    [ValidateSet('train','loss','all')][string]$Stage = 'all',
    [ValidateRange(1,8)][int]$Workers = 8,
    [switch]$ResumeCompatibleCode
)
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'Quest [800,1000) rule policy experiment'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$taskOutput = Join-Path $PSScriptRoot 'output/quest_800-1000'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskLog = Join-Path $taskOutput ('visible_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Write-Host 'Rule softmax | train 560,909 | validation 31,087 | test 30,423' -ForegroundColor Cyan
Write-Host 'Progress includes elapsed time and ETA. Outputs and logs:' $taskOutput
Write-Host 'Ctrl+C or closing this window stops this run and its child engines.'
Write-Host 'Rerun loss to resume saved L21 scores. Training restarts optimization and reuses verified features.'
try {
    if ($Stage -in @('train','all')) {
        & $Python -u (Join-Path $PSScriptRoot 'train.py') 2>&1 | Tee-Object -FilePath $taskLog -Append
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $LASTEXITCODE" }
    }
    if ($Stage -in @('loss','all')) {
        $taskLossArgs = @('-u', (Join-Path $PSScriptRoot 'evaluate_loss.py'), '--workers', $Workers)
        if ($ResumeCompatibleCode) { $taskLossArgs += '--resume-compatible-code' }
        & $Python @taskLossArgs 2>&1 | Tee-Object -FilePath $taskLog -Append
        if ($LASTEXITCODE -ne 0) { throw "L21 evaluation failed: $LASTEXITCODE" }
        & $Python -u (Join-Path $PSScriptRoot 'export_results.py') 2>&1 | Tee-Object -FilePath $taskLog -Append
        if ($LASTEXITCODE -ne 0) { throw "Result export failed: $LASTEXITCODE" }
    }
    Write-Host 'COMPLETE' -ForegroundColor Green
} catch {
    Write-Host $_ -ForegroundColor Red
} finally {
    Write-Host ('Log: ' + $taskLog)
    Read-Host 'Press Enter to close'
}
