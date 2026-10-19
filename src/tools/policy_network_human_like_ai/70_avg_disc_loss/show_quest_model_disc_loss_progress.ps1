param([string]$Output = (Join-Path $PSScriptRoot 'output\quest_finetune_models_800_2200_l10'))
$Host.UI.RawUI.WindowTitle = 'Quest fine-tuned models | L10 average disc loss | progress'
function Format-Duration([double]$Seconds) {
    [TimeSpan]::FromSeconds([Math]::Max(0, $Seconds)).ToString('hh\:mm\:ss')
}
function Read-SharedStatus([string]$Path) {
    # Allow the producer to atomically replace the file while we read it.
    $taskStream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $taskReader = [System.IO.StreamReader]::new($taskStream, [System.Text.Encoding]::UTF8)
    try { $taskText = $taskReader.ReadToEnd() }
    finally { $taskReader.Dispose() }
    return ($taskText | ConvertFrom-Json)
}
while ($true) {
    try { $taskStatus = Read-SharedStatus (Join-Path $Output 'status.json') }
    catch { Start-Sleep -Seconds 2; continue }
    Clear-Host
    Write-Host 'Quest fine-tuned models: average disc loss' -ForegroundColor Cyan
    Write-Host 'Reference: Egaroucid L10 | Test ratings: [800, 2200)'
    Write-Host 'Models: 9 | All original test positions: 538,036'
    Write-Host "Status: $($taskStatus.status)"
    Write-Host ''
    if ($taskStatus.status -eq 'complete') {
        Import-Csv -LiteralPath (Join-Path $Output 'model_avg_disc_loss.csv') -Encoding UTF8 |
            Select-Object training_band, adopted_epoch, positions, mean_disc_loss | Format-Table -AutoSize
        Write-Host "Saved results: $Output" -ForegroundColor Green
        break
    }
    if ($taskStatus.status -eq 'failed') {
        Write-Host $taskStatus.error -ForegroundColor Red
        Get-Content -LiteralPath (Join-Path $Output 'error.log') -Encoding UTF8 -Tail 20 -ErrorAction SilentlyContinue
        break
    }
    $percent = 100.0 * $taskStatus.completed_unique_positions / $taskStatus.unique_positions
    Write-Host ('Evaluated unique boards: {0:N0} / {1:N0} ({2:N2}%)' -f $taskStatus.completed_unique_positions, $taskStatus.unique_positions, $percent)
    Write-Host ('Workers: {0} | Speed: {1:N2} boards/sec' -f $taskStatus.workers, $taskStatus.unique_positions_per_sec)
    if ($null -ne $taskStatus.eta_sec) {
        Write-Host ('Estimated time remaining: ' + (Format-Duration $taskStatus.eta_sec)) -ForegroundColor Yellow
    }
    Write-Host ('Elapsed since resume: ' + (Format-Duration $taskStatus.elapsed_sec))
    Write-Host "Calculation PID: $($taskStatus.pid)"
    Write-Host ''
    Write-Host 'The calculation continues independently of this progress window.'
    Write-Host "Output: $Output"
    Start-Sleep -Seconds 5
}
