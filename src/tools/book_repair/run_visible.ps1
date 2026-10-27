param(
    [Parameter(Mandatory=$true)][string]$WorkDir,
    [int]$Threads = 24,
    [int]$HashLevel = 25,
    [int]$Limit = 0,
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
Set-Location -LiteralPath $repo
$runArguments = @('-u', (Join-Path $PSScriptRoot 'run_repair.py'), '--work-dir', $WorkDir,
                  '--threads', $Threads, '--hash-level', $HashLevel, '--limit', $Limit)
if ($Install) {
    $runArguments += @('--install-to', (Join-Path $repo 'bin/resources/book.egbk3'),
                       '--install-to', (Join-Path $repo 'src/tools/release_script/format_files/0_common_files/book.egbk3'),
                       '--install-to', (Join-Path $repo 'book/3_joined/egaroucid_book_joined.egbk3'))
}
Write-Host 'Full book leaf rebuild. Close this console to stop; rerun the same command to resume.'
& python @runArguments
$repairExit = $LASTEXITCODE
Write-Host "Repair exited with code $repairExit. Logs and checkpoints: $WorkDir"
Read-Host 'Press Enter to close'
exit $repairExit
