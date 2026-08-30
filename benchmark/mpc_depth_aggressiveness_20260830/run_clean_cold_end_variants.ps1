[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateRange(1000, 600000)]
    [int]$MoveTimeMilliseconds = 15000,
    [ValidateRange(1, 300)]
    [int]$CaseTimeoutSeconds = 105
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$experimentDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $experimentDirectory "../..")
)
$runner = Join-Path $repositoryRoot "bin/cold_endgame_benchmark.py"
$positions = Join-Path $repositoryRoot `
    "bin/problem/cold_endgame_2026-08-23_egrcd.txt"

$variants = @(
    [pscustomobject]@{
        name = "cold_end_runtime_control"
        executable = "bin/console_mpc_model_control.exe"
    },
    [pscustomobject]@{
        name = "cold_end_runtime_depth_m2"
        executable = "bin/console_end_depth_m2.exe"
    },
    [pscustomobject]@{
        name = "cold_end_runtime_depth_refit0"
        executable = "bin/console_end_depth_p0.exe"
    },
    [pscustomobject]@{
        name = "cold_end_runtime_depth_p2"
        executable = "bin/console_end_depth_p2.exe"
    },
    [pscustomobject]@{
        name = "cold_end_runtime_depth_p4"
        executable = "bin/console_end_depth_p4.exe"
    },
    [pscustomobject]@{
        name = "cold_end_runtime_model_rootconstrained"
        executable = "bin/console_end_mpc_model_rootconstrained.exe"
    },
    [pscustomobject]@{
        name = "cold_end_runtime_model_domainconstrained"
        executable = "bin/console_end_mpc_model_domainconstrained.exe"
    }
)

Push-Location $repositoryRoot
try {
    foreach ($variant in $variants) {
        $outputDirectory = Join-Path $experimentDirectory $variant.name
        Write-Host ("[start] {0}" -f $variant.name)
        & $Python $runner `
            --positions $positions `
            --min-empty 32 `
            --max-empty 44 `
            --max-per-empty 5 `
            --seed 20260823 `
            --exe $variant.executable `
            --threads 20 `
            --hash 29 `
            --movetime-ms $MoveTimeMilliseconds `
            --required-selectivity 74 `
            --case-timeout-seconds $CaseTimeoutSeconds `
            --output $outputDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "cold endgame benchmark failed: $($variant.name), exit=$LASTEXITCODE"
        }
        Write-Host ("[done]  {0}" -f $variant.name)
    }
}
finally {
    Pop-Location
}
