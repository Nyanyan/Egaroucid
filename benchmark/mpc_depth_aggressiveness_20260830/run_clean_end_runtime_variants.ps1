[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$experimentDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $experimentDirectory "../..")
)
$runner = Join-Path $repositoryRoot `
    "benchmark/end_mpc_shallow_scale_runtime_20260829/run_direct_compare.py"
$positions = Join-Path $repositoryRoot "bin/problem/endgame_test_20.txt"
$exact = Join-Path $repositoryRoot "benchmark/end_mpc_v2_representative_8_exact.tsv"

$variants = @(
    [pscustomobject]@{
        name = "end_runtime_clean_depth_m2"
        label = "depth_m2"
        control = "bin/end_mpc_control.exe"
        candidate = "bin/end_mpc_depth_m2.exe"
    },
    [pscustomobject]@{
        name = "end_runtime_clean_depth_refit0"
        label = "depth_refit0"
        control = "bin/end_mpc_control.exe"
        candidate = "bin/end_mpc_depth_refit0.exe"
    },
    [pscustomobject]@{
        name = "end_runtime_clean_depth_p2"
        label = "depth_p2"
        control = "bin/end_mpc_control.exe"
        candidate = "bin/end_mpc_depth_p2.exe"
    },
    [pscustomobject]@{
        name = "end_runtime_clean_depth_p4"
        label = "depth_p4"
        control = "bin/end_mpc_control.exe"
        candidate = "bin/end_mpc_depth_p4.exe"
    },
    [pscustomobject]@{
        name = "end_generic_runtime_clean_rootconstrained"
        label = "end_model_root"
        control = "bin/end_mpc_model_control.exe"
        candidate = "bin/end_mpc_model_rootconstrained.exe"
    },
    [pscustomobject]@{
        name = "end_generic_runtime_clean_domainconstrained"
        label = "end_model_domain"
        control = "bin/end_mpc_model_control.exe"
        candidate = "bin/end_mpc_model_domainconstrained.exe"
    }
)

Push-Location $repositoryRoot
try {
    foreach ($variant in $variants) {
        $outputDirectory = Join-Path $experimentDirectory $variant.name
        Write-Host ("[start] {0}" -f $variant.label)
        & $Python $runner `
            --control $variant.control `
            --candidate $variant.candidate `
            --candidate-label $variant.label `
            --positions $positions `
            --start 0 `
            --count 8 `
            --repetitions 1 `
            --levels "0,1,2" `
            --hash-level 29 `
            --timeout-seconds $TimeoutSeconds `
            --exact $exact `
            --output-dir $outputDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "runtime comparison failed: $($variant.label), exit=$LASTEXITCODE"
        }
        Write-Host ("[done]  {0}" -f $variant.label)
    }
}
finally {
    Pop-Location
}
