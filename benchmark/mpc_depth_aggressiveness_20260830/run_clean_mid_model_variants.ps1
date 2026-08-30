[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$experimentDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $experimentDirectory "../..")
)
$runner = Join-Path $experimentDirectory "run_accuracy_matrix.py"
$outputDirectory = Join-Path $experimentDirectory `
    "mid_model_runtime_clean_final300"

Push-Location $repositoryRoot
try {
    & $Python $runner `
        --variant "control=bin/mid_mpc_model_control.exe" `
        --variant "m4=bin/mid_mpc_model_m4.exe" `
        --variant "m2=bin/mid_mpc_model_m2.exe" `
        --variant "p0=bin/mid_mpc_model_p0.exe" `
        --variant "p2=bin/mid_mpc_model_p2.exe" `
        --variant "p4=bin/mid_mpc_model_p4.exe" `
        --datasets final `
        --levels 0 1 2 3 4 5 `
        --depths 16 `
        --reference-depth 16 `
        --limit 300 `
        --threads 1 `
        --hash 20 `
        --repetitions 1 `
        --output $outputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "midgame model comparison failed: exit=$LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

