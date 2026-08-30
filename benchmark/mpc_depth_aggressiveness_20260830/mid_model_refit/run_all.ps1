param(
    [string]$Python = "python",
    [string]$Compiler = "clang++",
    [switch]$SkipHoldoutCollection,
    [double[]]$Ridge = @(0.0001, 0.01, 1.0, 3.0, 10.0)
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Output = $PSScriptRoot

if (-not $SkipHoldoutCollection) {
    $CollectorSource = Join-Path $Output "collect_shallow_only.cpp"
    $Tool = Join-Path $Repo "bin/collect_shallow_only.exe"
    & $Compiler -O3 -mtune=native -march=native -pthread -std=c++20 $CollectorSource -o $Tool -lws2_32
    if ($LASTEXITCODE -ne 0) {
        throw "collect_shallow_only build failed: exit $LASTEXITCODE"
    }
    $Problem = (Resolve-Path (Join-Path $Repo "bin/problem/midgame_ggs_holdout_20260827.txt")).Path
    $Jobs = @(
        @{ Depth = 12; Shallow = "0,2,4,6,8"; Limit = 150; Reference = "mid_probcut_holdout_d12_20260827.tsv" },
        @{ Depth = 13; Shallow = "0,1,3,5,7,9"; Limit = 150; Reference = "mid_probcut_holdout_d13_20260827.tsv" },
        @{ Depth = 14; Shallow = "0,2,4,6,8"; Limit = 150; Reference = "mid_probcut_holdout_d14_20260827.tsv" },
        @{ Depth = 15; Shallow = "0,3,5,7,9,11"; Limit = 75; Reference = "mid_probcut_holdout_d15_20260827.tsv" },
        @{ Depth = 16; Shallow = "0,2,4,6,8,10"; Limit = 75; Reference = "mid_probcut_holdout_d16_20260827.tsv" }
    )
    Push-Location (Join-Path $Repo "bin")
    try {
        foreach ($Job in $Jobs) {
            $Destination = Join-Path $Output ("holdout_all_d{0}.tsv" -f $Job.Depth)
            $Reference = Join-Path $Repo ("benchmark/{0}" -f $Job.Reference)
            & $Tool $Problem $Reference $Job.Depth $Job.Shallow 20 $Job.Limit |
                Set-Content -Encoding utf8 $Destination
            if ($LASTEXITCODE -ne 0) {
                throw "dataset tool failed at depth $($Job.Depth): exit $LASTEXITCODE"
            }
        }
    }
    finally {
        Pop-Location
    }
}

foreach ($Value in $Ridge) {
    & $Python (Join-Path $Output "fit_mid_model.py") --ridge $Value
    if ($LASTEXITCODE -ne 0) {
        throw "fit_mid_model.py failed at ridge ${Value}: exit $LASTEXITCODE"
    }
    $Tag = ("{0:g}" -f $Value).Replace(".", "p")
    Copy-Item -Force (Join-Path $Output "fit_results.json") (Join-Path $Output "fit_results_ridge_${Tag}.json")
}

& $Python (Join-Path $Output "summarize_results.py")
if ($LASTEXITCODE -ne 0) {
    throw "summarize_results.py failed: exit $LASTEXITCODE"
}
