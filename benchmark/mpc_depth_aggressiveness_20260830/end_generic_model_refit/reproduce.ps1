param(
    [string]$PythonExe = "C:\Users\yaman\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [switch]$Recollect
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
Set-Location $repoRoot

$collector = "src/tools/probcut/collect_end_probcut_samples.py"
$corpus = "benchmark/mpc_quantile_stage1_202608_fresh_pool/corpus.jsonl"
$exe = "bin/end_mpc_context_tool_v2.exe"
$base = "benchmark/mpc_depth_aggressiveness_20260830/end_generic_model_refit"
$exclusions = @(
    "benchmark/end_mpc_enriched_dev_743/samples.jsonl",
    "benchmark/end_mpc_enriched_dev_202607/samples.jsonl",
    "benchmark/end_mpc_enriched_dev_202608a/samples.jsonl",
    "benchmark/end_mpc_enriched_dev_202608b/samples.jsonl",
    "benchmark/end_mpc_enriched_holdout_202606/samples.jsonl",
    "benchmark/end_mpc_v2_admission_dev/samples.jsonl"
)

function Add-Exclusions([System.Collections.Generic.List[string]]$Arguments) {
    foreach ($path in $exclusions) {
        $Arguments.Add("--exclude-samples")
        $Arguments.Add($path)
    }
}

if ($Recollect) {
    $highArguments = [System.Collections.Generic.List[string]]@(
        $collector, "--positions", $corpus, "--exe", $exe,
        "--output", "$base/high_depth_samples", "--resume",
        "--seed", "202608302", "--descendant-empty", "22",
        "--descendants-per-root", "1", "--mpc-levels", "0,1,2",
        "--min-deep", "19", "--max-deep", "22",
        "--trace-contexts-per-depth", "4", "--shallow-offsets", "0",
        "--max-samples", "4000", "--workers", "8",
        "--trace-timeout", "60", "--score-timeout", "120",
        "--max-roots-per-empty", "10"
    )
    Add-Exclusions $highArguments
    & $PythonExe @highArguments

    $deepArguments = [System.Collections.Generic.List[string]]@(
        $collector, "--positions", $corpus, "--exe", $exe,
        "--output", "$base/deep23_26_samples", "--resume",
        "--seed", "202608303", "--descendant-empty", "26",
        "--descendants-per-root", "1", "--mpc-levels", "0,1,2",
        "--min-deep", "23", "--max-deep", "26",
        "--trace-contexts-per-depth", "2", "--shallow-offsets", "0",
        "--max-samples", "800", "--workers", "8",
        "--trace-timeout", "90", "--score-timeout", "180",
        "--max-roots-per-empty", "5",
        "--exclude-samples", "$base/high_depth_samples/samples.jsonl"
    )
    Add-Exclusions $deepArguments
    & $PythonExe @deepArguments
}

& $PythonExe "$base/collect_known_deep30.py" `
    --exe $exe `
    --positions bin/problem/endgame_test_20.txt `
    --exact benchmark/end_mpc_v2_representative_8_exact.tsv `
    --output "$base/deep30_known_samples.jsonl"

& $PythonExe "$base/collect_trace_usage.py" `
    --exe $exe `
    --positions bin/problem/endgame_test_20.txt `
    --output "$base/trace_usage" `
    --limit 8 `
    --mpc-level 0 `
    --per-depth-cap 500 `
    --min-depth 19 `
    --max-depth 30 `
    --timeout 120

& $PythonExe "$base/run_analysis.py"
