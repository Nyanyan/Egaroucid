[CmdletBinding()]
param(
    [string]$Compiler = "clang++",
    [string]$OutputDirectory = "",
    [ValidateRange(1, 3600)]
    [int]$CompileTimeoutSeconds = 300,
    [ValidateRange(1, 3600)]
    [int]$RunTimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-NativeProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [ValidateRange(1, 3600)]
        [int]$TimeoutSeconds = 300
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    # Windows PowerShell 5.1 uses .NET Framework, whose ProcessStartInfo does
    # not yet expose ArgumentList. None of these compiler/test arguments end in
    # a backslash, so quoting every argument is sufficient and keeps paths with
    # spaces intact.
    $quotedArguments = foreach ($argument in $ArgumentList) {
        '"' + $argument.Replace('"', '\"') + '"'
    }
    $startInfo.Arguments = $quotedArguments -join " "

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $process.Start()) {
            throw "Failed to start process: $FilePath"
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $completed = $process.WaitForExit($TimeoutSeconds * 1000)
        $timedOut = -not $completed
        if ($timedOut) {
            $process.Kill()
            $process.WaitForExit()
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $stopwatch.Stop()
        return [ordered]@{
            exit_code = if ($timedOut) { 124 } else { $process.ExitCode }
            timed_out = $timedOut
            timeout_seconds = $TimeoutSeconds
            duration_ms = $stopwatch.ElapsedMilliseconds
            stdout = $stdout.TrimEnd()
            stderr = $stderr.TrimEnd()
        }
    }
    finally {
        $stopwatch.Stop()
        $process.Dispose()
    }
}

function ConvertTo-MarkdownCell {
    param([AllowEmptyString()][string]$Value)
    if ([string]::IsNullOrEmpty($Value)) {
        return "-"
    }
    return $Value.Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Get-CompatibleRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $baseFullPath = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\", "/")
    $targetFullPath = [System.IO.Path]::GetFullPath($TargetPath).TrimEnd("\", "/")
    if ($baseFullPath.Equals($targetFullPath, [StringComparison]::OrdinalIgnoreCase)) {
        return "."
    }
    $baseUri = [Uri]::new($baseFullPath + [System.IO.Path]::DirectorySeparatorChar)
    $targetUri = [Uri]::new($targetFullPath)
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
}

$scriptDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDirectory "../.."))
$binDirectory = Join-Path $repoRoot "bin"
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $scriptDirectory "coefficient_variant_regression"
}
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
$buildDirectory = Join-Path $outputPath "build"
[void](New-Item -ItemType Directory -Force -Path $outputPath)
[void](New-Item -ItemType Directory -Force -Path $buildDirectory)

$compilerCommand = Get-Command $Compiler -ErrorAction Stop
$compilerPath = $compilerCommand.Source
$compilerVersionResult = Invoke-NativeProcess `
    -FilePath $compilerPath `
    -ArgumentList @("--version") `
    -WorkingDirectory $repoRoot `
    -TimeoutSeconds $CompileTimeoutSeconds
$compilerVersion = ($compilerVersionResult.stdout -split "`r?`n")[0]

$gitHeadResult = Invoke-NativeProcess `
    -FilePath "git" `
    -ArgumentList @("rev-parse", "HEAD") `
    -WorkingDirectory $repoRoot `
    -TimeoutSeconds $CompileTimeoutSeconds
$gitHead = if ($gitHeadResult.exit_code -eq 0) {
    $gitHeadResult.stdout.Trim()
}
else {
    "unknown"
}

$coefficientTest = Join-Path $repoRoot "src/tools/probcut/test_mpc_coefficient_variants.cpp"
$midScaleTest = Join-Path $repoRoot "src/tools/probcut/test_mid_mpc_sigma_scale.cpp"
$endModelTest = Join-Path $repoRoot "src/tools/probcut/test_end_mpc_model.cpp"

$testCases = [System.Collections.Generic.List[object]]::new()
foreach ($precalculation in 0..1) {
    $testCases.Add([pscustomobject]@{
        id = "coefficient_production_precalc_$precalculation"
        description = "係数候補: production, USE_MPC_PRE_CALCULATION=$precalculation"
        source = $coefficientTest
        defines = [string[]]@("USE_MPC_PRE_CALCULATION=$precalculation")
        run_directory = $repoRoot
    })
    foreach ($variant in 1..5) {
        $testCases.Add([pscustomobject]@{
            id = "coefficient_mid_variant_${variant}_precalc_$precalculation"
            description = "係数候補: MID_MPC_RECALIBRATED_VARIANT=$variant, USE_MPC_PRE_CALCULATION=$precalculation"
            source = $coefficientTest
            defines = [string[]]@(
                "MID_MPC_RECALIBRATED_VARIANT=$variant",
                "USE_MPC_PRE_CALCULATION=$precalculation"
            )
            run_directory = $repoRoot
        })
    }
    foreach ($variant in 1..2) {
        $testCases.Add([pscustomobject]@{
            id = "coefficient_end_sigma_variant_${variant}_precalc_$precalculation"
            description = "係数候補: END_MPC_SIGMA_MODEL_VARIANT=$variant, USE_MPC_PRE_CALCULATION=$precalculation"
            source = $coefficientTest
            defines = [string[]]@(
                "END_MPC_SIGMA_MODEL_VARIANT=$variant",
                "USE_MPC_PRE_CALCULATION=$precalculation"
            )
            run_directory = $repoRoot
        })
    }
}
foreach ($precalculation in 0..1) {
    $testCases.Add([pscustomobject]@{
        id = "mid_sigma_scale_precalc_$precalculation"
        description = "production中盤MPC: USE_MPC_PRE_CALCULATION=$precalculation"
        source = $midScaleTest
        defines = [string[]]@("USE_MPC_PRE_CALCULATION=$precalculation")
        run_directory = $binDirectory
    })
}
foreach ($variant in 0..4) {
    $testCases.Add([pscustomobject]@{
        id = "end_recalibrated_variant_$variant"
        description = "終盤MPC: END_MPC_RECALIBRATED_VARIANT=$variant"
        source = $endModelTest
        defines = [string[]]@("END_MPC_RECALIBRATED_VARIANT=$variant")
        run_directory = $repoRoot
    })
}

$startedAt = [DateTimeOffset]::Now
$results = [System.Collections.Generic.List[object]]::new()
foreach ($testCase in $testCases) {
    $executable = Join-Path $buildDirectory ($testCase.id + ".exe")
    $compileArguments = [System.Collections.Generic.List[string]]::new()
    foreach ($argument in @("-O2", "-std=c++20", "-march=native", "-pthread")) {
        $compileArguments.Add($argument)
    }
    foreach ($define in $testCase.defines) {
        $compileArguments.Add("-D$define")
    }
    $compileArguments.Add($testCase.source)
    $compileArguments.Add("-o")
    $compileArguments.Add($executable)
    $compileArguments.Add("-lws2_32")

    Write-Host ("[compile] {0}" -f $testCase.id)
    $compileResult = Invoke-NativeProcess `
        -FilePath $compilerPath `
        -ArgumentList $compileArguments.ToArray() `
        -WorkingDirectory $repoRoot `
        -TimeoutSeconds $CompileTimeoutSeconds

    $runResult = $null
    if ($compileResult.exit_code -eq 0) {
        Write-Host ("[run]     {0}" -f $testCase.id)
        $runResult = Invoke-NativeProcess `
            -FilePath $executable `
            -ArgumentList @() `
            -WorkingDirectory $testCase.run_directory `
            -TimeoutSeconds $RunTimeoutSeconds
    }

    $status = if ($compileResult.exit_code -ne 0) {
        "compile_failed"
    }
    elseif ($runResult.exit_code -ne 0) {
        "run_failed"
    }
    else {
        "passed"
    }

    $relativeSource = (Get-CompatibleRelativePath $repoRoot $testCase.source).Replace("\", "/")
    $relativeExecutable = (Get-CompatibleRelativePath $repoRoot $executable).Replace("\", "/")
    $results.Add([pscustomobject][ordered]@{
        id = $testCase.id
        description = $testCase.description
        source = $relativeSource
        defines = [string[]]$testCase.defines
        executable = $relativeExecutable
        run_directory = (Get-CompatibleRelativePath $repoRoot $testCase.run_directory).Replace("\", "/")
        compile = $compileResult
        run = $runResult
        status = $status
    })
}

$finishedAt = [DateTimeOffset]::Now
$passedCount = @($results | Where-Object status -eq "passed").Count
$failedCount = $results.Count - $passedCount
$document = [ordered]@{
    schema_version = 1
    started_at = $startedAt.ToString("o")
    finished_at = $finishedAt.ToString("o")
    elapsed_ms = [long]($finishedAt - $startedAt).TotalMilliseconds
    repository_root = $repoRoot
    git_head = $gitHead
    compiler = $compilerPath
    compiler_version = $compilerVersion
    compile_flags = [string[]]@("-O2", "-std=c++20", "-march=native", "-pthread", "-lws2_32")
    total = $results.Count
    passed = $passedCount
    failed = $failedCount
    all_passed = ($failedCount -eq 0)
    results = $results
}

$jsonPath = Join-Path $outputPath "results.json"
$markdownPath = Join-Path $outputPath "report.md"
$document | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = [System.Text.StringBuilder]::new()
[void]$markdown.AppendLine("# MPC係数候補 回帰検査")
[void]$markdown.AppendLine()
[void]$markdown.AppendLine("- 実行開始: $($startedAt.ToString('o'))")
[void]$markdown.AppendLine(('- Git commit: `{0}`' -f $gitHead))
[void]$markdown.AppendLine(('- コンパイラ: `{0}`' -f (ConvertTo-MarkdownCell $compilerVersion)))
[void]$markdown.AppendLine('- コンパイル条件: `-O2 -std=c++20 -march=native -pthread -lws2_32`')
[void]$markdown.AppendLine("- 結果: $passedCount / $($results.Count) 件成功、$failedCount 件失敗")
[void]$markdown.AppendLine()
[void]$markdown.AppendLine("| ID | 検査内容 | マクロ定義 | compile | run | compile時間 (ms) | run時間 (ms) |")
[void]$markdown.AppendLine("|---|---|---|---:|---:|---:|---:|")
foreach ($result in $results) {
    $defines = if ($result.defines.Count -eq 0) { "production既定値" } else { $result.defines -join ", " }
    $compileStatus = if ($result.compile.exit_code -eq 0) { "成功" } else { "失敗 ($($result.compile.exit_code))" }
    $runStatus = if ($null -eq $result.run) { "未実行" } elseif ($result.run.exit_code -eq 0) { "成功" } else { "失敗 ($($result.run.exit_code))" }
    $runDuration = if ($null -eq $result.run) { "-" } else { [string]$result.run.duration_ms }
    [void]$markdown.AppendLine(
        ('| `{0}` | {1} | `{2}` | {3} | {4} | {5} | {6} |' -f `
            $result.id,
            (ConvertTo-MarkdownCell $result.description),
            (ConvertTo-MarkdownCell $defines),
            $compileStatus,
            $runStatus,
            $result.compile.duration_ms,
            $runDuration)
    )
}

$failedResults = @($results | Where-Object status -ne "passed")
if ($failedResults.Count -gt 0) {
    [void]$markdown.AppendLine()
    [void]$markdown.AppendLine("## 失敗内容")
    foreach ($result in $failedResults) {
        [void]$markdown.AppendLine()
        [void]$markdown.AppendLine(('### `{0}`' -f $result.id))
        [void]$markdown.AppendLine()
        [void]$markdown.AppendLine(('状態: `{0}`' -f $result.status))
        [void]$markdown.AppendLine()
        [void]$markdown.AppendLine('```text')
        [void]$markdown.AppendLine("compile stdout:")
        [void]$markdown.AppendLine($result.compile.stdout)
        [void]$markdown.AppendLine("compile stderr:")
        [void]$markdown.AppendLine($result.compile.stderr)
        if ($null -ne $result.run) {
            [void]$markdown.AppendLine("run stdout:")
            [void]$markdown.AppendLine($result.run.stdout)
            [void]$markdown.AppendLine("run stderr:")
            [void]$markdown.AppendLine($result.run.stderr)
        }
        [void]$markdown.AppendLine('```')
    }
}

$markdown.ToString() | Set-Content -LiteralPath $markdownPath -Encoding utf8 -NoNewline

Write-Host "JSON: $jsonPath"
Write-Host "Markdown: $markdownPath"
if ($failedCount -ne 0) {
    exit 1
}
exit 0
