param(
    [int]$DurationSeconds = 30,
    [int]$IntervalMilliseconds = 10,
    [string]$OutputDirectory = "",
    [string]$Label = "key_latency"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot "logs"
}
if (-not (Test-Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}

$logPath = Join-Path $OutputDirectory ("{0}_{1}.tsv" -f $Label, (Get-Date -Format "yyyyMMdd_HHmmss"))

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class KeyLatencyProbeWin32 {
    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(int vKey);

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll", SetLastError=true)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}
"@

function Get-ForegroundInfo {
    $hwnd = [KeyLatencyProbeWin32]::GetForegroundWindow()
    if ($hwnd -eq [IntPtr]::Zero) {
        return [pscustomobject]@{ Hwnd = ""; ProcessId = ""; ProcessName = ""; Title = "" }
    }

    [uint32]$processId = 0
    [KeyLatencyProbeWin32]::GetWindowThreadProcessId($hwnd, [ref]$processId) | Out-Null

    $titleBuilder = [System.Text.StringBuilder]::new(256)
    [KeyLatencyProbeWin32]::GetWindowText($hwnd, $titleBuilder, $titleBuilder.Capacity) | Out-Null

    $name = ""
    try {
        $name = (Get-Process -Id $processId -ErrorAction Stop).ProcessName
    } catch {
        $name = ""
    }

    [pscustomobject]@{
        Hwnd = ("0x{0:X}" -f $hwnd.ToInt64())
        ProcessId = $processId
        ProcessName = $name
        Title = ($titleBuilder.ToString() -replace "`t|`r|`n", " ")
    }
}

$keys = @(
    @{ Name = "Ctrl"; Vk = 0x11 },
    @{ Name = "LCtrl"; Vk = 0xA2 },
    @{ Name = "RCtrl"; Vk = 0xA3 },
    @{ Name = "V"; Vk = 0x56 },
    @{ Name = "C"; Vk = 0x43 },
    @{ Name = "N"; Vk = 0x4E },
    @{ Name = "S"; Vk = 0x53 },
    @{ Name = "Left"; Vk = 0x25 },
    @{ Name = "Right"; Vk = 0x27 },
    @{ Name = "Home"; Vk = 0x24 },
    @{ Name = "Shift"; Vk = 0x10 },
    @{ Name = "LShift"; Vk = 0xA0 },
    @{ Name = "RShift"; Vk = 0xA1 },
    @{ Name = "VK_KANA"; Vk = 0x15 },
    @{ Name = "VK_CONVERT"; Vk = 0x1C },
    @{ Name = "VK_NONCONVERT"; Vk = 0x1D },
    @{ Name = "VK_IME_ON"; Vk = 0x16 },
    @{ Name = "VK_IME_OFF"; Vk = 0x1A }
)

$writer = [System.IO.StreamWriter]::new($logPath, $false, [System.Text.UTF8Encoding]::new($false))
try {
    $writer.WriteLine("timestamp`telapsed_ms`tforeground_process`tforeground_pid`tforeground_hwnd`tforeground_title`tpressed_keys`traw_states")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $endMs = $DurationSeconds * 1000
    Write-Host "Sampling GetAsyncKeyState every $IntervalMilliseconds ms for $DurationSeconds seconds."
    Write-Host "Log: $logPath"

    while ($sw.ElapsedMilliseconds -lt $endMs) {
        $fg = Get-ForegroundInfo
        $pressed = New-Object System.Collections.Generic.List[string]
        $raw = New-Object System.Collections.Generic.List[string]
        foreach ($key in $keys) {
            $state = [KeyLatencyProbeWin32]::GetAsyncKeyState([int]$key.Vk)
            if (($state -band 0x8000) -ne 0) {
                $pressed.Add([string]$key.Name)
            }
            if ($state -ne 0) {
                $raw.Add(("{0}=0x{1:X4}" -f $key.Name, ($state -band 0xffff)))
            }
        }

        $writer.WriteLine(("{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}`t{7}" -f
            (Get-Date).ToString("o"),
            $sw.ElapsedMilliseconds,
            $fg.ProcessName,
            $fg.ProcessId,
            $fg.Hwnd,
            $fg.Title,
            ($pressed -join "+"),
            ($raw -join ";")))
        $writer.Flush()
        Start-Sleep -Milliseconds $IntervalMilliseconds
    }
} finally {
    $writer.Dispose()
}

Write-Host "Key latency probe saved to: $logPath"
