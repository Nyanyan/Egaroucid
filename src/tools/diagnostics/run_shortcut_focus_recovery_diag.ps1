param(
    [string]$ExePath = "",
    [string]$OutputDirectory = "",
    [string]$Label = "shortcut_focus_recovery",
    [int]$StartupDelaySeconds = 5,
    [int[]]$HoldMilliseconds = @(30, 60, 100, 200, 500),
    [int]$ExternalHoldMilliseconds = 120,
    [int]$ManualAltTabDelaySeconds = 5,
    [switch]$PromptBeforeKeys,
    [switch]$KeepAppOpen
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Join-Path $repoRoot "bin\Egaroucid.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot "logs"
}
if (-not (Test-Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}
if (-not (Test-Path $ExePath)) {
    throw "Egaroucid.exe not found: $ExePath"
}
if ($ManualAltTabDelaySeconds -lt 1) {
    throw "ManualAltTabDelaySeconds must be at least 1."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $OutputDirectory ("{0}_{1}.log" -f $Label, $timestamp)
$summaryPath = Join-Path $OutputDirectory ("{0}_{1}_summary.csv" -f $Label, $timestamp)
$externalPath = Join-Path $OutputDirectory ("{0}_{1}_external.txt" -f $Label, $timestamp)

Add-Type @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class ShortcutFocusRecoveryDiagWin32 {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT {
        public uint type;
        public INPUTUNION U;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct INPUTUNION {
        [FieldOffset(0)]
        public MOUSEINPUT mi;
        [FieldOffset(0)]
        public KEYBDINPUT ki;
        [FieldOffset(0)]
        public HARDWAREINPUT hi;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSEINPUT {
        public int dx;
        public int dy;
        public uint mouseData;
        public uint dwFlags;
        public uint time;
        public UIntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT {
        public ushort wVk;
        public ushort wScan;
        public uint dwFlags;
        public uint time;
        public UIntPtr dwExtraInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct HARDWAREINPUT {
        public uint uMsg;
        public ushort wParamL;
        public ushort wParamH;
    }

    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

    public const uint INPUT_KEYBOARD = 1;
    public const uint KEYEVENTF_EXTENDEDKEY = 0x0001;
    public const uint KEYEVENTF_KEYUP = 0x0002;

    public static void KeyDown(ushort vk) { SendKey(vk, 0); }
    public static void KeyUp(ushort vk) { SendKey(vk, KEYEVENTF_KEYUP); }

    private static void SendKey(ushort vk, uint flags) {
        if (IsExtendedKey(vk)) {
            flags |= KEYEVENTF_EXTENDEDKEY;
        }
        INPUT[] inputs = new INPUT[1];
        inputs[0].type = INPUT_KEYBOARD;
        inputs[0].U.ki.wVk = vk;
        inputs[0].U.ki.wScan = 0;
        inputs[0].U.ki.dwFlags = flags;
        inputs[0].U.ki.time = 0;
        inputs[0].U.ki.dwExtraInfo = UIntPtr.Zero;
        uint sent = SendInput(1, inputs, Marshal.SizeOf(typeof(INPUT)));
        if (sent != 1) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
    }

    private static bool IsExtendedKey(ushort vk) {
        return vk == 0x21 || vk == 0x22 || vk == 0x23 || vk == 0x24 ||
            vk == 0x25 || vk == 0x26 || vk == 0x27 || vk == 0x28 ||
            vk == 0x2D || vk == 0x2E || vk == 0x6F || vk == 0xA3 ||
            vk == 0xA5;
    }
}

"@

function Wait-AppWindow {
    param([System.Diagnostics.Process]$Process, [int]$TimeoutMilliseconds = 10000)
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "Egaroucid exited before a main window was created."
        }
        if ($Process.MainWindowHandle -ne [IntPtr]::Zero) {
            return $Process.MainWindowHandle
        }
        Start-Sleep -Milliseconds 100
    }
    throw "Timed out waiting for Egaroucid main window."
}

function Set-ForegroundHandle {
    param([IntPtr]$Handle, [string]$Name, [int]$TimeoutMilliseconds = 5000)
    [ShortcutFocusRecoveryDiagWin32]::ShowWindow($Handle, 9) | Out-Null
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        [ShortcutFocusRecoveryDiagWin32]::KeyDown([UInt16]0x12)
        Start-Sleep -Milliseconds 10
        [ShortcutFocusRecoveryDiagWin32]::KeyUp([UInt16]0x12)
        Start-Sleep -Milliseconds 50
        [ShortcutFocusRecoveryDiagWin32]::SetForegroundWindow($Handle) | Out-Null
        Start-Sleep -Milliseconds 150
        if ([ShortcutFocusRecoveryDiagWin32]::GetForegroundWindow() -eq $Handle) {
            Start-Sleep -Milliseconds 250
            return
        }
    }
    Write-Warning "Timed out waiting for $Name to become foreground. Continuing with current foreground window."
}

function Set-AppForeground {
    param([System.Diagnostics.Process]$Process, [int]$TimeoutMilliseconds = 5000)
    $handle = Wait-AppWindow -Process $Process
    Set-ForegroundHandle -Handle $handle -Name "Egaroucid" -TimeoutMilliseconds $TimeoutMilliseconds
}

function Get-LogLineCount {
    if (Test-Path $logPath) {
        return @(Get-Content -LiteralPath $logPath).Count
    }
    return 0
}

function Get-NewLogLines {
    param([int]$BeforeCount)
    if (-not (Test-Path $logPath)) {
        return @()
    }
    $allLines = @(Get-Content -LiteralPath $logPath)
    if ($allLines.Count -le $BeforeCount) {
        return @()
    }
    return $allLines[$BeforeCount..($allLines.Count - 1)]
}

function Send-ShortcutChord {
    param([UInt16[]]$Vks, [int]$HoldMs)
    foreach ($vk in $Vks) {
        [ShortcutFocusRecoveryDiagWin32]::KeyDown($vk)
        Start-Sleep -Milliseconds 20
    }
    Start-Sleep -Milliseconds $HoldMs
    for ($i = $Vks.Count - 1; $i -ge 0; --$i) {
        [ShortcutFocusRecoveryDiagWin32]::KeyUp($Vks[$i])
        Start-Sleep -Milliseconds 20
    }
    Start-Sleep -Milliseconds 350
}

$vk = @{
    Ctrl = [UInt16]0x11
    Shift = [UInt16]0x10
    V = [UInt16]0x56
    C = [UInt16]0x43
    N = [UInt16]0x4E
    S = [UInt16]0x53
    A = [UInt16]0x41
    Left = [UInt16]0x25
    Right = [UInt16]0x27
    Home = [UInt16]0x24
}

$tests = @(
    @{ Name = "A"; Vks = @($vk.A); Expected = "analyze" },
    @{ Name = "V"; Vks = @($vk.V); Expected = "show_disc_hint" },
    @{ Name = "Left"; Vks = @($vk.Left); Expected = "backward" },
    @{ Name = "Right"; Vks = @($vk.Right); Expected = "forward" },
    @{ Name = "Ctrl+V"; Vks = @($vk.Ctrl, $vk.V); Expected = "input_from_clipboard" },
    @{ Name = "Ctrl+C"; Vks = @($vk.Ctrl, $vk.C); Expected = "output_transcript" },
    @{ Name = "Ctrl+N"; Vks = @($vk.Ctrl, $vk.N); Expected = "new_game" },
    @{ Name = "Ctrl+S"; Vks = @($vk.Ctrl, $vk.S); Expected = "screen_shot" },
    @{ Name = "Shift+Home"; Vks = @($vk.Shift, $vk.Home); Expected = "go_to_random_generated_position" }
)

if (Test-Path $logPath) {
    Remove-Item -LiteralPath $logPath
}

$psi = [System.Diagnostics.ProcessStartInfo]::new($ExePath)
$psi.WorkingDirectory = Split-Path -Parent $ExePath
$psi.UseShellExecute = $false
$psi.Environment["EGAROUCID_SHORTCUT_DIAG_LOG"] = $logPath
$psi.Environment["EGAROUCID_SHORTCUT_DIAG_ONLY"] = "1"

Write-Host "Starting Egaroucid: $ExePath"
Write-Host "Shortcut diagnostic log: $logPath"
Write-Host "External phase log: $externalPath"
$proc = [System.Diagnostics.Process]::Start($psi)
$results = @()

try {
    Start-Sleep -Seconds $StartupDelaySeconds
    $proc.Refresh()
    if ($proc.HasExited) {
        throw "Egaroucid exited before shortcuts could be tested."
    }

    if ($PromptBeforeKeys) {
        Read-Host "Switch IME/layout to the target state, then press Enter"
    }

    Set-AppForeground -Process $proc
    Start-Sleep -Milliseconds 500

    $externalBefore = Get-LogLineCount
    $externalEvents = @()
    Write-Host ""
    Write-Host "Manual focus recovery phase:"
    Write-Host "1. Press Enter here to arm the external-window shortcut send."
    Write-Host ("2. You will have {0} seconds to use real Alt+Tab to switch to a non-Egaroucid window." -f $ManualAltTabDelaySeconds)
    Write-Host "3. The script will send the same shortcut set to whatever window is foreground."
    Write-Host "4. After the send finishes, use real Alt+Tab to return to Egaroucid."
    Read-Host "Press Enter, then immediately Alt+Tab to the external target window"
    for ($remaining = $ManualAltTabDelaySeconds; $remaining -gt 0; --$remaining) {
        Write-Host ("Sending external shortcuts in {0}..." -f $remaining)
        Start-Sleep -Seconds 1
    }
    foreach ($test in $tests) {
        $externalEvents += ("mode=manual_alt_tab_send`tkey={0}`thold_ms={1}" -f $test.Name, $ExternalHoldMilliseconds)
        Write-Host ("Sending {0} to current foreground external window" -f $test.Name)
        Send-ShortcutChord -Vks ([UInt16[]]$test.Vks) -HoldMs $ExternalHoldMilliseconds
    }
    Write-Host ""
    Write-Host "External shortcut send complete."
    Write-Host "Use real Alt+Tab to return to Egaroucid now if you want to reproduce the user focus path."
    Read-Host "When ready, press Enter here; the script will focus Egaroucid and run the post-recovery matrix"
    $externalEvents += "mode=manual_alt_tab"
    $externalAfterLines = @(Get-NewLogLines -BeforeCount $externalBefore)
    $unexpectedExternalDowns = @($externalAfterLines | Where-Object { $_ -match "`tdown=(?!undefined(`t|$))([^\t]+)" })

    $externalEvents += ("egaroucid_log_lines_during_external_phase={0}" -f $externalAfterLines.Count)
    $externalEvents += ("unexpected_egaroucid_downs_during_external_phase={0}" -f $unexpectedExternalDowns.Count)
    if ($unexpectedExternalDowns.Count -gt 0) {
        $externalEvents += "unexpected_down_lines:"
        $externalEvents += $unexpectedExternalDowns
    }
    $externalEvents | Set-Content -LiteralPath $externalPath -Encoding UTF8

    Start-Sleep -Seconds 1

    foreach ($hold in $HoldMilliseconds) {
        foreach ($test in $tests) {
            Write-Host ("Sending {0}, hold={1}ms after focus recovery" -f $test.Name, $hold)
            Set-AppForeground -Process $proc -TimeoutMilliseconds 3000
            $before = Get-LogLineCount
            Send-ShortcutChord -Vks ([UInt16[]]$test.Vks) -HoldMs $hold
            $newLines = @(Get-NewLogLines -BeforeCount $before)
            $matched = $newLines | Where-Object { $_ -match ("down=" + [regex]::Escape($test.Expected) + "(\t|$)") } | Select-Object -First 1
            $last = $newLines | Select-Object -Last 1
            $results += [pscustomobject]@{
                Phase = "after_external_focus"
                Key = $test.Name
                HoldMs = $hold
                Expected = $test.Expected
                Passed = [bool]$matched
                Evidence = if ($matched) { $matched } else { $last }
            }
        }
    }

    if ($KeepAppOpen) {
        Read-Host "Diagnostics complete. Press Enter to close Egaroucid"
    }
} finally {
    if ($proc -and -not $proc.HasExited) {
        $proc.CloseMainWindow() | Out-Null
        if (-not $proc.WaitForExit(3000)) {
            $proc.Kill()
            $proc.WaitForExit()
        }
    }
}

$results | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Summary:"
foreach ($row in $results) {
    $status = if ($row.Passed) { "PASS" } else { "FAIL" }
    Write-Host ("{0} {1} hold={2}ms expected={3}" -f $status, $row.Key, $row.HoldMs, $row.Expected)
    if (-not $row.Passed -and $row.Evidence) {
        Write-Host ("  Evidence: {0}" -f $row.Evidence)
    }
}
Write-Host ""
Write-Host "Shortcut log saved to: $logPath"
Write-Host "Summary CSV saved to: $summaryPath"
Write-Host "External phase log saved to: $externalPath"

if (($results | Where-Object { -not $_.Passed } | Measure-Object).Count -gt 0) {
    exit 1
}
