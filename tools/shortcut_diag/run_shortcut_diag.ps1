param(
    [ValidateSet("Default", "NegativeExtra", "CustomVkPositive")]
    [string]$Scenario = "Default",
    [string]$Label = "",
    [int]$StartupDelaySeconds = 5,
    [switch]$PromptBeforeKeys,
    [string]$LogDirectory = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$exePath = Join-Path $repoRoot "bin\Egaroucid.exe"
if ([string]::IsNullOrWhiteSpace($LogDirectory)) {
    $LogDirectory = Join-Path $repoRoot "review_packages"
}
if ([string]::IsNullOrWhiteSpace($Label)) {
    $Label = $Scenario.ToLowerInvariant()
}
$logPath = Join-Path $LogDirectory ("shortcut_diag_{0}_{1}.log" -f $Label, (Get-Date -Format "yyyyMMdd_HHmmss"))

if (-not (Test-Path $exePath)) {
    throw "Egaroucid.exe not found: $exePath"
}
if (-not (Test-Path $LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory | Out-Null
}
if (Test-Path $logPath) {
    Remove-Item -LiteralPath $logPath
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class Win32ShortcutDiag {
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
    public const uint KEYEVENTF_KEYUP = 0x0002;

    public static void KeyDown(ushort vk) {
        SendKey(vk, 0);
    }

    public static void KeyUp(ushort vk) {
        SendKey(vk, KEYEVENTF_KEYUP);
    }

    private static void SendKey(ushort vk, uint flags) {
        INPUT[] inputs = new INPUT[1];
        inputs[0].type = INPUT_KEYBOARD;
        inputs[0].U.ki.wVk = vk;
        inputs[0].U.ki.wScan = 0;
        inputs[0].U.ki.dwFlags = flags;
        inputs[0].U.ki.time = 0;
        inputs[0].U.ki.dwExtraInfo = UIntPtr.Zero;
        uint sent = SendInput(1, inputs, Marshal.SizeOf(typeof(INPUT)));
        if (sent != 1) {
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        }
    }
}
"@

function Wait-EgaroucidWindow {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutMilliseconds = 10000
    )

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

function Set-EgaroucidForeground {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutMilliseconds = 5000
    )

    $handle = Wait-EgaroucidWindow -Process $Process
    [Win32ShortcutDiag]::ShowWindow($handle, 9) | Out-Null

    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        [Win32ShortcutDiag]::SetForegroundWindow($handle) | Out-Null
        Start-Sleep -Milliseconds 150
        if ([Win32ShortcutDiag]::GetForegroundWindow() -eq $handle) {
            Start-Sleep -Milliseconds 250
            return
        }
    }

    throw "Timed out waiting for Egaroucid to become the foreground window."
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
    param([UInt16[]]$Vks)

    foreach ($vk in $Vks) {
        [Win32ShortcutDiag]::KeyDown($vk)
        Start-Sleep -Milliseconds 30
    }
    Start-Sleep -Milliseconds 120
    for ($i = $Vks.Count - 1; $i -ge 0; --$i) {
        [Win32ShortcutDiag]::KeyUp($Vks[$i])
        Start-Sleep -Milliseconds 30
    }
    Start-Sleep -Milliseconds 350
}

function Invoke-PositiveShortcutTest {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Name,
        [UInt16[]]$Vks,
        [string]$Expected
    )

    Write-Host ("Sending {0}" -f $Name)
    Set-EgaroucidForeground -Process $Process -TimeoutMilliseconds 3000
    $beforeCount = Get-LogLineCount
    Send-ShortcutChord -Vks $Vks
    $afterLines = @(Get-NewLogLines -BeforeCount $beforeCount)
    $matched = $afterLines | Where-Object { $_ -match ("down=" + [regex]::Escape($Expected) + "(\t|$)") } | Select-Object -First 1

    return [pscustomobject]@{
        Name = $Name
        Expected = $Expected
        Passed = [bool]$matched
        Evidence = $matched
    }
}

function Invoke-NegativeShortcutTest {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Name,
        [scriptblock]$Send,
        [string]$Forbidden
    )

    Write-Host ("Sending negative case {0}" -f $Name)
    Set-EgaroucidForeground -Process $Process -TimeoutMilliseconds 3000
    $beforeCount = Get-LogLineCount
    & $Send
    $afterLines = @(Get-NewLogLines -BeforeCount $beforeCount)
    $matched = $afterLines | Where-Object { $_ -match ("down=" + [regex]::Escape($Forbidden) + "(\t|$)") } | Select-Object -First 1

    return [pscustomobject]@{
        Name = $Name
        Expected = "not $Forbidden"
        Passed = -not [bool]$matched
        Evidence = $matched
    }
}

function Write-CustomShortcutConfig {
    $appDataDir = Join-Path $env:LOCALAPPDATA "Egaroucid"
    $shortcutConfigPath = Join-Path $appDataDir "shortcut_key.json"
    if (-not (Test-Path $appDataDir)) {
        New-Item -ItemType Directory -Path $appDataDir | Out-Null
    }

    $backupPath = $null
    if (Test-Path $shortcutConfigPath) {
        $backupPath = Join-Path $appDataDir ("shortcut_key.json.shortcut_diag_backup_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
        Copy-Item -LiteralPath $shortcutConfigPath -Destination $backupPath
    }

    $json = [ordered]@{
        analyze = @("F1")
        ai_put_black = @("Num0")
    } | ConvertTo-Json
    Set-Content -LiteralPath $shortcutConfigPath -Value $json -Encoding UTF8

    return [pscustomobject]@{
        Path = $shortcutConfigPath
        BackupPath = $backupPath
    }
}

function Restore-CustomShortcutConfig {
    param([object]$ConfigState)

    if ($null -eq $ConfigState) {
        return
    }
    if ($ConfigState.BackupPath -and (Test-Path $ConfigState.BackupPath)) {
        Move-Item -LiteralPath $ConfigState.BackupPath -Destination $ConfigState.Path -Force
    } elseif (Test-Path $ConfigState.Path) {
        Remove-Item -LiteralPath $ConfigState.Path
    }
}

$vk = @{
    Backspace = [UInt16]0x08
    Tab = [UInt16]0x09
    Enter = [UInt16]0x0D
    Shift = [UInt16]0x10
    Ctrl = [UInt16]0x11
    Alt = [UInt16]0x12
    Space = [UInt16]0x20
    End = [UInt16]0x23
    Home = [UInt16]0x24
    Left = [UInt16]0x25
    Right = [UInt16]0x27
    Numpad0 = [UInt16]0x60
    F1 = [UInt16]0x70
}
foreach ($code in [char[]]"ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
    $vk[[string]$code] = [UInt16][byte][char]$code
}

$customConfigState = $null
if ($Scenario -eq "CustomVkPositive") {
    $customConfigState = Write-CustomShortcutConfig
    Write-Host ("Wrote temporary shortcut config: {0}" -f $customConfigState.Path)
}

$psi = [System.Diagnostics.ProcessStartInfo]::new($exePath)
$psi.WorkingDirectory = Split-Path -Parent $exePath
$psi.UseShellExecute = $false
$psi.Environment["EGAROUCID_SHORTCUT_DIAG_LOG"] = $logPath
$psi.Environment["EGAROUCID_SHORTCUT_DIAG_ONLY"] = "1"

Write-Host "Starting Egaroucid: $exePath"
Write-Host "Scenario: $Scenario"
Write-Host "Diagnostic log: $logPath"
$proc = [System.Diagnostics.Process]::Start($psi)

try {
    Start-Sleep -Seconds $StartupDelaySeconds
    $proc.Refresh()
    if ($proc.HasExited) {
        throw "Egaroucid exited before shortcuts could be tested."
    }

    if ($PromptBeforeKeys) {
        Read-Host "Switch the input method to the desired state now, then press Enter"
    }

    Set-EgaroucidForeground -Process $proc
    Start-Sleep -Seconds 2

    $results = @()

    if ($Scenario -eq "Default") {
        $tests = @(
            @{ Name = "Space";      Vks = @($vk.Space);             Expected = "start_game" },
            @{ Name = "Ctrl+N";     Vks = @($vk.Ctrl, $vk.N);       Expected = "new_game" },
            @{ Name = "A";          Vks = @($vk.A);                 Expected = "analyze" },
            @{ Name = "B";          Vks = @($vk.B);                 Expected = "ai_put_black" },
            @{ Name = "W";          Vks = @($vk.W);                 Expected = "ai_put_white" },
            @{ Name = "V";          Vks = @($vk.V);                 Expected = "show_disc_hint" },
            @{ Name = "U";          Vks = @($vk.U);                 Expected = "show_umigame_value" },
            @{ Name = "D";          Vks = @($vk.D);                 Expected = "show_graph_value" },
            @{ Name = "S";          Vks = @($vk.S);                 Expected = "show_graph_sum_of_loss" },
            @{ Name = "P";          Vks = @($vk.P);                 Expected = "show_laser_pointer" },
            @{ Name = "G";          Vks = @($vk.G);                 Expected = "put_1_move_by_ai" },
            @{ Name = "Right";      Vks = @($vk.Right);             Expected = "forward" },
            @{ Name = "Left";       Vks = @($vk.Left);              Expected = "backward" },
            @{ Name = "Backspace";  Vks = @($vk.Backspace);         Expected = "undo" },
            @{ Name = "Home";       Vks = @($vk.Home);              Expected = "go_to_first_position" },
            @{ Name = "End";        Vks = @($vk.End);               Expected = "go_to_last_position" },
            @{ Name = "Shift+Home"; Vks = @($vk.Shift, $vk.Home);   Expected = "go_to_random_generated_position" },
            @{ Name = "Ctrl+L";     Vks = @($vk.Ctrl, $vk.L);       Expected = "save_this_branch" },
            @{ Name = "Ctrl+R";     Vks = @($vk.Ctrl, $vk.R);       Expected = "generate_random_board" },
            @{ Name = "Q";          Vks = @($vk.Q);                 Expected = "stop_calculating" },
            @{ Name = "Ctrl+V";     Vks = @($vk.Ctrl, $vk.V);       Expected = "input_from_clipboard" },
            @{ Name = "Ctrl+E";     Vks = @($vk.Ctrl, $vk.E);       Expected = "edit_board" },
            @{ Name = "Ctrl+C";     Vks = @($vk.Ctrl, $vk.C);       Expected = "output_transcript" },
            @{ Name = "Ctrl+S";     Vks = @($vk.Ctrl, $vk.S);       Expected = "screen_shot" }
        )
        foreach ($test in $tests) {
            $results += Invoke-PositiveShortcutTest -Process $proc -Name $test.Name -Vks ([UInt16[]]$test.Vks) -Expected $test.Expected
        }
    } elseif ($Scenario -eq "NegativeExtra") {
        $results += Invoke-NegativeShortcutTest -Process $proc -Name "Ctrl+Shift+N" -Forbidden "new_game" -Send {
            Send-ShortcutChord -Vks ([UInt16[]]@($vk.Ctrl, $vk.Shift, $vk.N))
        }
        $results += Invoke-NegativeShortcutTest -Process $proc -Name "F1 then A" -Forbidden "analyze" -Send {
            Send-ShortcutChord -Vks ([UInt16[]]@($vk.F1, $vk.A))
        }
        $results += Invoke-NegativeShortcutTest -Process $proc -Name "Num0 then A" -Forbidden "analyze" -Send {
            Send-ShortcutChord -Vks ([UInt16[]]@($vk.Numpad0, $vk.A))
        }
        $results += Invoke-NegativeShortcutTest -Process $proc -Name "release F1 while holding A" -Forbidden "analyze" -Send {
            [Win32ShortcutDiag]::KeyDown($vk.F1)
            Start-Sleep -Milliseconds 50
            [Win32ShortcutDiag]::KeyDown($vk.A)
            Start-Sleep -Milliseconds 200
            [Win32ShortcutDiag]::KeyUp($vk.F1)
            Start-Sleep -Milliseconds 300
            [Win32ShortcutDiag]::KeyUp($vk.A)
            Start-Sleep -Milliseconds 350
        }
    } elseif ($Scenario -eq "CustomVkPositive") {
        $tests = @(
            @{ Name = "F1";   Vks = @($vk.F1);      Expected = "analyze" },
            @{ Name = "Num0"; Vks = @($vk.Numpad0); Expected = "ai_put_black" }
        )
        foreach ($test in $tests) {
            $results += Invoke-PositiveShortcutTest -Process $proc -Name $test.Name -Vks ([UInt16[]]$test.Vks) -Expected $test.Expected
        }
    }

    Start-Sleep -Seconds 1
}
finally {
    if ($proc -and -not $proc.HasExited) {
        $proc.CloseMainWindow() | Out-Null
        if (-not $proc.WaitForExit(3000)) {
            $proc.Kill()
            $proc.WaitForExit()
        }
    }
    Restore-CustomShortcutConfig -ConfigState $customConfigState
}

if (-not (Test-Path $logPath)) {
    throw "No diagnostic log was created."
}

$lines = Get-Content -LiteralPath $logPath
Write-Host ""
Write-Host "Diagnostic log tail:"
$lines | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "Summary:"
foreach ($result in $results) {
    if ($result.Passed) {
        Write-Host ("PASS {0} -> {1}" -f $result.Name, $result.Expected)
    } else {
        Write-Host ("FAIL {0} -> {1}" -f $result.Name, $result.Expected)
        if ($result.Evidence) {
            Write-Host ("  Evidence: {0}" -f $result.Evidence)
        }
    }
}

$failed = @($results | Where-Object { -not $_.Passed })
Write-Host ""
Write-Host ("Passed {0}/{1}" -f ($results.Count - $failed.Count), $results.Count)
Write-Host "Log saved to: $logPath"
if ($failed.Count -gt 0) {
    exit 1
}
