param(
    [string]$ExePath = "",
    [string]$OutputDirectory = "",
    [string]$Label = "clipboard_path",
    [string]$ClipboardText = "f5d6c3d3c4f4f6",
    [int]$StartupDelaySeconds = 5,
    [int]$HoldMilliseconds = 120,
    [switch]$PromptBeforeKeys,
    [switch]$ManualMenuCheck,
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

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$shortcutLogPath = Join-Path $OutputDirectory ("{0}_{1}_shortcut.log" -f $Label, $timestamp)
$clipboardLogPath = Join-Path $OutputDirectory ("{0}_{1}_clipboard.log" -f $Label, $timestamp)

if (-not (Test-Path $ExePath)) {
    throw "Egaroucid.exe not found: $ExePath"
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class ClipboardPathDiagWin32 {
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
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
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

function Set-AppForeground {
    param([System.Diagnostics.Process]$Process, [int]$TimeoutMilliseconds = 5000)
    $handle = Wait-AppWindow -Process $Process
    [ClipboardPathDiagWin32]::ShowWindow($handle, 9) | Out-Null
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        [ClipboardPathDiagWin32]::SetForegroundWindow($handle) | Out-Null
        Start-Sleep -Milliseconds 150
        if ([ClipboardPathDiagWin32]::GetForegroundWindow() -eq $handle) {
            Start-Sleep -Milliseconds 250
            return
        }
    }
    throw "Timed out waiting for Egaroucid to become foreground."
}

function Send-CtrlV {
    param([int]$HoldMs)
    [ClipboardPathDiagWin32]::KeyDown([UInt16]0x11)
    Start-Sleep -Milliseconds 20
    [ClipboardPathDiagWin32]::KeyDown([UInt16]0x56)
    Start-Sleep -Milliseconds $HoldMs
    [ClipboardPathDiagWin32]::KeyUp([UInt16]0x56)
    Start-Sleep -Milliseconds 20
    [ClipboardPathDiagWin32]::KeyUp([UInt16]0x11)
    Start-Sleep -Milliseconds 1000
}

Set-Clipboard -Value $ClipboardText
Write-Host "Clipboard text set. Length: $($ClipboardText.Length)"

$psi = [System.Diagnostics.ProcessStartInfo]::new($ExePath)
$psi.WorkingDirectory = Split-Path -Parent $ExePath
$psi.UseShellExecute = $false
$psi.Environment["EGAROUCID_SHORTCUT_DIAG_LOG"] = $shortcutLogPath
$psi.Environment["EGAROUCID_CLIPBOARD_DIAG_LOG"] = $clipboardLogPath

Write-Host "Starting Egaroucid: $ExePath"
Write-Host "Shortcut diagnostic log: $shortcutLogPath"
Write-Host "Clipboard diagnostic log: $clipboardLogPath"
$proc = [System.Diagnostics.Process]::Start($psi)

try {
    Start-Sleep -Seconds $StartupDelaySeconds
    $proc.Refresh()
    if ($proc.HasExited) {
        throw "Egaroucid exited before Ctrl+V could be tested."
    }

    if ($PromptBeforeKeys) {
        Read-Host "Switch IME/layout to the target state, then press Enter"
    }

    Set-AppForeground -Process $proc
    Start-Sleep -Seconds 1

    Write-Host "Sending Ctrl+V"
    Send-CtrlV -HoldMs $HoldMilliseconds

    if ($ManualMenuCheck) {
        Write-Host ""
        Write-Host "Manual step needed: use the Egaroucid menu In/Out -> Input from Clipboard now."
        Write-Host "This distinguishes menu path success from Ctrl+V shortcut failure."
        Read-Host "Press Enter after the manual menu check"
        Start-Sleep -Milliseconds 500
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

Write-Host ""
if (Test-Path $shortcutLogPath) {
    Write-Host "Shortcut log tail:"
    Get-Content -LiteralPath $shortcutLogPath | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "No shortcut log was created."
}

if (Test-Path $clipboardLogPath) {
    Write-Host ""
    Write-Host "Clipboard log:"
    Get-Content -LiteralPath $clipboardLogPath | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "No clipboard log was created."
}

Write-Host ""
Write-Host "Shortcut log saved to: $shortcutLogPath"
Write-Host "Clipboard log saved to: $clipboardLogPath"
