param(
    [string[]]$Tags = @("v6.5.2", "v7.0.0"),
    [string]$WorktreeParent = "",
    [string]$OutputDirectory = "",
    [int[]]$HoldMilliseconds = @(30, 60, 100, 200, 500),
    [int]$StartupDelaySeconds = 5,
    [switch]$CreateOnly,
    [switch]$PromptBeforeKeys,
    [string]$BuildCommand = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($WorktreeParent)) {
    $WorktreeParent = (Resolve-Path (Join-Path $repoRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot "logs"
}
if (-not (Test-Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class LegacyInputDiagWin32 {
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

function Ensure-Worktree {
    param([string]$Tag)
    $name = "Egaroucid_diag_" + ($Tag -replace "[^A-Za-z0-9_]", "_")
    $path = Join-Path $WorktreeParent $name
    if (-not (Test-Path $path)) {
        git -C $repoRoot worktree add $path $Tag | Out-Null
    }
    return (Resolve-Path $path).Path
}

function Add-LegacyPatch {
    param([string]$WorktreePath)

    $mainScene = Join-Path $WorktreePath "src\gui\main_scene.hpp"
    if (-not (Test-Path $mainScene)) {
        throw "main_scene.hpp not found in $WorktreePath"
    }

    $text = Get-Content -LiteralPath $mainScene -Raw
    if ($text -match "legacy_input_diagnostic_log_frame") {
        return
    }

    if ($text -notmatch "#include <cstdlib>") {
        $text = $text -replace "#include <algorithm>\r?\n", "#include <algorithm>`r`n#include <cstdlib>`r`n#include <fstream>`r`n#include <sstream>`r`n"
    }

    $helper = @'

inline std::string legacy_input_diagnostic_log_path() {
#if SIV3D_PLATFORM(WINDOWS)
    char* env_path = nullptr;
    size_t env_path_size = 0;
    if (_dupenv_s(&env_path, &env_path_size, "EGAROUCID_LEGACY_INPUT_DIAG_LOG") != 0 || env_path == nullptr || env_path[0] == '\0') {
        if (env_path) {
            free(env_path);
        }
        return "";
    }
    const std::string path = env_path;
    free(env_path);
    return path;
#else
    const char* env_path = std::getenv("EGAROUCID_LEGACY_INPUT_DIAG_LOG");
    return env_path == nullptr ? "" : std::string(env_path);
#endif
}

inline void legacy_input_diagnostic_log_frame(const bool analyzing, const bool searching, const bool use_umigame_value) {
    const std::string path = legacy_input_diagnostic_log_path();
    if (path.empty()) {
        return;
    }

    static uint64_t previous_ms = 0;
    const uint64_t now_ms = tim();
    const uint64_t delta_ms = previous_ms == 0 ? 0 : now_ms - previous_ms;
    previous_ms = now_ms;

    std::ofstream fout(path, std::ios::app);
    if (!fout) {
        return;
    }

    fout << "ts_ms=" << now_ms
        << "\tdelta_ms=" << delta_ms
        << "\tfocused=" << (Window::GetState().focused ? "1" : "0")
        << "\tKeyN_down=" << (KeyN.down() ? "1" : "0")
        << "\tKeyN_pressed=" << (KeyN.pressed() ? "1" : "0")
        << "\tKeyV_down=" << (KeyV.down() ? "1" : "0")
        << "\tKeyV_pressed=" << (KeyV.pressed() ? "1" : "0")
        << "\tKeyBackspace_down=" << (KeyBackspace.down() ? "1" : "0")
        << "\tKeyBackspace_pressed=" << (KeyBackspace.pressed() ? "1" : "0")
        << "\tKeyLeft_down=" << (KeyLeft.down() ? "1" : "0")
        << "\tKeyLeft_pressed=" << (KeyLeft.pressed() ? "1" : "0")
        << "\tKeyRight_down=" << (KeyRight.down() ? "1" : "0")
        << "\tKeyRight_pressed=" << (KeyRight.pressed() ? "1" : "0")
        << "\tanalyzing=" << (analyzing ? "1" : "0")
        << "\tsearching=" << (searching ? "1" : "0")
        << "\tuse_umigame_value=" << (use_umigame_value ? "1" : "0")
        << "\tframe_over_50ms=" << (delta_ms > 50 ? "1" : "0")
        << "\tframe_over_100ms=" << (delta_ms > 100 ? "1" : "0")
        << "\tframe_over_200ms=" << (delta_ms > 200 ? "1" : "0")
        << '\n';
}
'@

    $classIndex = $text.IndexOf("class Main_scene")
    if ($classIndex -lt 0) {
        throw "Could not find class Main_scene in $mainScene"
    }
    $text = $text.Insert($classIndex, $helper + "`r`n")

    $updatePattern = "void update\(\) override \{\r?\n"
    if ($text -notmatch $updatePattern) {
        throw "Could not find Main_scene::update in $mainScene"
    }
    $text = [regex]::Replace(
        $text,
        $updatePattern,
        "void update() override {`r`n        legacy_input_diagnostic_log_frame(ai_status.analyzing, global_searching, getData().menu_elements.use_umigame_value);`r`n",
        1
    )

    Set-Content -LiteralPath $mainScene -Value $text -Encoding UTF8
}

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
    [LegacyInputDiagWin32]::ShowWindow($handle, 9) | Out-Null
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        [LegacyInputDiagWin32]::SetForegroundWindow($handle) | Out-Null
        Start-Sleep -Milliseconds 150
        if ([LegacyInputDiagWin32]::GetForegroundWindow() -eq $handle) {
            Start-Sleep -Milliseconds 250
            return
        }
    }
    throw "Timed out waiting for Egaroucid to become foreground."
}

function Send-KeyPress {
    param([UInt16]$Vk, [int]$HoldMs)
    [LegacyInputDiagWin32]::KeyDown($Vk)
    Start-Sleep -Milliseconds $HoldMs
    [LegacyInputDiagWin32]::KeyUp($Vk)
    Start-Sleep -Milliseconds 350
}

foreach ($tag in $Tags) {
    Write-Host "Preparing $tag"
    $worktree = Ensure-Worktree -Tag $tag
    Add-LegacyPatch -WorktreePath $worktree
    Write-Host "Worktree patched: $worktree"

    if (-not [string]::IsNullOrWhiteSpace($BuildCommand)) {
        Write-Host "Running build command in $worktree"
        Push-Location $worktree
        try {
            Invoke-Expression $BuildCommand
        } finally {
            Pop-Location
        }
    }

    if ($CreateOnly) {
        continue
    }

    $exePath = Join-Path $worktree "bin\Egaroucid.exe"
    if (-not (Test-Path $exePath)) {
        Write-Warning "No exe found for $tag at $exePath. Build this worktree, then rerun without -CreateOnly."
        continue
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logPath = Join-Path $OutputDirectory ("legacy_input_{0}_{1}.log" -f ($tag -replace "[^A-Za-z0-9_]", "_"), $timestamp)
    $summaryPath = Join-Path $OutputDirectory ("legacy_input_{0}_{1}_summary.csv" -f ($tag -replace "[^A-Za-z0-9_]", "_"), $timestamp)

    $psi = [System.Diagnostics.ProcessStartInfo]::new($exePath)
    $psi.WorkingDirectory = Split-Path -Parent $exePath
    $psi.UseShellExecute = $false
    $psi.Environment["EGAROUCID_LEGACY_INPUT_DIAG_LOG"] = $logPath

    Write-Host "Starting ${tag}: $exePath"
    Write-Host "Legacy diagnostic log: $logPath"
    $proc = [System.Diagnostics.Process]::Start($psi)
    $rows = @()
    try {
        Start-Sleep -Seconds $StartupDelaySeconds
        $proc.Refresh()
        if ($proc.HasExited) {
            throw "$tag exited before keys could be tested."
        }
        if ($PromptBeforeKeys) {
            Read-Host "Set the target IME/layout/load state for $tag, then press Enter"
        }
        Set-AppForeground -Process $proc

        $keys = @(
            @{ Name = "N"; Vk = [UInt16]0x4E; DownField = "KeyN_down=1" },
            @{ Name = "V"; Vk = [UInt16]0x56; DownField = "KeyV_down=1" },
            @{ Name = "Backspace"; Vk = [UInt16]0x08; DownField = "KeyBackspace_down=1" },
            @{ Name = "Left"; Vk = [UInt16]0x25; DownField = "KeyLeft_down=1" },
            @{ Name = "Right"; Vk = [UInt16]0x27; DownField = "KeyRight_down=1" }
        )

        foreach ($hold in $HoldMilliseconds) {
            foreach ($key in $keys) {
                $beforeCount = if (Test-Path $logPath) { @(Get-Content -LiteralPath $logPath).Count } else { 0 }
                Write-Host ("Sending {0}, hold={1}ms" -f $key.Name, $hold)
                Send-KeyPress -Vk $key.Vk -HoldMs $hold
                $lines = if (Test-Path $logPath) { @(Get-Content -LiteralPath $logPath) } else { @() }
                $newLines = if ($lines.Count -gt $beforeCount) { $lines[$beforeCount..($lines.Count - 1)] } else { @() }
                $matched = $newLines | Where-Object { $_ -like ("*" + $key.DownField + "*") } | Select-Object -First 1
                $slow = @($newLines | Where-Object { $_ -match "frame_over_(50|100|200)ms=1" }).Count
                $rows += [pscustomobject]@{
                    Tag = $tag
                    Key = $key.Name
                    HoldMs = $hold
                    DownSeen = [bool]$matched
                    SlowFrameLines = $slow
                    Evidence = $matched
                }
            }
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

    $rows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
    Write-Host "Legacy log saved to: $logPath"
    Write-Host "Summary CSV saved to: $summaryPath"
}
