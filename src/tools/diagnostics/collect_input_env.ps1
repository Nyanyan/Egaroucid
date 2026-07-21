param(
    [string]$ExePath = "",
    [string]$OutputDirectory = ""
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

$reportPath = Join-Path $OutputDirectory ("input_env_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class InputEnvWin32 {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll", SetLastError=true)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

    [DllImport("user32.dll")]
    public static extern IntPtr GetKeyboardLayout(uint idThread);
}
"@

function Get-ForegroundInputInfo {
    $hwnd = [InputEnvWin32]::GetForegroundWindow()
    if ($hwnd -eq [IntPtr]::Zero) {
        return $null
    }

    $titleBuilder = [System.Text.StringBuilder]::new(512)
    [InputEnvWin32]::GetWindowText($hwnd, $titleBuilder, $titleBuilder.Capacity) | Out-Null

    [uint32]$processId = 0
    $threadId = [InputEnvWin32]::GetWindowThreadProcessId($hwnd, [ref]$processId)
    $hkl = [InputEnvWin32]::GetKeyboardLayout($threadId).ToInt64()
    $langId = $hkl -band 0xffff
    $proc = $null
    try {
        $proc = Get-Process -Id $processId -ErrorAction Stop
    } catch {
        $proc = $null
    }

    [ordered]@{
        hwnd = ("0x{0:X}" -f $hwnd.ToInt64())
        title = $titleBuilder.ToString()
        process_id = $processId
        process_name = if ($proc) { $proc.ProcessName } else { $null }
        thread_id = $threadId
        keyboard_layout_hkl = ("0x{0:X}" -f $hkl)
        lang_id = ("0x{0:X4}" -f $langId)
    }
}

function Get-JsonSummary {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return [ordered]@{
            path = $Path
            exists = $false
        }
    }

    $item = Get-Item -LiteralPath $Path
    $summary = [ordered]@{
        path = $Path
        exists = $true
        length = $item.Length
        last_write_time = $item.LastWriteTime.ToString("o")
    }

    try {
        $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $props = @($json.PSObject.Properties.Name)
        $summary.top_level_keys = $props
        $summary.top_level_key_count = $props.Count
        $summary.preview = [ordered]@{}
        foreach ($name in ($props | Select-Object -First 20)) {
            $value = $json.$name
            if ($null -eq $value) {
                $summary.preview[$name] = $null
            } elseif ($value -is [string] -or $value -is [bool] -or $value -is [int] -or $value -is [double]) {
                $summary.preview[$name] = $value
            } elseif ($value -is [System.Array]) {
                $summary.preview[$name] = @($value)
            } else {
                $summary.preview[$name] = "<object>"
            }
        }
    } catch {
        $summary.parse_error = $_.Exception.Message
    }
    return $summary
}

$os = Get-CimInstance Win32_OperatingSystem
$computerInfo = $null
try {
    $computerInfo = Get-ComputerInfo -Property WindowsProductName,WindowsVersion,OsHardwareAbstractionLayer,OsBuildNumber -ErrorAction Stop
} catch {
    $computerInfo = $null
}

$experiencePack = $null
try {
    $experiencePack = Get-AppxPackage MicrosoftWindows.Client.CBS -ErrorAction Stop |
        Select-Object Name, Version, PackageFullName
} catch {
    $experiencePack = $null
}

$languageList = $null
try {
    $languageList = Get-WinUserLanguageList | ForEach-Object {
        [ordered]@{
            language_tag = $_.LanguageTag
            input_method_tips = @($_.InputMethodTips)
            handwriting = $_.Handwriting
        }
    }
} catch {
    $languageList = $_.Exception.Message
}

$keyboardPreload = @{}
foreach ($subkey in "Preload", "Substitutes") {
    $path = "HKCU:\Keyboard Layout\$subkey"
    if (Test-Path $path) {
        $keyboardPreload[$subkey] = (Get-ItemProperty -Path $path).PSObject.Properties |
            Where-Object { $_.Name -notmatch "^PS" } |
            ForEach-Object { [ordered]@{ name = $_.Name; value = $_.Value } }
    }
}

$processPatterns = @(
    "AutoHotkey", "PowerToys", "Ditto", "ClipClip", "Clipdiary", "ClipboardFusion",
    "Logi", "lghub", "Razer", "Synapse", "Corsair", "iCUE", "SteelSeries", "Kbd"
)
$interestingProcesses = Get-Process | Where-Object {
    $name = $_.ProcessName
    $processPatterns | Where-Object { $name -like "*$_*" }
} | Select-Object ProcessName, Id, Path

$exeItem = if (Test-Path $ExePath) { Get-Item -LiteralPath $ExePath } else { $null }
$exeDirectory = if ($exeItem) { $exeItem.DirectoryName } else { Split-Path -Parent $ExePath }
$portableAppData = Join-Path $exeDirectory "appdata"
$localAppData = Join-Path $env:LOCALAPPDATA "Egaroucid"

$report = [ordered]@{
    collected_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    os = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build_number = $os.BuildNumber
        ubr = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion" -ErrorAction SilentlyContinue).UBR
        product_name = if ($computerInfo) { $computerInfo.WindowsProductName } else { $null }
        windows_version = if ($computerInfo) { $computerInfo.WindowsVersion } else { $null }
        feature_experience_pack = $experiencePack
    }
    culture = [ordered]@{
        culture = (Get-Culture).Name
        ui_culture = (Get-UICulture).Name
        home_location = (Get-WinHomeLocation).HomeLocation
    }
    user_language_list = $languageList
    keyboard_layout_registry = $keyboardPreload
    foreground_input = Get-ForegroundInputInfo
    possible_interfering_processes = @($interestingProcesses)
    exe = [ordered]@{
        path = $ExePath
        exists = [bool]$exeItem
        working_directory = $exeDirectory
        last_write_time = if ($exeItem) { $exeItem.LastWriteTime.ToString("o") } else { $null }
        length = if ($exeItem) { $exeItem.Length } else { $null }
    }
    config_dirs = [ordered]@{
        portable = [ordered]@{
            path = $portableAppData
            exists = Test-Path $portableAppData
            shortcut_key_json = Get-JsonSummary (Join-Path $portableAppData "shortcut_key.json")
            setting_json = Get-JsonSummary (Join-Path $portableAppData "setting.json")
        }
        installed = [ordered]@{
            path = $localAppData
            exists = Test-Path $localAppData
            shortcut_key_json = Get-JsonSummary (Join-Path $localAppData "shortcut_key.json")
            setting_json = Get-JsonSummary (Join-Path $localAppData "setting.json")
        }
    }
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "Input environment report saved to: $reportPath"
