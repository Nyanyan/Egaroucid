# Shortcut and Input Diagnostics

This directory separates two suspected problem families:

- Legacy input sampling loss after `v6.5.2`, especially `v6.5.2` versus `v7.0.0`.
- Shortcut manager / IME / clipboard instability from `v7.5.0+`, especially current `main`.

All scripts write logs under `src/tools/diagnostics/logs` by default. Scripts that launch Egaroucid use environment variables so normal runs are unchanged.

## Current main / 7.8.2

Collect environment only:

```powershell
.\src\tools\diagnostics\collect_input_env.ps1
```

Run shortcut VK matrix and key hold sweep:

```powershell
.\src\tools\diagnostics\run_shortcut_vk_diag.ps1 -PromptBeforeKeys
```

Run once for each IME state:

- ENG
- Chinese IME
- Japanese IME Direct Input / A
- Japanese IME Hiragana / あ
- Japanese IME full-width alphanumeric
- after Convert / NonConvert / Kana toggles

The shortcut log includes:

- `raw`
- `win`
- `input_source`
- `hook_backend`
- `hook_backend_reason`
- `hook_installed`
- `hook_error`
- `hook_subclass_error`
- `hook_install_thread_id`
- `hook_hwnd_thread_id`
- `hook_same_thread`
- `hook_same_process`
- `hook_current_wndproc_matches`
- `message_state_synchronized`
- `observed_keyboard_message_count`
- `focused`
- `pressed_vks`
- `down_vks`
- `message_down_vks`
- `expected_vks`
- `extra_non_modifier_vks`
- `missing_expected_vks`
- `modifier_mismatch`
- `no_down`
- `legacy_string_matcher_used`
- `legacy_string_matcher_reason`
- `candidate`
- `down`
- `pressed`

Interpretation:

- `input_source=message` means Win32 queued key messages are feeding shortcut edge detection.
- `input_source=polling_fallback` means the WndProc observer was unavailable and the legacy polling fallback was used.
- `hook_backend=set_window_subclass` means the observer was installed with `SetWindowSubclass`.
- `hook_backend=guarded_raw_wndproc` means the observer is still WM-message based, but uses the guarded raw WndProc path.
- `hook_backend=polling_fallback` appears only when shortcut input is not coming from the WM observer.
- `hook_backend_reason=cross_thread` is expected when the shortcut/check thread differs from the HWND owner thread.
- `hook_backend_reason=subclass_failed`, `raw_chain_unsafe`, or `raw_install_failed` points at a hook installation or chain safety problem.
- `hook_install_thread_id`, `hook_hwnd_thread_id`, and `hook_same_thread` show whether `SetWindowSubclass` is applicable.
- `hook_same_process=1` is expected for the Siv3D window.
- `hook_current_wndproc_matches=1` means the active observer is still present. For `set_window_subclass` this checks the subclass observer; for `guarded_raw_wndproc` it checks the raw WndProc.
- `message_state_synchronized=1` means the per-frame shortcut snapshot was copied from the WM observer state.
- `observed_keyboard_message_count` should increase when keyboard messages reach the observer.
- `legacy_string_matcher_used=1` means the old Siv3D string matcher handled an unsupported custom shortcut name; normal WM-backed not-matched results must not use it.
- In older logs, `extra_non_modifier_vks=Convert/Kana/NonConvert` with `win=Ctrl+V` means IME VK pollution and exact matching is too strict.
- `pressed_vks=Ctrl+V` with `no_down=1` means the combination was held but the expected down edge was missed.
- `missing_expected_vks=Ctrl` points at generic versus left/right Ctrl recognition.
- No log movement usually means focus or input never reached Egaroucid.

Run clipboard path diagnosis:

```powershell
.\src\tools\diagnostics\run_clipboard_path_diag.ps1 -PromptBeforeKeys
```

For the menu comparison, keep the app open and perform the menu action manually:

```powershell
.\src\tools\diagnostics\run_clipboard_path_diag.ps1 -PromptBeforeKeys -ManualMenuCheck -KeepAppOpen
```

Interpretation:

- Menu succeeds but Ctrl+V fails: shortcut recognition problem.
- Menu also fails: clipboard read or import parsing problem.
- Shortcut log has `down=input_from_clipboard` but clipboard log has `failed=1`: not a shortcut problem.

Run focus recovery diagnosis:

```powershell
.\src\tools\diagnostics\run_shortcut_focus_recovery_diag.ps1 -PromptBeforeKeys
```

This starts Egaroucid once and uses a manual real `Alt+Tab` phase: after Egaroucid starts, press Enter in PowerShell, use the countdown to switch to a non-Egaroucid window with real `Alt+Tab`, then the script sends the shortcut set to the current foreground window. After that, return to Egaroucid and press Enter in PowerShell so the script can run the post-recovery matrix.

Interpretation:

- Any `unexpected_egaroucid_downs_during_external_phase` means Egaroucid received shortcut down events while it was not foreground.
- Matrix failures after the external-window phase point at focus recovery state, stale modifiers, or first-frame recovery issues.
- `input_source=message` and `hook_installed=1` should still be present after focus returns.
- `hook_current_wndproc_matches=0` means the active observer is no longer present or the raw WndProc chain became unsafe.
- A stalled `observed_keyboard_message_count` while `raw` changes means Siv3D sees keyboard state but the shortcut WndProc is not receiving keyboard messages.

Run system-level key sampling in parallel when needed:

```powershell
.\src\tools\diagnostics\run_key_latency_probe.ps1 -DurationSeconds 30 -IntervalMilliseconds 10
```

Use this to compare Windows `GetAsyncKeyState` against Egaroucid's internal shortcut log.

## Legacy v6.5.2 / v7.0.0

Create and patch worktrees only:

```powershell
.\src\tools\diagnostics\run_legacy_input_diag.ps1 -CreateOnly
```

This creates sibling worktrees like:

- `..\Egaroucid_diag_v6_5_2`
- `..\Egaroucid_diag_v7_0_0`

The script injects a minimal frame logger into `src/gui/main_scene.hpp` in each worktree. It records:

- frame timestamp
- frame delta
- focus state
- `KeyN`, `KeyV`, `KeyBackspace`, `KeyLeft`, `KeyRight` down/pressed
- analyzing/searching state
- `use_umigame_value`
- frame-over-50/100/200ms markers

After building each worktree, rerun:

```powershell
.\src\tools\diagnostics\run_legacy_input_diag.ps1 -PromptBeforeKeys
```

If you know the build command, pass it:

```powershell
.\src\tools\diagnostics\run_legacy_input_diag.ps1 -BuildCommand "msbuild Egaroucid.sln /p:Configuration=Release /p:Platform=x64"
```

Legacy interpretation:

- `v7.0.0` short presses miss clearly more than `v6.5.2`: supports frame delay or `.down()` sampling-window loss.
- Misses only under analysis/search/high `use_umigame_value`: points at main-thread or search load.
- Misses in first frames after refocus: points at focus/input-state recovery.

## Required Manual Steps

The scripts can send keys, but they cannot reliably switch your IME into every target state. When a script is run with `-PromptBeforeKeys`, switch the IME/layout manually, then press Enter in the PowerShell prompt.

For the clipboard menu-vs-shortcut distinction, run `run_clipboard_path_diag.ps1` with `-ManualMenuCheck` and manually click `In/Out -> Input from Clipboard` when prompted.
