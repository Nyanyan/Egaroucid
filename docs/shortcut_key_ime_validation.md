# Windows Shortcut Key IME Validation

## Scope

This PR updates Windows shortcut matching so Egaroucid does not depend only on
Siv3D `Input::name()` for shortcut identity. On Windows, shortcut matching now
normalizes keyboard input through virtual-key codes, checks exact key
combinations from one keyboard-state snapshot, and keeps the existing Siv3D
string fallback for unmapped or non-Windows cases.

The fix is intended to cover both default shortcuts and user-customized
shortcuts. Custom shortcuts that can be mapped to Windows virtual-key codes use
the same IME/layout-stable path as the defaults. Unknown custom names still use
the original Siv3D fallback.

## Modified Files

- `src/gui/function/shortcut_key.hpp`
  - Adds Windows virtual-key name normalization for shortcut capture and
    matching.
  - Handles common alphanumeric keys, function keys, numpad keys, navigation
    keys, OEM punctuation keys, Windows keys, IME keys, media/browser keys, and
    `0xNN` virtual-key names.
  - Enforces exact non-modifier matching by scanning the current Windows
    keyboard snapshot instead of checking only a short hard-coded key list.
  - Enforces stricter Ctrl/Shift/Alt family matching, including left/right
    modifier cases.
  - Treats `down` as true only when the exact shortcut is active and an
    expected key became newly pressed.
  - Adds opt-in diagnostic logging controlled by environment variables:
    `EGAROUCID_SHORTCUT_DIAG_LOG` and `EGAROUCID_SHORTCUT_DIAG_ONLY`.
- `tools/shortcut_diag/run_shortcut_diag.ps1`
  - Runs Egaroucid with the diagnostic environment variables enabled.
  - Supports `Default`, `NegativeExtra`, and `CustomVkPositive` scenarios.
  - Writes validation logs to `review_packages` by default so they can be
    attached to review comments without being committed.
- `docs/shortcut_key_ime_validation.md`
  - This validation record.

Local build-only changes in `src/gui/function/const/gui_common.hpp` are not part
of this PR validation scope.

## Validation Environment

- OS: Windows
- Build used for testing: `bin/Egaroucid.exe`
- Build command used locally:
  - `MSBuild.exe Egaroucid.sln /p:Configuration=Release /p:Platform=x64 /m`
- Static check:
  - `git diff --check`
- Available input methods on this machine:
  - Microsoft ENG input method
  - Microsoft Chinese Pinyin input method

Because this machine only has Microsoft ENG and Microsoft Chinese Pinyin
available, additional validation should be performed by other developers with
other IMEs and keyboard layouts, especially Japanese IME and any layout reported
by users in the original issue.

## Latest Test Results

All listed tests passed with the current shortcut-key patch.

| Scenario | Input method state | Result | Attachment log |
| --- | --- | --- | --- |
| Default shortcuts | ENG startup | 24/24 | `review_packages/shortcut_diag_custom_vk_patch_smoke_20260612_194934.log` |
| Extra-key negative cases | ENG startup | 4/4 | `review_packages/shortcut_diag_negative_extra_keys_20260612_195044.log` |
| Extra-key negative cases, committed script self-check | Current active input method | 4/4 | `review_packages/shortcut_diag_script_selfcheck_negative_20260612_200725.log` |
| Custom shortcut VK cases | ENG startup | 2/2 | `review_packages/shortcut_diag_custom_vk_positive_20260612_195216.log` |
| Default shortcuts | Chinese Pinyin startup | 24/24 | `review_packages/shortcut_diag_custom_vk_patch_chinese_all_20260612_195425.log` |
| Default shortcuts | ENG startup, then switched to Chinese Pinyin before testing | 24/24 | `review_packages/shortcut_diag_custom_vk_patch_eng_start_then_chinese_all_20260612_195600.log` |

Earlier contaminated manual runs, including the run where Alt was accidentally
held during testing, are intentionally excluded from this validation record.

## Covered Shortcut Cases

The `Default` scenario validates all currently assigned default shortcuts:

- `Space` -> `start_game`
- `Ctrl+N` -> `new_game`
- `A` -> `analyze`
- `B` -> `ai_put_black`
- `W` -> `ai_put_white`
- `V` -> `show_disc_hint`
- `U` -> `show_umigame_value`
- `D` -> `show_graph_value`
- `S` -> `show_graph_sum_of_loss`
- `P` -> `show_laser_pointer`
- `G` -> `put_1_move_by_ai`
- `Right` -> `forward`
- `Left` -> `backward`
- `Backspace` -> `undo`
- `Home` -> `go_to_first_position`
- `End` -> `go_to_last_position`
- `Shift+Home` -> `go_to_random_generated_position`
- `Ctrl+L` -> `save_this_branch`
- `Ctrl+R` -> `generate_random_board`
- `Q` -> `stop_calculating`
- `Ctrl+V` -> `input_from_clipboard`
- `Ctrl+E` -> `edit_board`
- `Ctrl+C` -> `output_transcript`
- `Ctrl+S` -> `screen_shot`

The `NegativeExtra` scenario validates that incomplete exact matching does not
trigger shortcuts:

- `Ctrl+Shift+N` does not trigger `new_game`
- `A+F1` does not trigger `analyze`
- `A+Num0` does not trigger `analyze`
- Releasing `F1` while `A` remains held does not fire `analyze` as a new
  `down` event

The `CustomVkPositive` scenario temporarily writes a local
`shortcut_key.json`, restores it afterward, and validates custom bindings:

- `F1` -> `analyze`
- `Num0` -> `ai_put_black`

## Reproduction Commands

Build Egaroucid first, then run from the repository root:

```powershell
.\tools\shortcut_diag\run_shortcut_diag.ps1 -Scenario Default -Label eng_default
.\tools\shortcut_diag\run_shortcut_diag.ps1 -Scenario Default -Label chinese_default -PromptBeforeKeys
.\tools\shortcut_diag\run_shortcut_diag.ps1 -Scenario Default -Label eng_then_chinese -PromptBeforeKeys
.\tools\shortcut_diag\run_shortcut_diag.ps1 -Scenario NegativeExtra -Label negative_extra
.\tools\shortcut_diag\run_shortcut_diag.ps1 -Scenario CustomVkPositive -Label custom_vk
```

Use `-PromptBeforeKeys` when the tester needs to switch the active input method
after Egaroucid starts and before automated key input begins.

## Remaining Validation Needed

Other developers should repeat the default, negative, and custom shortcut
scenarios on input methods and keyboard layouts not available on this machine.
In particular:

- Japanese IME
- Non-US keyboard layouts
- Any IME/layout combination reported by users in the original issue
- Additional customized shortcut names if a user report includes a specific
  unsupported key
