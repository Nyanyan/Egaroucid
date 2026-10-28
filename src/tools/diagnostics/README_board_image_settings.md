# Board image editor settings (Issue #651)

The editor remembers the selected mark, coordinate visibility, color mode, and
outer frame color. It updates shared settings on each selection, writes them to
`setting.json` during normal application shutdown, and restores them on startup.
Cell marks, evaluated values, and arrows are retained in memory for each position
in the current progression, including positions in the inspection branch. They
survive navigation and reopening the editor. Removing a position discards its
annotations; replacing the progression discards all of them. Annotations are
never written to disk and are discarded when the application exits.

## Settings

| JSON key | Values | Default |
| --- | --- | --- |
| `board_image_mark` | 0: black square, 1: black star, 2: white square, 3: white star, 4: evaluation, 5: erase, 6: outlined square, 7: outlined star, 8: black circle, 9: white circle, 10: outlined circle, 11: flip, 12: arrow | 0 |
| `board_image_coordinate` | 0: include, 1: exclude | 0 |
| `board_image_color` | 0: default color, 1: monochrome | 0 |
| `board_image_outer_frame_color` | 0: white, 1: dark gray | 0 |

Missing keys and values with the wrong JSON type retain their defaults. Numeric
indices outside the supported range are clamped. The frame color is retained
while its control is hidden in monochrome mode.

The editor presents the mark shape and its black/white/outline style separately.
The combined IDs above preserve the original six choices in old configuration
files. Flip and outline marks use white on black discs, black on white discs,
and white on empty cells. On a white monochrome board, a thin black halo keeps
white marks on empty cells visible.

To draw an arrow, click its starting cell, then its ending cell. Between clicks,
the arrow preview follows the cursor. Selecting another tool, clearing, saving
an image, or leaving the editor cancels an unfinished arrow. Arrows are black
with white outlines. The erase tool removes the topmost arrow under the cursor
(including its head), without erasing a cell mark underneath on that click.
The clear button removes all cell marks, values, arrows, and pending endpoints
for the displayed position.

## State regression tests

Run these fast tests from the repository root with MinGW `g++` on PATH:

```powershell
New-Item -ItemType Directory -Force build/issue651 | Out-Null
g++ -std=c++20 -Wall -Wextra -pedantic src/tools/diagnostics/test_board_image_edit.cpp -o build/issue651/test_board_image_edit.exe
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
& build/issue651/test_board_image_edit.exe
```

These tests use the production annotation header directly. They check old mark
IDs, all tool/style mappings, arrow endpoint validation, duplicate and reversed
arrows, cancellation, independent position snapshots, clearing, branch
retention, and removal/recreation of a position or the entire progression.

## Manual regression check

Build with the official release script:

```powershell
python src/tools/release_script/build_gui.py -c SIMD_Portable
```

Use a separate portable application directory with its own `appdata` and
`document` folders so that the test does not change personal settings.

1. Start with a `setting.json` that has no `board_image_*` keys. Open the board
   image editor from the output menu and check all four defaults.
2. Select white star, exclude coordinates, dark gray frame, and monochrome.
   Return to the main scene and reopen the editor. Check the restored choices.
3. Switch back to color and check that the dark gray frame is still selected.
   Select monochrome again, then close the application directly from the editor.
   Check that the four saved values are `3, 1, 1, 1` in table order.
4. Restart and reopen the editor. Check white star, excluded coordinates, and
   monochrome. Switch to color to check the retained dark gray frame.
5. Select erase, include coordinates, default color, and white frame. Return to
   the main scene and close the application. Check saved values `5, 0, 0, 0`.
6. In the test configuration, set mark to `999`, coordinate to `-9`, color to
   `"invalid"`, and frame to `7`. Start and close the application normally.
   Check normalized values `12, 0, 0, 1`.

For the additional requirements in [the Issue comment](https://github.com/Nyanyan/Egaroucid/issues/651#issuecomment-6470903584):

1. In the initial position A, place flip marks on black, white, and empty cells.
   Check their contrast colors. Add circle, square, and star outlines, including
   circle outlines over both disc colors.
2. Create an arrow with two cell clicks and check its cursor preview between
   clicks. Select erase and click its shaft over an existing cell mark. Check
   that only the arrow is removed. Add another arrow for the retention check.
3. Check monochrome rendering and save an image. Inspect the saved PNG for all
   marks and the complete arrow.
4. Return to main, play `f5` to reach B, and reopen the editor. Check that B is
   empty. Add a white filled circle on B. Return to A using the graph and reopen
   its editor; check that A's marks and arrow are restored.
5. Clear A and check that every mark and arrow disappears. Add a new mark to A,
   return to B, use Undo to remove B, then play `f5` again. Reopen B's editor and
   check that the earlier circle is gone.
6. Start a new human-versus-human game and open A's editor. Check that the mark
   from the previous progression is gone. Check evaluation mark placement on a
   legal cell and its removal with the clear button.
7. Select outlined circle, place a mark, close, restart, and reopen the editor.
   Check that outlined circle is selected (`board_image_mark = 10`), while the
   placed mark is gone.

## Verification record

On 2026-10-28, the MSVC `SIMD_Portable` build and the state regression tests
succeeded on Windows 11. The original persistence checks were run before adding
the new mark types. The seven additional GUI checks above passed, including
normal shutdown/restart, removing and recreating B, and replacing the progression.
All eight new labels were checked in all four runtime language files and against
their release templates. Build logs, configuration snapshots, and the inspected
PNG were kept locally under `build/issue651/` (ignored by Git).
