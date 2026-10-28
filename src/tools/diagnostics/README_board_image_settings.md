# Board image editor settings (Issue #651)

The editor remembers the selected mark, coordinate visibility, color mode, and
outer frame color. It updates shared settings on each selection, writes them to
`setting.json` during normal application shutdown, and restores them on startup.
Marks placed on individual board cells are not saved.

## Settings

| JSON key | Values | Default |
| --- | --- | --- |
| `board_image_mark` | 0: black square, 1: black star, 2: white square, 3: white star, 4: evaluation, 5: erase | 0 |
| `board_image_coordinate` | 0: include, 1: exclude | 0 |
| `board_image_color` | 0: default color, 1: monochrome | 0 |
| `board_image_outer_frame_color` | 0: white, 1: dark gray | 0 |

Missing keys and values with the wrong JSON type retain their defaults. Numeric
indices outside the supported range are clamped. The frame color is retained
while its control is hidden in monochrome mode.

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
   Check normalized values `5, 0, 0, 1`.

## Verification record

On 2026-10-28, the MSVC `SIMD_Portable` build succeeded on Windows 11. Checks 1–6
passed through the actual GUI; the saved JSON snapshots also matched the
expected values. Build logs and settings snapshots were kept locally under
`build/issue651/` (ignored by Git).
