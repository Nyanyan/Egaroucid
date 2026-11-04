# Board image editor settings (Issue #651)

The editor remembers the selected mark, coordinate visibility, color mode,
outer frame color, and file format. It updates shared settings on each selection, writes them to
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
| `board_image_format` | 0: PNG, 1: SVG, 2: EPS | 0 |

Missing keys and values with the wrong JSON type retain their defaults. Numeric
indices outside the supported range are clamped. The frame color is retained
while its control is hidden in monochrome mode.

The editor presents a 3-by-3 symbol palette: square, star, and circle rows, each
with black, white, and outline variants. Four more tiles show flip, arrow, and
eraser symbols and a localized evaluation label (not an example numeric value).
Selected tiles have a white border; unselected tiles have a thin dark gray
border. Neither selection nor hovering fills their backgrounds.
There is no separate mark-style text selector.
The palette and board share the same mark renderer. The IDs above preserve
the original six choices in old configuration files.
Flip marks use solid black arrows on empty cells and white discs, and solid white
arrows on black discs, in both color modes (Issue #662). Outline marks use white
on black discs, black on white discs, and white on empty cells. On a white
monochrome board, a thin black halo keeps white shape marks on empty cells visible.

To draw an arrow, click its starting cell, then its ending cell. Between clicks,
the preview endpoint snaps to the center of the hovered cell; moving within the
same cell leaves it fixed. The preview is hidden outside the board. Clicking
uses the same cell calculation as the preview. Selecting another tool, clearing, saving
an image, or leaving the editor cancels an unfinished arrow. Arrows are black
with white outlines. The erase tool removes the topmost arrow under the cursor
(including its head), without erasing a cell mark underneath on that click.
The bottom action row contains Back, Clear all, and Save. Only Save has a white
fill; Back and Clear all have matching unfilled white borders. Clear all removes
all cell marks, values, arrows, and pending endpoints for the displayed position.
Image options stay on aligned rows, with the frame row disabled in monochrome
mode so that the layout does not shift.

## Vector export (Issue #548)

The "File format" row selects PNG, SVG, or EPS. PNG keeps the screen capture:
the clipped frame is saved and copied to the clipboard, and its resolution
follows the window scale. SVG and EPS are drawn again as vector shapes instead
of captured pixels, so they contain no bitmap and do not depend on the window
scale. They are saved to the same folder with the same `date_transcript` name
and the `.svg` or `.eps` extension.

Every save also copies the picture to the clipboard. PNG saves copy the
captured image as before (`PNG` and `CF_DIBV5`). SVG and EPS saves copy the
same captured image, and on Windows they also place the vector picture first,
as UTF-8 SVG with a terminating null in the registered `image/svg+xml` format.
Applications that accept SVG from the clipboard can paste it as a vector image;
other applications paste the captured image. EPS has no common clipboard
format, so EPS saves also use SVG for the vector copy. Other platforms copy only
the captured image, because Siv3D's clipboard holds text or bitmaps.

The vector image covers the same area as the PNG (447 x 447 px with
coordinates, 420 x 420 px without them, where 1 px is 1 pt in EPS). Outside the
round outer frame it is transparent when coordinates are excluded, as in the
PNG. Discs, dots, arcs, and frame corners are true circles and arcs. Star and
arrow frames follow Siv3D's `Shape2D::drawFrame`, including the flat cut of
sharp corners. Coordinates and evaluation values are filled outlines of the
glyphs that the editor draws (`Font::renderOutlines` of the same bold font), placed
like `DrawableText::drawAt`, so no font is needed to view or print the files.
SVG layers are named `background`, `coordinates`, `grid`, `discs`, `frame`,
`marks`, and `arrows`. EPS files are plain ASCII PostScript that uses only path
and painting operators (no images or fonts).

Implementation:

- `src/gui/function/vector_image.hpp`: a small vector image model and its SVG
  and EPS writers. Standard C++ only; no Siv3D or other libraries.
- `src/gui/function/board_image_vector.hpp`: mark colors, sizes, and flip, star,
  and arrow geometry shared by the editor and the export, and the builder for
  the whole picture. Standard C++ only; text arrives as glyph outlines.
- `src/gui/screen_shot.hpp`: Siv3D glyph outlines, GUI colors and geometry,
  file output (`save_board_image_vector`), and the save with the clipboard copy
  (`take_board_image_vector`, which shares `capture_board_image` with PNG).

The editor and the export call the same `board_image_mark_style()`, so each
mark uses the same colors, halo, and backing in both. When changing the drawing
in `Board_image` or `draw_board()`, change `build_board_image_vector()` too.

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
retention, and removal/recreation of a position or the entire progression. The
grid tests check movement within a cell, exact cell boundaries, and board edges.

The vector export has its own test of the production headers. With an output
folder, it also writes sample SVG and EPS files for viewers:

```powershell
New-Item -ItemType Directory -Force build/issue548 | Out-Null
g++ -std=c++20 -Wall -Wextra -pedantic src/tools/diagnostics/test_board_image_vector.cpp -o build/issue548/test_board_image_vector.exe
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
& build/issue548/test_board_image_vector.exe build/issue548
```

It checks number formatting, SVG and EPS syntax (XML escaping, groups, circles,
arcs, Bezier quarters, ASCII-only EPS), that `board_image_mark_style()` gives
the same result as the previous decision code for shape and value marks, the
solid flip colors for every disc and color mode, the Siv3D star, arrow, and frame
geometry, the `drawAt` text box, and all
coordinate/color/frame combinations of a sample board. The `board-image-vector`
case of `src/tools/operation_check/gui.py` checks the same export inside the GUI
with the real fonts and files.

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

For the vector export (Issue #548):

1. Place every mark type, an evaluation value, and two arrows. Save as PNG,
   then select SVG and save, then EPS and save. Check the three files in the
   screenshot folder and the "board image saved to ... and copied to clipboard"
   log line.
2. Open the SVG in a browser and in an editor such as Inkscape, and the EPS in a
   PostScript viewer or `gs -dEPSCrop`. Compare them with the PNG of the same
   settings, then zoom in to check that edges and text stay sharp.
3. Repeat with coordinates excluded (transparent outside the frame), with
   monochrome, and with the dark gray frame. Check a Chinese UI, whose bold font
   is synthesized.
4. Select EPS, close, restart, and reopen the editor. Check that EPS is still
   selected (`board_image_format = 2`).
5. After an SVG save, use "Read clipboard image" of `desktop_fixture.py`.
   Check that `fixture-image.json` has `svg` (with `xml: true`), `png`, and
   `dibv5`, and that the `svg` hash equals the SHA-256 of the saved `.svg`
   file. After an EPS save, check that the same three entries exist.
6. Paste after SVG and EPS saves into an application that accepts SVG from the
   clipboard (for example, Inkscape or a recent Word) and into one that does
   not (for example, Paint). Check that the first pastes the vector picture and
   the second pastes the captured image.

## Verification record

On 2026-10-28, the MSVC `SIMD_Portable` build and the state regression tests
succeeded on Windows 11. The original persistence checks were run before adding
the new mark types. The seven additional GUI checks above passed, including
normal shutdown/restart, removing and recreating B, and replacing the progression.
All eight new labels were checked in all four runtime language files and against
their release templates. Build logs, configuration snapshots, and the inspected
PNG were kept locally under `build/issue651/` (ignored by Git).

The palette/layout follow-up was checked at Japanese 0.8 scale and English 1.0
scale. The actual flip icon, direct selection of white square and outlined
circle tiles, placement, image saving, clearing, and the Back button were
verified. For arrow snapping, the cursor was moved to two different corners of
F6, then to G6, then outside the board. The preview stayed at the F6 center,
switched to the G6 center, and disappeared outside. Clicking away from the G6
center produced the same endpoint as the preview.

On 2026-10-29, the neutral border styling and shortened Save label passed the
MSVC `SIMD_Portable` build and GUI checks at Japanese 0.8 and English 1.0 scale.
The localized evaluation label fit its tile, selected the value tool, and placed
a numeric value on the board. Save, Clear all, and the outlined Back button
worked. Save/value translations in all four runtime language files matched
their release templates (Value / 評価値 / 评估值 / 評估值).
