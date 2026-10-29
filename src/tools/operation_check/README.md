# Operation checks (Issue #652)

Reusable checks for release candidates and feature development. The scripts use
Python 3.9+ and its standard library. Run them from any working directory; the
examples below assume the repository root. Build the application first. These
scripts test the supplied executable, not whichever source revision is checked
out, and do not build or download anything automatically.

## Run

```powershell
# Console: an existing build with its resources/ directory
python src/tools/operation_check/console.py --exe bin/Egaroucid_for_Console.exe

# GUI: build with the official Visual Studio script, then check the Portable build
python src/tools/release_script/build_gui.py -c SIMD_Portable
python src/tools/operation_check/gui.py --exe bin/Egaroucid.exe

# List cases without starting the application
python src/tools/operation_check/console.py --list
python src/tools/operation_check/gui.py --list

# Run selected cases when changing a feature (repeat --case as needed)
python src/tools/operation_check/console.py --case threads --case ponder
python src/tools/operation_check/gui.py --case settings --case annotations

# Also check that the packaged translations match the release templates
python src/tools/operation_check/gui.py --case resources --templates src/tools/release_script/format_files/0_common_files/languages
```

Use `--resources PATH` if the distribution's resource directory is elsewhere.
Use `--output-dir PATH` to choose where evidence is kept, `--timeout SECONDS`
to change the per-process timeout (default 30 seconds), and `--fail-fast` to stop
after the first failing case. The full GUI suite takes about a minute on a
development PC and opens/closes the test application several times. Run lengthy
checks in a visible console; the runner prints the current case, completed/total
count, elapsed time, and a provisional ETA. Ctrl+C interrupts the run and stops
the current child application. GUI checks need an interactive desktop and a
working graphics driver. They do not inject mouse/keyboard input.

Exit codes: **0** all selected cases passed; **1** at least one case failed;
**2** invalid arguments or a missing executable/resource directory. Misspelled
case names are errors. Failure checks remain active under `python -O`.

Each run creates a new timestamped directory under
`build/operation_check/{console,gui}/`. `results.json` contains the executable,
resource path, each case's result/duration/artifact directory, and whether the
selected run completed. Cases continue after failures unless `--fail-fast` is
set. Input, command arguments, stdout, stderr, and exception details are kept in
each case directory. Timeouts and missing output count as failures even if the
application would otherwise return exit code zero. Interrupted runs retain the
last completed results; `complete: false` means cases remain unfinished.

## Console coverage

| Case | Checks |
| --- | --- |
| `metadata` | Version, CLI help and interactive help |
| `argument-error` | Missing required CLI argument fails |
| `perft` | 1,396 leaves at depth 5 in both pass-counting modes |
| `history` | Mixed-case moves, transcript, undo, redo, branching and reset |
| `illegal-moves` | Invalid coordinates, occupied/illegal cells and partial transcript rollback |
| `search` | Quiet search from two positions, all four initial hints |
| `endgame` | One-empty exact move/score (+64), setboard/new, explicit pass |
| `threads`, `ponder` | Thread changes, invalid counts, search and shutdown with/without pondering |
| `gtp` | Numbered/framed replies, identity, known_command, legal moves, turn, play, undo, clear, genmove and illegal move rejection |
| `solve-file` | Exact endgame read from a problem file |
| `selfplay` | Complete game replayed by an independent array-based rules implementation |
| `book` | Release book imports positions and supplies an initial move |

Tests compare legal moves and exact endgame scores rather than unstable heuristic
scores or random tie-breaking. They use one thread and hash level 18 by default.
The engine's two specific messages about generating a missing hash18 seed are
allowed; raw logs retain them. Other engine errors fail normal cases. The
invalid-input cases separately verify their expected errors.

## GUI coverage

The GUI binary includes an opt-in `--operation-check CASE_DIRECTORY` mode. Only
`gui.py` should invoke it: a request marker and a unique token identify the
isolated run. With no flag, the application follows its usual startup and
operation. Installed (non-Portable) GUI builds reject this mode **before loading
personal settings**; build SIMD_Portable, Generic_Portable, or AVX512_Portable.
The Python runner checks for the protocol marker in the executable and refuses
older binaries before launching them, since those might ignore the flag and load
personal settings. Use an unpacked build produced by the release build script.

The probe waits for normal loading and main-scene creation, calls the production
save/import/profile/cache/search functions, and closes through the normal Close
scene. Native results must have the current request token and expected case.
Normal shutdown must also produce shortcut settings. Native checks/logs are in
`native-result.json`, `native-XX.json`, and `native.log` inside the case directory.

| Case | Checks |
| --- | --- |
| `resources` | Required release files, four language JSONs and editor labels; optional template comparison |
| `startup` | Resources/fonts, initial board/legal moves, nonblank frame, normal shutdown |
| `settings-defaults` | Old settings lacking board-image options |
| `settings-invalid` | Wrong types and out-of-range board-image options |
| `settings` | Save/reload editor/display options through production persistence; restart from those saved files |
| `transcript` | Transcript detection, illegal transcript rejection, board text import/export; independent rules comparison |
| `profiles` | AI/display profile save/load and multilingual names |
| `game-files` | Game JSON/CSV save/load, history, multilingual metadata and multiline memo |
| `annotations` | Per-position storage/pruning, progression reset, arrows/duplicates, clear, grid mapping |
| `forced-openings` | Legal continuations from a registered opening |
| `search` | Legal AI result using the GUI-initialized engine |
| `scenes-english`, `scenes-japanese`, `scenes-simplified-chinese`, `scenes-traditional-chinese-taiwan` | Nine scene transitions/draws each: main, text import, board editor, game information, export, board image, opening settings, shortcut settings, game library |

Scene cases require a nonuniform frame of at least 600x400 pixels and save each
as `frame-SCENE.png`. Inspect these artifacts for clipping, overlapping text and
styling. This is a rendering smoke check, not a pixel-perfect visual comparison.
The scripts do **not** verify button hit targets, physical keyboard/IME behavior,
drag-and-drop, clipboard integration, native file dialogs, network services,
installers or the entire set of book learning/repair operations. Keep interactive
release checks for those, including the IME/focus and image-editor procedures in
`../diagnostics/`. Passing this suite is not a claim of exhaustive GUI coverage.

## Isolation and extending the suite

Every run stages the chosen executable, sibling DLLs, LICENSE (when present),
and release resources. It never copies the original `appdata/` or `document/`.
GUI cases get their own settings, profiles, game directories and an empty book;
automatic update checks are disabled for that invocation. The Windows GUI reads
resources relative to its working directory, so cases link the disposable
resource snapshot locally (copying if hard links are unavailable). All generated
files stay under the output directory. Console book tests use the staged release
book. Original settings and personal books are never written. Evidence is kept
after completion; remove old run directories when no longer needed.

Add a function accepting `Context`, then register `Case(name, description,
function)` in `console.py` or `gui.py`. Stable case names make targeted regression
runs easy. Use `require(condition, message)` for explicit invariants, and
`ctx.run(...)` for subprocesses with logging and timeouts. For a new native GUI
check, add its branch to `Gui_probe::check_case()` and a Python case; rebuild the
GUI. Prefer checking the public production path and saved results instead of
duplicating the function's implementation in the test.

The runner and independent oracle have fast self-tests (no GUI SDK needed):

```powershell
python -m unittest discover -s src/tools/operation_check -p "test_*.py" -v
```

These verify failure/continuation, fail-fast/incomplete reports, invalid names and
timeouts, retained timeout logs, basic rules-oracle positions, and rejection of
stale/failed/incompatible GUI responses and binaries without the isolated mode.

## Verification record

On 2026-10-29 on Windows 11, the console executable passed all 13 cases (also
under `python -O`). The GUI `SIMD_Portable` build succeeded with the official
Visual Studio build script and passed all 15 cases, including the optional full
translation/template comparison. The GUI run took about 65 seconds and generated
36 scene images; Japanese board-image and English main-scene images were also
inspected visually. The 12 runner/oracle/response self-tests passed. A deliberate
settings-restoration mismatch in the real GUI produced a failed native result
despite a normal process exit, and the Python driver correctly rejected it.
Local evidence is under `build/issue652/` (ignored by Git).
