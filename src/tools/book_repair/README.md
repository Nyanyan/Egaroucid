# Full book leaf repair

`recalculate_leaves.cpp` rebuilds every leaf from the current search engine and
evaluation files. It visits every record, including unreachable positions and
previously undefined leaves. A leaf is the best legal move whose child is absent
from the book; child membership accounts for passes and both terminal orientations.

The default preserves an existing leaf level in 1..60. An undefined leaf uses its
board's level in 1..60, otherwise Lv5. `--level N` explicitly overrides this policy.
Each search runs to the engine's requested depth/selectivity without a time limit
or a Book lookup. Independent roots share the normal synchronized transposition
table. The original record order, boards, board values, board levels and line counts
are preserved byte for byte. Records without an unregistered legal move receive
an undefined value, `MOVE_NOMOVE`, and an undefined level.

On Windows, use Visual Studio 2022 Community with C++ tools and an AVX2-capable CPU.
The Python runner builds with MSVC `/O2 /arch:AVX2`; other platforms use g++ with
`-O3 -march=native -pthread` (Windows/MSVC is the validated configuration).
Python 3.9+ is required. Optional `psutil` adds measured CPU, thread and RAM status.

Hash levels 25..29 are supported. A level-29 table contains 536,870,912 entries.
The validated MSVC/AVX2 build uses 24 bytes per entry (12 GiB for the table and
about 12.3 GiB for the repair process); other build configurations may differ.
The supplied resources stop at level 27;
prepare a separate resource snapshot for a larger table:

```powershell
python src/tools/book_repair/prepare_hash_resources.py --resources bin/resources --output ignored/hash29-resources --hash-level 29
```

Pass that directory as `--resources` and set `--hash-level 29` in `run_repair.py`.
The helper copies the evaluation files and deterministically generates missing
hash data using the engine's random fallback contract. SHA-256 identities and the
seed are saved. Hash data changes transposition-table indexing, not search level.
An existing checkpoint intentionally rejects different binaries/hash settings;
changing them requires a validated migration, preserving the old checkpoint and
checking its original engine, input, model and completed results before import.

Run a pilot in a visible PowerShell console:

```powershell
./src/tools/book_repair/run_visible.ps1 -WorkDir D:/github/othello/Egaroucid/ignored/book-leaf-repair -Threads 24 -Limit 128
```

Resume that exact calculation, rebuild all leaves, validate, and install the result
to the default, release-template, and joined books:

```powershell
./src/tools/book_repair/run_visible.ps1 -WorkDir D:/github/othello/Egaroucid/ignored/book-leaf-repair -Threads 24 -Install
```

For other destinations, invoke `run_repair.py` directly with repeated
`--install-to PATH`. Without these arguments it only produces `repaired.egbk3`
in the work directory. Never install an incomplete pilot output.

The work directory contains an immutable input copy, SHA-256 identities for engine,
binary, model and hash resources, a checkpoint flushed after each completed search,
progress logs with elapsed time and ETA, and JSON reports. The runner locks the work
directory against duplicate runs. Closing the console stops the calculation; rerun
the same work directory to resume. In-flight searches are repeated. A partial final
checkpoint line is discarded; complete malformed, duplicate or invalid records are
rejected. Changed resources or engine binaries cannot reuse the checkpoint.

Before installation the tool verifies every leaf's score, legal move, absence from
registered children, and requested level, and verifies every non-leaf byte against
the input. Output is reread and verified before renaming. Existing destination files
are backed up by SHA-256 before replacement; files changed by another process
during the calculation are not overwritten. All installed copies must match the
verified output hash. `repair-report.json`, `validation.json`, and `result.json`
record the calculation and installation. Preserve these with release provenance.

`check_repair.py --exe PATH --work-dir NEW_PATH` runs a 950-position fixture with
independent Python Othello rules. It checks all eight symmetries, passes, terminal
positions, exact two-empty scores, level fallback, preservation of non-leaf bytes,
pilot/resume, partial checkpoint recovery, and rejection of corrupt output,
duplicate checkpoint rows, changed inputs and truncated files.

To audit without searching, use the built executable:

```powershell
./ignored/book-leaf-repair/recalculate_leaves.exe --mode audit --input bin/resources/book.egbk3 --report ignored/book-leaf-repair/audit.json
```
