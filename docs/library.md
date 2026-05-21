# Egaroucid Engine Library (Experimental)

`libegaroucid` is an experimental C ABI wrapper for the Egaroucid engine.

- Status: experimental / may change
- License: GPL-3.0-or-later (same as Egaroucid)
- Public header: `#include <egaroucid/egaroucid.h>`

This API is intended for external frontends and language bindings (Python/Rust/C#/etc.) without exposing internal C++ engine headers.

## Build

Library-only build:

```bash
cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF
cmake --build build_lib --config Release
```

Default console build remains available (for example, `BUILD_CONSOLE=ON`).

## Public API

Header:

```c
#include <egaroucid/egaroucid.h>
```

Main functions:

- `egaroucid_version()`
- `egaroucid_global_init(resource_dir)`
- `egaroucid_create()` / `egaroucid_destroy()`
- `egaroucid_search_array(...)`
- `egaroucid_stop(...)`

## Resource Directory

Call `egaroucid_global_init()` before search.

`resource_dir` should point to a directory that contains:

- `eval.egev2`
- `eval_move_ordering_end.egev`
- `book.egbk3`
- `hash/` (hash files such as `hash25.eghs`)

Typical example:

- `bin/resources`

The initializer also accepts a parent directory that contains `resources/`.

## Board Representation

`egaroucid_search_array()` takes `int board[64]` and `player`.

- Cell values:
  - `-1`: empty (`EGAROUCID_EMPTY`)
  - `0`: black (`EGAROUCID_BLACK`)
  - `1`: white (`EGAROUCID_WHITE`)
- Player values:
  - `0`: black to move
  - `1`: white to move

Index mapping is:

- `0 = a1`, `1 = b1`, ..., `7 = h1`
- `8 = a2`, ..., `63 = h8`

Returned `result.move` uses the same index mapping.
`-1` means pass / no move.

## Minimal Usage

See:

- `examples/cpp/simple.cpp`

This example initializes the library, creates an engine, searches from the initial position, prints the selected move/value, and destroys the engine.

Build and run the example:

```bash
cmake --build build_lib --config Release --target egaroucid_cpp_example
./build_lib/examples/Release/egaroucid_cpp_example bin/resources 12
```

On Windows (PowerShell):

```powershell
.\build_lib\examples\Release\egaroucid_cpp_example.exe bin/resources 12
```

Arguments:

- 1st argument: resource directory (default: `bin/resources`)
- 2nd argument: number of plies to play from the initial position (default: `10`)

## Current Limitations

- API is experimental and may change.
- `time_limit_ms` is currently accepted but ignored.
- Error reporting is intentionally minimal in this first version.
- Language-specific bindings are not included yet; they can be added as separate wrappers later.
