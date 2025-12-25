# Egaroucid AI Coding Guidelines

このファイルは、GitHub Copilotが「これは今後のためにメモしておくべき汎用的な内容だ」と感じたら、ただちに編集してください。

## Important Notice

### Building the Project

**GUI版** (src/guiフォルダを含み、src/Egaroucid.cppがメインファイル):
- GitHub Copilot側でコンパイルはしないでください
- ビルドが必要な場合は、Visual Studioのビルドスクリプトを使用:
  - 単体ビルド: `python src/tools/release_script/build_gui.py -c SIMD`
  - 一括ビルド: `python src/tools/release_script/build_gui_all.py`
  - 構成: SIMD, Generic, AVX512 (他にSIMD_Portable, AVX512_Portable, Generic_Portableも利用可能)

**Console版** (src/guiフォルダを含まない。src/Egaroucid_for_Console.cppがメインファイル):
- Visual Studioのビルドスクリプト(推奨):
  - 単体ビルド: `python src/tools/release_script/build_console.py -c SIMD`
  - 一括ビルド: `python src/tools/release_script/build_console_all.py`
  - 構成: SIMD, Generic, AVX512, SIMD_GGS
- clang++/g++での直接コンパイルも可能:
  - `clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console_copilot.exe -mtune=native -march=native -pthread -std=c++20`
  - `g++ -O2 ./src/Egaroucid_for_Console.cpp -o  ./bin/Egaroucid_for_Console_copilot.exe -mtune=native -march=native  -mfpmath=both -pthread -std=c++20`

## Project Overview
Egaroucid is a high-performance Othello/Reversi AI engine written in C++20, licensed under GPL-3.0-or-later. The engine uses bitboard representation, advanced search algorithms, SIMD optimization (AVX2), and parallel search techniques to achieve world-class play.

## Architecture

### Core Components
- **Board Representation**: Bitboard-based (64-bit) for efficient board manipulation and move generation
- **Search Engine**: NegaScout/PVS with YBWC (Young Brothers Wait Concept) parallel search
- **Evaluation**: SIMD-optimized (AVX2) pattern-based evaluation with 612K+ parameters per phase
- **Thread Pool**: Custom thread pool (`thread_pool.hpp`) with thread ID management for parallel search
- **Transposition Table**: Hash-based move ordering and position caching
- **Opening Book**: Pre-computed optimal move database for early game

### Key Files (in `src/engine/`)
- `ai.hpp` - Main AI algorithm and search coordination
- `search.hpp` - Search structure and common definitions
- `ybwc.hpp` - YBWC parallel search implementation
- `midsearch.hpp` - Midgame search algorithms
- `endsearch.hpp` - Endgame perfect search
- `evaluate_simd.hpp` - AVX2-optimized evaluation function
- `board.hpp` - Bitboard representation and move generation
- `thread_pool.hpp` - Custom thread pool for parallel search
- `transposition_table.hpp` - Transposition table implementation
- `common.hpp` - Constants (HW=8, HW2=64, board masks, cell types)
- `setting.hpp` - Compile-time feature flags (USE_SIMD, USE_YBWC_NWS, USE_LAZY_SMP2, etc.)

## Critical Patterns & Conventions

### 1. Bitboard Fundamentals
- Board state is stored as two `uint64_t` values: `player` and `opponent`
- Cells indexed 0-63 (A1=0, H8=63) in row-major order
- Use bitwise operations for move generation and evaluation
- Constants: `HW=8` (board width/height), `HW2=64` (total cells)

### 2. Search Architecture
- **MPC (Multi-ProbCut)**: Probabilistic pruning with levels (0-100)
- **YBWC Parallelism**: Splits search at depth ≥6 (mid) or ≥16 (end)
- **Lazy SMP**: Optional multi-threaded search with varying depths/MPC levels
- **Thread IDs**: Each search assigns thread_id for thread pool coordination
- **Depth Management**: Midgame uses iterative deepening; endgame searches to completion

### 3. Evaluation System
- SIMD-based evaluation using AVX2 intrinsics (`__m256i`)
- Pattern features: hv2/3/4, diagonals, corner9, edge patterns
- Evaluation range: [-4092, 4092] for midgame
- Feature extraction in `calc_eval_features(Board*, Eval_search*)`
- Phase-based parameters (30 phases from 0-60 discs)

### 4. Thread Pool Usage
- Global `thread_pool` object manages worker threads
- Thread IDs (0-99) used to partition work and limit concurrent tasks per ID
- `push(thread_id, &pushed, task)` - returns future, sets `pushed` flag
- Set max threads per ID: `set_max_thread_size(id, max)`
- Check availability: `get_n_idle()`, `get_n_using_thread(id)`

### 5. Compile-Time Features (setting.hpp)
Check these preprocessor flags before modifying search/eval code:
- `USE_SIMD_EVALUATION` - Enable AVX2 evaluation
- `USE_YBWC_NWS` - Enable YBWC for Null Window Search
- `USE_LAZY_SMP2` - Enable Lazy SMP parallelism
- `USE_KILLER_MOVE_MO` - Enable killer move heuristic
- `USE_SEARCH_STATISTICS` - Collect search statistics

## Development Workflow

### Building
Primary build command (from project root):
```bash
clang++ -O3 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console_clang.exe -mtune=native -march=native -pthread -std=c++20 -DINCLUDE_GGS -lws2_32
```
- **Required**: `-march=native` for SIMD, `-std=c++20`, `-pthread` for threading
- **Optional flags**: `-DINCLUDE_GGS` (GGS server support), `-DIS_GGS_TOURNAMENT`

### Search Parameters
- **Level**: Search depth/strength (typically 1-60)
- **MPC Level**: 0-100 (100 = no probcut, lower = more aggressive pruning)
- **Time Management**: `time_limit` in milliseconds, uses iterative deepening
- **Book Usage**: `use_book` flag with `book_acc_level` for accuracy

### Performance Considerations
- **Bitboard Operations**: Use `pop_count_ull()` for counting, bit shifts for access
- **SIMD Alignment**: Ensure 256-bit alignment for AVX2 operations
- **Thread Coordination**: Minimize lock contention; use `searching` flags for early termination
- **Hash Table**: Prefetch transposition table entries (see Edax optimization reference)

## External References
The codebase draws inspiration from:
- **Edax-Reversi**: Stability cutoff thresholds, hash prefetching
- **Thread Pool Designs**: bshoshany/thread-pool, progschj/ThreadPool, SandSnip3r/thread-pool
- **Apache-Licensed Code**: nodec thread pool executor (see `thread_pool.hpp`)

## Common Tasks

### Adding a Search Feature
1. Check `setting.hpp` for relevant feature flags
2. Modify search logic in `midsearch.hpp` or `endsearch.hpp`
3. Update `Search` structure if new state needed
4. Ensure thread-safety if using `use_multi_thread`

### Modifying Evaluation
1. Edit pattern definitions in `evaluate_simd.hpp`
2. Update `N_PATTERN_PARAMS` if adding features
3. Regenerate evaluation weights (external training process)
4. Test with `mid_evaluate(Board*)` wrapper

### Debugging Search Issues
- Enable `USE_SEARCH_STATISTICS` for node counts
- Check `global_searching` and `*searching` flags for premature termination
- Verify thread pool state: `get_n_idle()`, `get_max_thread_size(id)`
- Use `show_log` parameter in AI functions for verbose output

### GUI (Opening Setting Scene & related `src/gui/` code)
- The forced-opening UI lives in `src/gui/opening_setting_scene.hpp`; when a folder is toggled off its entire contents (child folders and openings) must render in a gray, de-emphasized state that reflects inherited disablement, but the checkboxes and edit actions must remain usable. Only active inline-edit/rename sessions should block interaction.
- Drag & drop needs to support moving opening entries across folders, not just within the same list. Ensure folder rows expose drop targets, update the underlying storage (filesystem + in-memory vectors), and refresh both folder and opening lists so the UI stays in sync.
- Reordering inside a folder now relies exclusively on drag & drop—do not reintroduce up/down buttons.
- While editing or renaming items, keep destructive controls and bottom action buttons hidden/disabled to prevent conflicting operations; reuse the shared inline-edit helpers to keep button placement consistent.

#### Drag & Drop Visual Feedback
- **Color Constants**: Use `gui_list::DragColors` namespace in `list.hpp` for all drag-related colors (DraggedItemBackground, DropTargetFrame, etc.)
- **Yellow Frame**: Show yellow frame when dragging game OR folder over a valid drop target folder (excluding self when dragging folder)
- **Folder Drag Text**: Use black text color for readability on yellow background
- **Auto-Scroll**: Implement with `update_drag_auto_scroll()` from `list.hpp`, speed ~0.15 for smooth scrolling
- **Drop Target Logic**: Check both `drag_state.is_dragging_game` and `drag_state.is_dragging_folder` conditions, exclude self-drops

#### Scroll Management
- When adding new items (openings/games), auto-scroll to bottom if total items exceed visible window
- Calculate: `total_with_form = total_items + 1`, then `strt_idx = total_with_form - N_GAMES_ON_WINDOW`
- Always call `init_scroll_manager()` when canceling add/edit mode to reset scroll state
- Vertical ellipsis (︙) positioning: top at `sy`, bottom at `SY + height * n_games + 18`

### Board Representation & Symmetry
**Critical: Representative board calculation must follow util.hpp implementation**

When working with board symmetry (e.g., in book.hpp or related tools):
- Use the exact 8-way symmetry calculation from `util.hpp::representative_board()`
- The 8 transformations are:
  1. Original
  2. Black line mirror (transpose)
  3. Vertical mirror
  4. Black line + vertical
  5. Horizontal mirror
  6. Black line + horizontal
  7. Horizontal + vertical (180° rotation)
  8. Black line + horizontal + vertical (white line mirror)
- Always select the lexicographically smallest (player, opponent) pair as the representative
- Use `compare_representative_board()` helper to maintain consistency

### Localization & UI Text
**Critical: Always use `language.get()` for UI strings**

All user-facing text must go through the language system:
```cpp
// ❌ BAD - Hardcoded string
button.init(..., U"New CSV", ...);

// ✅ GOOD - Localized string
button.init(..., language.get("opening_setting", "new_category"), ...);
```

**Language Files** (all 4 must be updated together):
1. `bin/resources/languages/japanese.json`
2. `bin/resources/languages/english.json`
3. `bin/resources/languages/simplified_chinese.json`
4. `bin/resources/languages/traditional_chinese_taiwan.json`

**Terminology**:
- User-facing: "カテゴリ" (category) not "CSV"
- Internal code: `csv_file` variable names are OK
- All tooltips, help text, and button labels must use `language.get()`
