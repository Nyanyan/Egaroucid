# コンテストbook生成ツール

このフォルダには、コンテスト用の独自形式bookを作るための棋譜生成スクリプトとbook構築スクリプトがあります。

生成されるbookは開始局面ごとに別ファイルとなり、`trained` フォルダに `.egcb` 形式で保存されます。`trained` は `.gitignore` の `**/trained` によりgit管理対象外です。

`data/book_records` の開始局面別フォルダ名と `trained` のbookファイル名には、`records321_14_random_setup` 内の通し番号を `0000000_...` の形で先頭に付けます。

## コンソールのコンパイル

リポジトリルートで実行してください。

```powershell
clang++ -O3 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console_clang.exe -mtune=native -march=native -pthread -std=c++20 -DINCLUDE_GGS -lws2_32 -DIS_GGS_TOURNAMENT
```

## 棋譜生成

1つの開始局面から棋譜を生成します。

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 512 --threads 1
```

既定では、`data/r14_random_setup_probability_priority_20260722.jsonl` の上から順、すなわちリポジトリ内の `random_setup(14)` 実装における発生確率が高い開始局面から棋譜を生成します。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 512 --threads 1
```

`--resume` と併用すると、目標棋譜数に達した局面はEgaroucidを起動せずに飛ばします。`--skip` と `--limit` は、この優先度順の位置を基準にします。従来の `records321_14_random_setup` のファイル順を使う場合だけ、`--start-list-order` を指定してください。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 256 --threads 28 --resume --skip 534
```

`--games` は再実行のたびに追加する棋譜数ではなく、開始局面ごとのユニーク棋譜の目標総数です。既に一部の棋譜がある状態で `--games 512` を再実行すると合計512件まで生成し、既に512件以上あれば棋譜生成を行いません。

全開始局面の処理では `--resume` を付けると、既に目標総数へ到達した局面について子プロセスの起動も省略します。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 512 --resume
```

途中の開始局面から走査したい場合は、一覧に含まれる局面を `--start-board` に指定します。指定した局面を含めて処理を開始します。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --start-board "<initial board>"
```

`--skip` と併用すると、`--start-board` の局面からさらに指定数だけ飛ばします。

既定値は、開始局面あたり512棋譜、16棋譜ごとの反復更新、レベル19、1手あたり2石損まで、1局合計4石損まで、30空きで打ち切りです。打ち切り局面の `leaf value` は30手99.9%読みで評価します。棋譜数、ロス制限、打ち切り空き数などはコマンドラインオプションで変更できます。

生成engineの既定値は大会用 `bin/Egaroucid_for_Console_clang.exe` です。別binaryを使う既存運用では、単局・全局面のどちらのdriverでも `--exe <path>` で上書きできます。

棋譜生成は、合計ロス0の棋譜をすべて列挙してから合計ロス1へ進む、という順序で進みます。ユニーク棋譜の目標総数に達するか、上限lossまで列挙し終わると終了します。

`generate_records.py` は `--batch-size 16` ごとに一旦 `build_book.py` を実行し、`trained` に仮bookを作ります。次のバッチからはその仮bookを `-contestbook` としてEgaroucidに渡し、bookにある手評価を活用しながら残りの手を探索して棋譜生成を続けます。既存の棋譜は開始局面フォルダ内の transcript で重複判定し、同じ棋譜は再保存しません。

開始局面ごとの `generation_manifest.json` には、生成binaryのpath・SHA-256、探索設定、各batch前後の棋譜file hashとユニーク棋譜数を記録します。中断中のbatchや外部から追加された棋譜は、その旨を次回起動時に履歴へ残します。同じ棋譜フォルダへの生成と同じbookへの公開はOS管理のlockで直列化します。`.lock` file自体は残りますが、lock状態はプロセス終了時にOSが解放するため、異常終了後にfileを削除する必要はありません。

`--use-existing-book` を指定した場合、manifestに利用設定は残りますが、engine標準bookの内容まではhash化しません。厳密な生成provenanceが必要な場合は、既定の `-nobook` 動作を使ってください。

`--threads` を2以上にすると、Egaroucid側は共有タスクキューで並列化します。1本しか進行がない間はその探索に全スレッドを使い、棋譜上の分岐で新しい進行が増えたらキューへ追加します。各workerは空き次第、自分でキューから次の進行を取り出して処理します。実行中タスクと待ちタスクの合計がworker数を下回った場合は、合法手評価や葉の30手99.9%読みの直前に空きworker分のhelper slotを予約し、残っている各タスクが複数スレッドを使えるようにします。

## book構築

1つの開始局面のbookを構築します。

```powershell
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

これは低水準の直接builderであり、既存出力へ直接書きます。検証済みstaging、manifest、排他lockが必要な管理用構築では `build_all_books.py` を使ってください。

開始局面一覧を上から順に処理してbookを構築します。

```powershell
python src/tools/gen_contest_book/build_all_books.py --resume
```

`generate_records.py` と `build_all_books.py` によるbook公開では、一時ファイルへ構築し、ヘッダと各行を検証してから既存 `.egcb` をatomicに置き換えます。同じ場所の `.egcb.manifest.json` には、正規化した開始局面、構築オプション、builder・生成provenanceを含む入力・出力のhashを記録します。`--resume` ではbookとmanifestがともに検証を通り、入力とオプションも一致する場合だけスキップします。manifestのない従来book、途中までしか書かれていないbook、入力棋譜や設定が変わったbookは再構築します。

book構築時は、生成棋譜に加えて `data/game_records` 内の実戦棋譜も同じ形式として読み込みます。開始局面から合計4石損までの局面を収録する設定が既定値です。生成棋譜に `leaf empty` がある場合、`build_book.py` はその空き数に合わせてbook掲載範囲を自動決定します。新規の30空き棋譜では30空きまで出力できます。過去の28空き棋譜だけから構築する場合は28空きまで、過去の30空き棋譜だけから構築する場合は30空きまで出力できます。明示したい場合は `--cut-empty 28` や `--cut-empty 30` を指定してください。

## 対局時の読み込み

コンソール側は、既定ではcontest bookを読み込みません。使う場合だけ、helpには表示されない隠しコマンドラインオプション `-contestbook <dir>` を指定してください。通常は `src/tools/gen_contest_book/trained` を指定します。

## テスト方法

まず1つの開始局面だけで小さく試します。開始局面には `data/records321_14_random_setup` 内の1行を指定してください。

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 16 --batch-size 4 --threads 1
```

bookファイルが `trained` に1開始局面1ファイルで作られていることを確認します。

```powershell
Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 5
Get-Content -Encoding UTF8 (Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 1).FullName -TotalCount 8
```

先頭に `# contest_book_v1` があり、以降に `<64マスの盤面> X <評価値> <手>:<スコア> ...` の行が出ていれば、少なくとも構築形式は正しく出力されています。

対局時の読み込み確認は、`-contestbook` と `-noise` を指定してコンソールを起動し、ログを見ます。開始局面に対応するbookが見つかると `contest book loaded ...`、`go` でbookから着手できると `contest book selected ...` が出ます。`hint` では `contest book hinted ...`、`analyze` では `contest book analyzed ...` が出ます。

```powershell
.\bin\Egaroucid_for_Console.exe -noise -contestbook src/tools/gen_contest_book/trained <other options>
```

実戦投入前には、対象の開始局面で `go` / `hint` / `analyze` がcontest bookを参照すること、選ばれた手が合法手であること、bookがない開始局面では通常探索に戻ることを確認してください。

---

# Contest Book Tools

This directory contains scripts for generating records and building the independent contest book format.

Generated books are stored as one `.egcb` file per start position under the `trained` directory. The `trained` directory is ignored by git through the `**/trained` rule in `.gitignore`.

Start-specific directories under `data/book_records` and book files under `trained` are prefixed with the start-list serial number from `records321_14_random_setup`, for example `0000000_...`.

## Compile Console

Run from the repository root:

```powershell
clang++ -O3 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console_clang.exe -mtune=native -march=native -pthread -std=c++20 -DINCLUDE_GGS -lws2_32 -DIS_GGS_TOURNAMENT
```

## Generate Records

Generate records for one start:

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 512 --threads 1
```

By default, generate records in the order recorded in
`data/r14_random_setup_probability_priority_20260722.jsonl`: higher occurrence
probability under the repository-local `random_setup(14)` implementation comes
first.

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 512 --threads 1
```

Combined with `--resume`, a position that already has the target record count
is skipped without launching Egaroucid.  `--skip` and `--limit` refer to this
priority order.  Specify `--start-list-order` only when the historical
`records321_14_random_setup` file order is needed.

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 256 --threads 28 --resume --skip 534
```

`--games` is a target total of unique records for each start, not an amount to
append on every invocation. Re-running either generator with `--games 512`
continues a partial directory up to 512 and does no record-generation work for
a directory that already contains 512 or more unique transcripts.

Use `--resume` with the all-start driver to avoid spawning the per-start process
for starts that have already reached the target:

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 512 --resume
```

To resume scanning from a specific start position, pass a board from the start list to `--start-board`. The specified position is included.

```powershell
python src/tools/gen_contest_book/generate_all_records.py --start-board "<initial board>"
```

When combined with `--skip`, the script skips that many additional positions after `--start-board`.

Defaults are 512 records per start, iterative updates every 16 records, level 19, per-move loss 2, total loss 4, and cut at 30 empties. The cutoff position's `leaf value` is evaluated with a 30-ply 99.9% selective endgame search. Record counts, loss limits, and the cutoff empty count can be changed with command-line options.

The default generation engine is the tournament build at
`bin/Egaroucid_for_Console_clang.exe`. Existing workflows can select another
binary with `--exe <path>` on either the single-start or all-start driver.

Record generation exhausts all records with total loss 0 before moving to total loss 1, and continues in increasing-loss order. It stops when the target total of unique records is reached or all records up to the configured loss limit are exhausted.

`generate_records.py` runs `build_book.py` after each `--batch-size 16` records and writes a provisional book under `trained`. The next batch passes that provisional book back to Egaroucid through `-contestbook`, so move scores already present in the book are reused while missing legal moves are still searched. Existing records are deduplicated by transcript in the start-position record directory and are not written again.

Each start directory has a `generation_manifest.json` recording the generation
binary path and SHA-256, search settings, and record-file hashes and unique
counts before and after every batch. Interrupted batches and externally changed
record sets are recorded explicitly on the next run. Generation for the same
record directory and publication of the same book are serialized with OS-owned
locks. The `.lock` files remain on disk, but their lock state is released by the
OS when a process exits, so they do not become stale after a crash.

With `--use-existing-book`, the manifest records that setting but does not hash
the engine's separate standard-book data. Use the default `-nobook` behavior
when strict generation provenance is required.

When `--threads` is 2 or larger, Egaroucid parallelizes with a shared task queue. While there is only one progression, that search can use all threads. When branch points create more progressions, they are pushed to the queue, and each worker pulls the next progression as soon as it becomes free. If the number of running plus queued tasks drops below the worker count, Egaroucid reserves helper slots before scoring each legal move and before the leaf 30-ply 99.9% search, so the remaining tasks can use multiple threads without exceeding the configured budget.

## Build Books

Build one book:

```powershell
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

This is the low-level direct builder and still writes directly to its output.
Use `build_all_books.py` when validated staging, manifests, and publication
locking are required.

Build books in start-list order:

```powershell
python src/tools/gen_contest_book/build_all_books.py --resume
```

Book publication through `generate_records.py` and `build_all_books.py` is
transactional for each `.egcb`: `build_book.py` writes a staging file, the
driver validates its header and rows, and only then atomically replaces the
published file. A neighboring `.egcb.manifest.json` records the normalized
start, build options, builder/input hashes (including generation provenance),
and output hash. With `--resume`, a book is skipped only when both the book and
manifest validate and all inputs and options still match. A legacy book without
a manifest, a partial/corrupt book, or a book whose records/options changed is
rebuilt.

The builder reads generated records and also treats game records in `data/game_records` as the same record format. By default, it records positions up to a total loss of 4 discs from the start position. When generated records contain `leaf empty`, `build_book.py` automatically uses that empty count as the book output cutoff. New 30-empty records produce a 30-empty book, while old 28-empty or 30-empty records can still be rebuilt at their own cutoff. Use `--cut-empty 28` or `--cut-empty 30` to force a cutoff explicitly.

## Runtime Loading

The console does not load a contest book by default. To enable it, pass the hidden command-line option `-contestbook <dir>`; it is accepted by normal and GGS builds but is not shown in help. In normal use, point it at `src/tools/gen_contest_book/trained`.

## Root Table

`contest_root_table.egcb` is an optional, single-file table of verified moves
at a fixed root disc count (normally 14 for `s8r14`). It complements rather
than replaces the per-start deep books: the runtime first uses a matching deep
book, then falls back to the root table only at the initial board. It never
uses a root-table row after the root ply.

Build a temporary table from verified individual books with:

```powershell
python src/tools/gen_contest_book/build_root_table.py --books-dir src/tools/gen_contest_book/trained --require-starts-dir src/tools/gen_contest_book/data/records321_14_random_setup --output ignored/ggs_620_progress/temporary_contest_root_table.egcb
```

The builder canonicalizes every board and move with the same representative
ordering as the C++ runtime, rejects conflicting duplicate roots, validates
the completed table, and writes a neighbouring source/output hash manifest.
For a full enumerated start set, pass that set with `--require-starts-dir`; the
build fails unless coverage is exact. Shallow teacher results can therefore be
accumulated for all starts independently of the smaller collection of deep
books. A compact teacher artifact can avoid creating one file per root: pass
repeatable `--root-results <file>` inputs with data rows in the same
`<board> <side> <value> <move>:<score> ...` format. Its hash is recorded in the
root-table manifest alongside deep-book sources.

`build_root_table.py` deliberately refuses to write the tournament path
`trained/contest_root_table.egcb` directly. That file is published only after
the complete fixed-start, color-swapped match has passed its audit. The
publication command reruns the recorded audit with the fixed seed and 100,000
bootstrap repetitions, checks every input SHA-256 again, rebuilds the table
from the frozen teacher rows, and requires it to match the table used in the
games. It then writes a `contest_root_table.egcb.publication.json` record
linking the published table to the audit and its inputs:

```powershell
python src/tools/gen_contest_book/publish_verified_root_table.py --audit-json ignored/ggs_620_progress/root_table_match_audit.md.json
```

An existing tournament table is not replaced unless `--replace-existing` is
given explicitly. To add later rows, prepare, play, and audit the complete
replacement table; the publisher never silently mixes unaudited old rows.

To generate a new, disjoint batch of verified starting moves, pass every
previous accepted teacher file or published root table with repeatable
`--exclude-root-results`. The generator canonicalizes those rows before
selection, records each exclusion file's SHA-256 in its state and manifest, and
refuses to resume if that list changes. For example:

```powershell
python src/tools/gen_contest_book/generate_ggs_root_teacher.py --coverage ignored/ggs_620_progress/r14_corpus_coverage_reaudit_20260722.json --exe bin/Egaroucid_for_Console_clang.exe --output ignored/ggs_620_progress/new_teacher_rows.txt --method time_then_verify --time-seconds 60 --threads 28 --hash 29 --min-depth 30 --min-selectivity 74 --fallback-level 30 --verify-level 31 --cohort-seed 623 --limit 50 --exclude-root-results ignored/ggs_620_progress/r14_seed622_50_t60_verified.txt
```

`--time-seconds 60` passes `-time 60` to Console: it gives each color a
60-second clock. It is not a requirement that every root search run for 60
seconds. Console's time-management code assigns the actual search time from
the board and the remaining clock, which is the behavior used in a GGS game.

The new rows remain outside `trained` until a fixed-start color-swapped
comparison against the same executable with no book has a valid audit and does
not include a neutral value in its pre-specified confidence intervals.

When the completed formal comparison has selected the level-30/level-31
method, pass its JSON report to the teacher generator. This rejects an
incomplete or unsuccessful comparison, a different corpus, a different
Console or evaluation file, and any teacher setting other than the selected
`hint_then_verify`, level 30, level 31, 28-thread, hash-29, depth-30,
selectivity-74 setup. The report and its `experiment_state.json` are recorded
in the teacher state and manifest, so a resume also rejects changed evidence:

```powershell
python src/tools/gen_contest_book/generate_ggs_root_teacher.py --coverage ignored/ggs_620_progress/r14_corpus_current_audit_20260722.json --exe bin/Egaroucid_for_Console_clang.exe --output ignored/ggs_620_progress/r14_priority_teacher_rows.txt --method hint_then_verify --teacher-level 30 --verify-level 31 --min-depth 30 --min-selectivity 74 --threads 28 --hash 29 --random-seed 620 --root-order ggs-r14-probability --priority-manifest ignored/ggs_620_progress/r14_random_setup_probability_priority_20260722.jsonl --formal-comparison-report ignored/ggs_620_progress/r14_formal_root_teacher_method_comparison_seeded_20260722/formal_comparison_report.json --limit 500 --checkpoint-every 25
```

For a long calculation, use `--checkpoint-every 500`. Each completed position
is first written and flushed to a small companion file. After 500 positions,
the generator flushes and `fsync`s each replacement file, atomically rewrites
the complete output, state, and manifest, then removes the companion file. On
`--resume`, any companion-file entries are replayed before the next search.
This avoids rewriting the complete result set after every position while
retaining completed positions after an interruption.

To write a Japanese-and-English progress report that includes the durable
per-position records not yet compacted into the state file, run:

```powershell
python src/tools/gen_contest_book/report_ggs_root_teacher_progress.py --state ignored/ggs_620_progress/new_teacher_rows.txt.state.json --output ignored/ggs_620_progress/new_teacher_rows_progress.md
```

After a compacted teacher output has reached the required number of processed
positions, prepare a hash-recorded temporary table and starting-position list
without selecting rows by game results:

```powershell
python src/tools/gen_contest_book/prepare_root_table_match.py --teacher-results ignored/ggs_620_progress/new_teacher_rows.txt --output-dir ignored/ggs_620_progress/new_root_table_match_input --minimum-processed 500
```

After the color-swapped games are complete, audit the complete match set,
the prepared table, the run metadata, and the engine logs before considering
any addition to `trained`:

```powershell
python src/tools/gen_contest_book/audit_root_table_matches.py --results ignored/ggs_620_progress/root_table_matches.jsonl --prepared-input ignored/ggs_620_progress/new_root_table_match_input/prepared_match_input.json --metadata ignored/ggs_620_progress/root_table_matches.jsonl.meta.json --output ignored/ggs_620_progress/root_table_match_audit.md --bootstrap-seed 624 --minimum-processed 500
```

For a time-bounded teacher calculation that intentionally did not require
level-31 verification, the match runner refuses the input by default. Pass
`--allow-unverified-teacher` only to measure that temporary table in the same
two-game-per-start protocol:

```powershell
python src/tools/gen_contest_book/run_prepared_root_table_match.py --prepared-input ignored/ggs_620_progress/time_bounded_match_input/prepared_match_input.json --output ignored/ggs_620_progress/time_bounded_matches.jsonl --allow-unverified-teacher
```

The audit records that level-31 verification was absent and therefore always
sets `eligible_for_adoption` to false. Such a result can guide a later,
separately calculated and fully verified table, but the command cannot publish
the temporary table to `trained`.

## Testing

Start with a small test for one start position. Use one line from `data/records321_14_random_setup` as `<initial board>`.

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 16 --batch-size 4 --threads 1
```

Check that one `.egcb` file per start position was written under `trained`.

```powershell
Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 5
Get-Content -Encoding UTF8 (Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 1).FullName -TotalCount 8
```

The file should start with `# contest_book_v1`, followed by lines like `<64-cell board> X <value> <move>:<score> ...`. If so, the build output is at least structurally valid.

For runtime loading, run the console with `-contestbook` and `-noise`, then watch the log. When a matching start-position book is found, the log shows `contest book loaded ...`. When `go` selects a move from the book, it shows `contest book selected ...`. The `hint` command logs `contest book hinted ...`, and `analyze` logs `contest book analyzed ...`.

```powershell
.\bin\Egaroucid_for_Console.exe -noise -contestbook src/tools/gen_contest_book/trained <other options>
```

Before using the book in a real match, confirm that `go` / `hint` / `analyze` use the contest book for the target start position, that selected moves are legal, and that positions without a matching book fall back to normal search.
