# コンテストbook生成ツール

このフォルダには、コンテスト用の独自形式bookを作るための棋譜生成スクリプトとbook構築スクリプトがあります。

生成されるbookは開始局面ごとに別ファイルとなり、`trained` フォルダに `.egcb` 形式で保存されます。`trained` は `.gitignore` の `**/trained` によりgit管理対象外です。

`data/book_records` の開始局面別フォルダ名と `trained` のbookファイル名には、`records321_14_random_setup` 内の通し番号を `0000000_...` の形で先頭に付けます。

## コンソールのコンパイル

リポジトリルートで実行してください。

```powershell
clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console.exe -mtune=native -march=native -pthread -std=c++20
```

## 棋譜生成

1つの開始局面から棋譜を生成します。

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 512 --threads 1
```

開始局面一覧を上から順に処理して棋譜を生成します。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 512 --threads 1 --resume
```

途中の開始局面から走査したい場合は、一覧に含まれる局面を `--start-board` に指定します。指定した局面を含めて処理を開始します。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --start-board "<initial board>" --resume
```

`--skip` と併用すると、`--start-board` の局面からさらに指定数だけ飛ばします。

既定値は、開始局面あたり512棋譜、32棋譜ごとの反復更新、レベル21、1手あたり2石損まで、1局合計4石損まで、30空きで打ち切りです。これらはコマンドラインオプションで変更できます。

棋譜生成は、合計ロス0の棋譜をすべて列挙してから合計ロス1へ進む、という順序で進みます。指定した棋譜数に達するか、上限lossまで列挙し終わると終了します。

`generate_records.py` は `--batch-size 32` ごとに一旦 `build_book.py` を実行し、`trained` に仮bookを作ります。次のバッチからはその仮bookを `-contestbook` としてEgaroucidに渡し、bookにある手評価を活用しながら残りの手を探索して棋譜生成を続けます。既存の棋譜は開始局面フォルダ内の transcript で重複判定し、同じ棋譜は再保存しません。

`--threads` を2以上にすると、Egaroucid側は共有タスクキューで並列化します。1本しか進行がない間はその探索に全スレッドを使い、棋譜上の分岐で新しい進行が増えたらキューへ追加します。各workerは空き次第、自分でキューから次の進行を取り出して処理します。実行中タスクと待ちタスクの合計がworker数を下回った場合は、合法手評価や30空き完全読みの直前に空きworker分のhelper slotを予約し、残っている各タスクが複数スレッドを使えるようにします。

## book構築

1つの開始局面のbookを構築します。

```powershell
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

開始局面一覧を上から順に処理してbookを構築します。

```powershell
python src/tools/gen_contest_book/build_all_books.py --resume
```

book構築時は、生成棋譜に加えて `data/game_records` 内の実戦棋譜も同じ形式として読み込みます。開始局面から合計4石損までの局面を収録する設定が既定値です。

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
clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console.exe -mtune=native -march=native -pthread -std=c++20
```

## Generate Records

Generate records for one start:

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 512 --threads 1
```

Generate records for starts in list order:

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 512 --threads 1 --resume
```

To resume scanning from a specific start position, pass a board from the start list to `--start-board`. The specified position is included.

```powershell
python src/tools/gen_contest_book/generate_all_records.py --start-board "<initial board>" --resume
```

When combined with `--skip`, the script skips that many additional positions after `--start-board`.

Defaults are 512 records per start, iterative updates every 32 records, level 21, per-move loss 2, total loss 4, and cut at 30 empties. These can be changed with command-line options.

Record generation exhausts all records with total loss 0 before moving to total loss 1, and continues in increasing-loss order. It stops when the requested number of records is reached or all records up to the configured loss limit are exhausted.

`generate_records.py` runs `build_book.py` after each `--batch-size 32` records and writes a provisional book under `trained`. The next batch passes that provisional book back to Egaroucid through `-contestbook`, so move scores already present in the book are reused while missing legal moves are still searched. Existing records are deduplicated by transcript in the start-position record directory and are not written again.

When `--threads` is 2 or larger, Egaroucid parallelizes with a shared task queue. While there is only one progression, that search can use all threads. When branch points create more progressions, they are pushed to the queue, and each worker pulls the next progression as soon as it becomes free. If the number of running plus queued tasks drops below the worker count, Egaroucid reserves helper slots before scoring each legal move and before the 30-empty exact solve, so the remaining tasks can use multiple threads without exceeding the configured budget.

## Build Books

Build one book:

```powershell
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

Build books in start-list order:

```powershell
python src/tools/gen_contest_book/build_all_books.py --resume
```

The builder reads generated records and also treats game records in `data/game_records` as the same record format. By default, it records positions up to a total loss of 4 discs from the start position.

## Runtime Loading

The console does not load a contest book by default. To enable it, pass the hidden command-line option `-contestbook <dir>`; it is accepted by normal and GGS builds but is not shown in help. In normal use, point it at `src/tools/gen_contest_book/trained`.

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
