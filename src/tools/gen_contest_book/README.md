# コンテストbook生成ツール

このフォルダには、コンテスト用の独自形式bookを作るための棋譜生成スクリプトとbook構築スクリプトがあります。

生成されるbookは開始局面ごとに別ファイルとなり、`trained` フォルダに `.egcb` 形式で保存されます。`trained` は `.gitignore` の `**/trained` によりgit管理対象外です。

## コンソールのコンパイル

リポジトリルートで実行してください。

```powershell
clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console.exe -mtune=native -march=native -pthread -std=c++20
```

## 棋譜生成

1つの開始局面から棋譜を生成します。

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 128 --threads 1
```

開始局面一覧を上から順に処理して棋譜を生成します。

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 128 --threads 1 --resume
```

既定値は、レベル21、1手あたり2石損まで、1局合計4石損まで、30空きで打ち切りです。これらはコマンドラインオプションで変更できます。

棋譜生成は、合計ロス0の棋譜をすべて列挙してから合計ロス1へ進む、という順序で進みます。指定した棋譜数に達するか、上限lossまで列挙し終わると終了します。

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

コンソール側は、既定で `src/tools/gen_contest_book/trained` を参照します。別フォルダを使う場合は `-contestbook <dir>` を指定してください。

## テスト方法

まず1つの開始局面だけで小さく試します。開始局面には `data/records321_14_random_setup` 内の1行を指定してください。

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 16 --threads 1
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

bookファイルが `trained` に1開始局面1ファイルで作られていることを確認します。

```powershell
Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 5
Get-Content -Encoding UTF8 (Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 1).FullName -TotalCount 8
```

先頭に `# contest_book_v1` があり、以降に `<64マスの盤面> X <評価値> <手>:<スコア> ...` の行が出ていれば、少なくとも構築形式は正しく出力されています。

対局時の読み込み確認は、コンソールを `-noise` つきで起動してログを見ます。開始局面に対応するbookが見つかると `contest book loaded ...`、bookから着手できると `contest book selected ...` が出ます。

```powershell
.\bin\Egaroucid_for_Console.exe -noise -contestbook src/tools/gen_contest_book/trained <other options>
```

実戦投入前には、対象の開始局面で `contest book selected` が出ること、選ばれた手が合法手であること、bookがない開始局面では通常探索に戻ることを確認してください。

---

# Contest Book Tools

This directory contains scripts for generating records and building the independent contest book format.

Generated books are stored as one `.egcb` file per start position under the `trained` directory. The `trained` directory is ignored by git through the `**/trained` rule in `.gitignore`.

## Compile Console

Run from the repository root:

```powershell
clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console.exe -mtune=native -march=native -pthread -std=c++20
```

## Generate Records

Generate records for one start:

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 128 --threads 1
```

Generate records for starts in list order:

```powershell
python src/tools/gen_contest_book/generate_all_records.py --games 128 --threads 1 --resume
```

Defaults are level 21, per-move loss 2, total loss 4, and cut at 30 empties. These can be changed with command-line options.

Record generation exhausts all records with total loss 0 before moving to total loss 1, and continues in increasing-loss order. It stops when the requested number of records is reached or all records up to the configured loss limit are exhausted.

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

The console looks under `src/tools/gen_contest_book/trained` by default. Use `-contestbook <dir>` to override the runtime directory.

## Testing

Start with a small test for one start position. Use one line from `data/records321_14_random_setup` as `<initial board>`.

```powershell
python src/tools/gen_contest_book/generate_records.py "<initial board>" --games 16 --threads 1
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

Check that one `.egcb` file per start position was written under `trained`.

```powershell
Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 5
Get-Content -Encoding UTF8 (Get-ChildItem src/tools/gen_contest_book/trained/*.egcb | Select-Object -First 1).FullName -TotalCount 8
```

The file should start with `# contest_book_v1`, followed by lines like `<64-cell board> X <value> <move>:<score> ...`. If so, the build output is at least structurally valid.

For runtime loading, run the console with `-noise` and watch the log. When a matching start-position book is found, the log shows `contest book loaded ...`. When a move is actually selected from the book, it shows `contest book selected ...`.

```powershell
.\bin\Egaroucid_for_Console.exe -noise -contestbook src/tools/gen_contest_book/trained <other options>
```

Before using the book in a real match, confirm that `contest book selected` appears for the target start position, that the selected move is legal, and that positions without a matching book fall back to normal search.
