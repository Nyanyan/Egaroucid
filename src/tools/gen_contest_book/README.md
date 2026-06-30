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

既定値は、レベル21、1手あたり3石損まで、1局合計6石損まで、30空きで打ち切りです。これらはコマンドラインオプションで変更できます。

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

Defaults are level 21, per-move loss 3, total loss 6, and cut at 30 empties. These can be changed with command-line options.

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
