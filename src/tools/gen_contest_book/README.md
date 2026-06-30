# Contest Book Tools

This directory contains scripts for generating records and building the independent contest book format.

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

Defaults are level 21, per-move loss 3, total loss 6, and cut at 32 empties. These can be changed with command-line options.

## Build Books

Build one book:

```powershell
python src/tools/gen_contest_book/build_book.py "<initial board>"
```

Build books in start-list order:

```powershell
python src/tools/gen_contest_book/build_all_books.py --resume
```

The generated book files are written under `data/books`. Runtime loading uses the same sanitized start-board filename with the `.egcb` extension.
