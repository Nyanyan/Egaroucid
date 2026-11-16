# Book transcript compression / 棋譜からの定石圧縮

棋譜に含まれる推奨手を圧縮し、バイナリファイルとして保存します。盤面の8対称と同一局面への合流をまとめ、保持する局面の推奨手をすべて復元できることを検証します。局面は手番側と相手側の石配置の組で識別し、同じ正規化局面に集まる推奨手はすべて保持します。

Compress recommended moves from game transcripts into a binary file. The tool merges the eight board symmetries and transpositions, and verifies that every retained position and its recommended moves can be reconstructed. A position is identified by the pair of bitboards for the side to move and its opponent. All recommendations that map to the same canonical position are retained.

## 必要環境 / Requirements

- 圧縮: Python 3.9以降。追加パッケージは不要です。 / Compression: Python 3.9 or later; no third-party packages.
- C++読み込みサンプル: このリポジトリとC++20対応コンパイラ。 / C++ reader example: this repository and a C++20 compiler.

## 入力例と実行 / Sample input and compression

リポジトリのルートから次を実行します。 / Run these commands from the repository root:

```powershell
cd src/tools/book_transcript_compression
python compress_book.py --transcripts sample.jsonl
```

同梱の`sample.jsonl`は、1行につき1個のJSONオブジェクトを持つUTF-8テキストです。 / The included `sample.jsonl` is UTF-8 text with one JSON object per line:

```jsonl
{"transcript":"d3c3c4","side":0}
{"transcript":"d3c3c4","side":1}
```

この棋譜では黒が`d3`、白が`c3`、黒が`c4`へ着手します。1行目は黒の着手を、2行目は白の着手を、それぞれの着手直前の局面の推奨手として登録します。対称な局面・着手もまとめて扱います。

In this transcript, Black plays `d3`, White plays `c3`, and Black plays `c4`. The first row registers Black's moves and the second row registers White's move as recommendations for the positions immediately before those moves. Symmetric positions and moves are merged.

| フィールド / Field | 内容 / Meaning |
| --- | --- |
| `transcript` | 必須。標準初期局面からの小文字の座標列。パスは省略し、自動処理します。 / Required. Lowercase coordinates starting from the standard initial position. Omit passes; they are handled automatically. |
| `side` | 必須。推奨手を登録する色。`0`=黒、`1`=白。 / Required. Side whose moves are registered: `0`=Black, `1`=White. |
| `start_ply` | 任意、既定値`0`。この着手番号以降の推奨手を登録します。それ以前の着手も局面の再現に使います。 / Optional, default `0`. Register recommendations from this move index onward; earlier moves are still replayed. |
| `policy_plies` | 任意。登録する着手番号の配列。省略時は`start_ply`以降の指定色の全着手を登録します。指定する各番号は`start_ply`以上で、指定色の着手を指す必要があります。 / Optional. Array of move indices to register. If omitted, register all moves by `side` from `start_ply` onward. Each index must be at least `start_ply` and refer to a move by `side`. |

着手番号は`0`始まりで、パスを数えません。たとえば、次は3手目の`c4`だけを登録します。 / Move indices start at `0` and exclude passes. For example, this row registers only the third move, `c4`:

```jsonl
{"transcript":"d3c3c4","side":0,"start_ply":2,"policy_plies":[2]}
```

SQLite入力も使用できます。拡張子は`.sqlite3`、`.sqlite`、`.db`に対応し、`games`テーブルの`payload`列に同じJSONオブジェクトを文字列として保存してください。行は`id`順に読み込みます。

SQLite input is also supported for `.sqlite3`, `.sqlite`, and `.db` files. Store the same JSON objects as strings in the `games.payload` column. Rows are read in `id` order.

## 出力と上限 / Output and limits

既定では、このスクリプトと同じフォルダ内の`out`を自動作成し、次の2ファイルを保存します。 / By default, the tool creates `out` beside the script and writes:

- `out/book_data.bin`: 20バイトのヘッダーと圧縮したバイト列。 / A 20-byte header followed by compressed bytes.
- `out/book_compression.json`: 入力のSHA-256、比較した圧縮候補、選択結果、出力サイズ、復元検証結果。 / Input SHA-256 hashes, candidate encodings, selected encoding, output size, and round-trip verification results.

```powershell
# 40石以下の局面を対象に、ヘッダー込みで40,000バイト以下へ圧縮
# Keep positions with at most 40 discs; limit the entire file to 40,000 bytes
python compress_book.py --transcripts sample.jsonl --max-bytes 40000 --max-discs 40

# 保存先を指定 / Custom output paths
python compress_book.py --transcripts sample.jsonl --output out/custom_book.bin --report out/custom_report.json

# 旧形式のC++文字列を取り込み / Import legacy C++ packed strings
python compress_book.py --legacy-dir path/to/legacy_book
```

`--legacy-dir`には`black.txt`と`white.txt`を含むフォルダを指定します。 / `--legacy-dir` must point to a directory containing `black.txt` and `white.txt`.

`--max-bytes`の既定値は`99999`で、20バイトのヘッダーも上限に含みます。従来の`--max-packed-chars`は`--max-bytes`へ変更しました。`--max-discs`の既定値は`60`です。上限に収まらなければ対象局面の最大石数を1ずつ減らし、収まる最大石数を採用します。保持対象から外れた局面の推奨手は出力に含めません。

`--max-bytes` defaults to `99999`, including the 20-byte header, and replaces the former `--max-packed-chars` option. `--max-discs` defaults to `60`. If needed, the tool decreases the maximum disc count one at a time and selects the highest limit that fits. Recommendations for excluded positions are omitted.

## C++での読み込み / Reading in C++

`book_reader.hpp`は`../../engine/board.hpp`をincludeし、Egaroucidの`Board`、合法手生成、石の反転、パス、対称変換を使って復号します。`Book::lookup`は指定された盤面の向きで推奨手のビットボードを返し、未登録なら`0`を返します。次は初期局面の推奨手を取得する最小例です。

`book_reader.hpp` includes `../../engine/board.hpp` and decodes using Egaroucid's `Board`, legal move generation, flipping, passes, and symmetry transforms. `Book::lookup` returns a bitboard of recommended moves in the orientation of the supplied board, or `0` if the position is absent. This minimal example looks up the initial position:

```cpp
#include "book_reader.hpp"

int main() {
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    const auto book = transcript_book::load_book("out/book_data.bin");
    Board board;
    board.reset();
    const uint64_t recommended_moves = book.lookup(board);
    return recommended_moves ? 0 : 1;
}
```

`read_book.cpp`は読み込み、棋譜の再生、推奨手の座標表示まで行う実行用サンプルです。以下は、このツールのフォルダで実行するPowerShellとg++の例です。

`read_book.cpp` is a runnable example that loads the book, replays an optional transcript, and prints recommended coordinates. Run these PowerShell commands with g++ from this tool's directory:

```powershell
g++ -std=c++20 -O2 -DHAS_NO_AVX2 read_book.cpp -o out/read_book.exe
.\out\read_book.exe out/book_data.bin
.\out\read_book.exe out/book_data.bin d3c3
```

同梱の入力例では、初期局面と`d3c3`後の出力はそれぞれ次のとおりです。 / With the included sample input, the initial position and the position after `d3c3` produce:

```text
positions: 3
moves: d3 c4 f5 e6
```

```text
positions: 3
moves: c4
```

初期局面の4手は対称な着手です。`positions`は本全体の登録局面数です。合法手のない局面を直接`Book::lookup`へ渡す場合は、呼び出し側でパスを処理してください。 / The four initial moves are symmetric. `positions` counts registered positions in the whole book. When calling `Book::lookup` directly on a position with no legal moves, handle the pass before the lookup.

## バイナリ形式 / Binary format

このツール独自の`BTBC`形式です。整数はリトルエンディアンで保存します。 / This tool uses its own `BTBC` format. Integers are little-endian.

| オフセット / Offset | バイト数 / Bytes | 内容 / Contents |
| --- | --- | --- |
| 0 | 4 | マジック / Magic: `BTBC` |
| 4 | 1 | バージョン / Version: `1` |
| 5 | 1 | フラグ: bit 0=算術符号化、bit 1=深さ優先、bit 2=親の推奨手を符号化の文脈に使用 / Flags: bit 0=arithmetic coding, bit 1=DFS, bit 2=parent-policy context |
| 6 | 2 | 予約領域、`0` / Reserved, `0` |
| 8 | 4 | 木のノード数 / Tree node count (`uint32`) |
| 12 | 4 | 登録局面数 / Registered position count (`uint32`) |
| 16 | 4 | 圧縮データの有効ビット数 / Encoded bit count (`uint32`) |
| 20 | 可変 / Variable | 圧縮データ。各バイトの上位ビットから読み、末尾を`0`で埋めます。 / Compressed payload, most significant bit first; unused trailing bits are zero. |

## 検証 / Verification

C++サンプルをビルドした後、次を実行します。6種類の圧縮方式での復元、バイト境界、不正なヘッダー、サイズ上限、既定の保存先、およびPythonとC++の復号結果の一致を確認します。

After building the C++ example, run the following to check all six codec modes, byte boundaries, invalid headers, size limits, default output paths, and agreement between Python and C++ decoders:

```powershell
python test_codec.py -v
```

C++実行ファイルは既定で`out/read_book.exe`を探します。別の場所に置く場合は環境変数`BOOK_READER_EXE`にパスを指定してください。見つからない場合はC++との比較をスキップします。 / Tests look for `out/read_book.exe` by default. Set `BOOK_READER_EXE` to use another path. If the executable is absent, C++ comparisons are skipped.

`--export-reference`では、正規化した盤面の`player`、`opponent`、推奨手ビットボードをそれぞれ64ビット整数として、リトルエンディアンで順に保存します。各レコードは24バイトで、盤面の組をキーにソートします。以下の2ファイルは同じ内容になります。

`--export-reference` writes canonical `player`, `opponent`, and recommended-move bitboards as three consecutive little-endian 64-bit integers. Records are 24 bytes each, sorted by the board pair. The following two files have identical contents:

```powershell
python compress_book.py --transcripts sample.jsonl --export-reference out/python_reference.bin
.\out\read_book.exe out/book_data.bin --export-reference out/cpp_reference.bin
```
