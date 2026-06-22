# Policy Network / 方策ネットワーク

Related issue: #613

関連 issue: #613

This tool trains and evaluates a compact Othello policy network with
`tensorflow.keras`.

このツールは `tensorflow.keras` を使って、軽量なオセロ方策ネットワークを学習・評価します。

## Model / モデル

Input:

- 64 black-disc bits
- 64 white-disc bits

入力:

- 黒石有無 64 bit
- 白石有無 64 bit

Output:

- 64-way softmax policy distribution
- Coordinate mapping is the Egaroucid policy index: `a1 -> 63`, `h8 -> 0`

出力:

- 64 手の softmax 方策分布
- 座標は Egaroucid の policy index に合わせます: `a1 -> 63`, `h8 -> 0`

The current selected architecture is `128x3` with LeakyReLU `alpha=0.03`
and 57,792 parameters. It was selected from the search in `RESULTS.md`.

現在の採用構成は LeakyReLU `alpha=0.03` の `128x3`、57,792 parameters です。
選定理由と探索結果は `RESULTS.md` にあります。

## Training Data / 学習データ

Training reads board data from `$EGAROUCID_DATA/train_data/board_data/records259`
through `records310`. Each board record is 19 bytes:

学習では `$EGAROUCID_DATA/train_data/board_data/records259` から `records310`
までの board data を読みます。各レコードは 19 byte です:

1. `uint64` player-to-move bitboard / 手番側 bitboard
2. `uint64` opponent bitboard / 相手側 bitboard
3. `int8` player color (`0` black, `1` white) / 手番色 (`0` 黒, `1` 白)
4. `int8` policy / 実際の着手
5. `int8` score / スコア

The learner converts player/opponent bitboards back to fixed black/white inputs
before training.

学習時には、手番側/相手側 bitboard を黒/白固定の 128 入力へ戻してから使います。

## Training / 学習

Small smoke test:

簡単な動作確認:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 16x1 --epochs 1 --max-train-samples 2000 --max-val-samples 512
```

Hyper-parameter search example:

ハイパーパラメータ探索例:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 48x3,64x3,80x3,64x4 --epochs 10 --patience 3 --max-train-samples 300000 --max-val-samples 50000 --batch-size 4096
```

Final training example:

最終学習例:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 128x3 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192
```

Artifacts are written under `src/tools/policy_network/trained/<timestamp>/`.
That directory is ignored by git.

成果物は `src/tools/policy_network/trained/<timestamp>/` に出力されます。
このディレクトリは git 管理外です。

## Human-Game Evaluation / 人間棋譜での評価

Evaluate legal-masked top-N accuracy on records1 human games:

records1 の人間棋譜に対して、合法手で mask した top-N 一致率を評価します:

```powershell
python src\tools\policy_network\evaluate_policy_topn.py --output-dir src\tools\policy_network\trained\records1_eval
```

The evaluator reads `$EGAROUCID_DATA/train_data/board_data/records1`, masks the
network output to legal moves, and checks whether the actual human move is
within the top N moves.

評価スクリプトは `$EGAROUCID_DATA/train_data/board_data/records1` を読み、
NN 出力を合法手だけに mask し、実際の人間手が top N 以内に入るかを調べます。

## C++ Sample / C++ サンプル

Build:

ビルド:

```powershell
g++ -std=c++17 -O3 src\tools\policy_network\policy_network_sample.cpp -o src\tools\policy_network\policy_network_sample.exe
```

Show policy distribution from a transcript:

棋譜から方策分布を表示:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\<run>\best_policy_network_weights.bin --transcript f5d6c3 --top 10
```

Show policy distribution from a board string:

盤面文字列から方策分布を表示:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\<run>\best_policy_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 10
```

`BOARD65` is 64 board characters plus a side-to-move character. `X`, `0`, and
`*` mean black; `O` and `1` mean white; `-` and `.` mean empty.

`BOARD65` は 64 文字の盤面と 1 文字の手番です。`X`, `0`, `*` は黒、`O`, `1`
は白、`-`, `.` は空きマスです。
