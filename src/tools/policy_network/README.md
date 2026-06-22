# Policy Network / ポリシーネットワーク

Related issue: #613

関連 issue: #613

This directory contains a lightweight TensorFlow/Keras policy network for
Othello/Reversi. The model predicts a 64-square move distribution from board
data records.

このディレクトリには、Othello/Reversi 用の軽量な TensorFlow/Keras policy
network が入っています。board data の局面から 64 マスの着手分布を予測します。

## Input / 入力

The input is side-relative, not fixed black/white:

入力は固定の黒/白ではなく、手番相対です:

- 64 player-to-move disc bits
- 64 opponent disc bits
- total 128 float inputs

- 手番側の石の 64 bit
- 相手側の石の 64 bit
- 合計 128 個の float 入力

The board-data format stores these fields as `player` and `opponent`; the
training and evaluation scripts use them directly.

board data には `player` と `opponent` として保存されているので、学習・評価
スクリプトはそれをそのまま使います。

## Model / モデル

The selected model is small enough for fast C++ inference while keeping good
accuracy:

最終モデルは C++ で高速に推論できるサイズを保ちつつ、精度が良かった構成です:

- hidden width: 128
- hidden depth: 3
- activation: LeakyReLU, `alpha=0.03`
- output: 64-way softmax policy
- parameters: 57,792

- 中間層幅: 128
- 中間層数: 3
- activation: LeakyReLU, `alpha=0.03`
- 出力: 64 手 softmax policy
- パラメータ数: 57,792

## Training / 学習

Training data is read from:

学習データは以下から読みます:

```powershell
$env:EGAROUCID_DATA + "\train_data\board_data"
```

Records `records259` through `records310` are used by default.

デフォルトでは `records259` から `records310` を使います。

Quick smoke test:

簡単な動作確認:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 16x1 --epochs 1 --max-train-samples 2000 --max-val-samples 512
```

Search several lightweight shapes:

軽量な構成をいくつか探索:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 64x3,96x3,128x3,96x4 --epochs 12 --patience 4 --max-train-samples 500000 --max-val-samples 100000 --batch-size 8192
```

Final training used:

最終学習では以下を使いました:

```powershell
python -u src\tools\policy_network\train_policy_network.py --configs 128x3 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192 --output-dir src\tools\policy_network\trained\playerop_final_issue613_128x3
```

The binary C++ weights are written as:

C++ 用の binary weight は以下に出力されます:

```text
src/tools/policy_network/trained/playerop_final_issue613_128x3/best_policy_network_weights.bin
```

`trained/` is ignored by git because model artifacts are large/generated.

`trained/` は生成物が大きいため git 管理外です。

## WTHOR Human-Game Evaluation / WTHOR 人間棋譜評価

Evaluate legal-masked top-N accuracy on WTHOR human-game board data:

WTHOR 人間棋譜 board data に対して、合法手で mask した top-N 一致率を評価します:

```powershell
python src\tools\policy_network\evaluate_policy_topn.py --batch-size 65536 --predict-batch-size 8192 --verbose
```

The evaluator reads the WTHOR board-data directory
`$EGAROUCID_DATA/train_data/board_data/records1`, masks the policy distribution
to legal moves, and checks whether the actual human move is inside the top N
legal moves. Generated output file names use `wthor`.

評価スクリプトは WTHOR board data ディレクトリ
`$EGAROUCID_DATA/train_data/board_data/records1` を読み、policy 分布を合法手だけに
mask してから、実際の人間の手が top N 合法手以内に入るかを調べます。生成される
ファイル名には `wthor` を使います。

## C++ Sample / C++ サンプル

Build:

ビルド:

```powershell
g++ -std=c++17 -O3 src\tools\policy_network\policy_network_sample.cpp -o src\tools\policy_network\policy_network_sample.exe
```

Run from a board string:

盤面文字列から実行:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\playerop_final_issue613_128x3\best_policy_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 10
```

Run from a transcript:

棋譜から実行:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\playerop_final_issue613_128x3\best_policy_network_weights.bin --transcript d3c5e6f5 --top 10
```

For board strings, `X`, `x`, `B`, `b`, and `*` mean black; `O`, `o`, `W`,
`w`, and `1` mean white; `-` and `.` mean empty. Add `--side white` when the
side to move is white.

盤面文字列では `X`, `x`, `B`, `b`, `*` が黒、`O`, `o`, `W`, `w`, `1` が白、
`-`, `.` が空きマスです。白番を入力する場合は `--side white` を付けます。
