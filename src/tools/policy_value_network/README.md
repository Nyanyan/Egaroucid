# Policy-Value Network / ポリシー・バリューネットワーク

Related issue: #613

関連 issue: #613

This directory contains a lightweight TensorFlow/Keras policy-value network for
Othello/Reversi. It follows the AlphaZero-style idea of a shared trunk with two
heads: a policy vector `p` and a scalar value `v`.

このディレクトリには、Othello/Reversi 用の軽量な TensorFlow/Keras
policy-value network が入っています。AlphaZero 風に、共有 trunk から policy
ベクトル `p` と value スカラー `v` の 2 head を出します。

Reference / 参考:

- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv:1712.01815, https://arxiv.org/abs/1712.01815

## Input / 入力

The input is side-relative:

入力は手番相対です:

- 64 player-to-move disc bits
- 64 opponent disc bits
- total 128 float inputs

- 手番側の石の 64 bit
- 相手側の石の 64 bit
- 合計 128 個の float 入力

This is the same corrected player/opponent input as `src/tools/policy_network`.

これは `src/tools/policy_network` と同じ、修正後の player/opponent 入力です。

## Outputs / 出力

- `policy`: 64-way softmax move distribution
- `value`: tanh scalar in `[-1, 1]`

- `policy`: 64 手 softmax 着手分布
- `value`: `[-1, 1]` の tanh スカラー

The value target is `score / 64` from board data. The score is used from the
player-to-move perspective.

value 教師は board data の `score / 64` です。score は手番側目線の値として使って
います。

## Selected Model / 採用モデル

The best balanced model from the experiments was:

実験で policy と value のバランスが良かった構成は以下です:

- hidden width: 128
- hidden depth: 3
- activation: LeakyReLU, `alpha=0.03`
- value loss weight: 0.1
- parameters: 57,921

- 中間層幅: 128
- 中間層数: 3
- activation: LeakyReLU, `alpha=0.03`
- value loss weight: 0.1
- パラメータ数: 57,921

## Training / 学習

Records `records259` through `records310` are used by default.

デフォルトでは `records259` から `records310` を使います。

Quick smoke test:

簡単な動作確認:

```powershell
python src\tools\policy_value_network\train_policy_value_network.py --configs 16x1 --epochs 1 --patience 1 --max-train-samples 2000 --max-val-samples 512 --batch-size 256
```

Search:

探索:

```powershell
python -u src\tools\policy_value_network\train_policy_value_network.py --configs pv96w01:96:3:0.1,pv96w025:96:3:0.25,pv128w01:128:3:0.1,pv128w025:128:3:0.25,pv128w05:128:3:0.5,pv96d4w025:96:4:0.25 --epochs 12 --patience 4 --max-train-samples 500000 --max-val-samples 100000 --batch-size 8192 --output-dir src\tools\policy_value_network\trained\search_issue613
```

Final training:

最終学習:

```powershell
python -u src\tools\policy_value_network\train_policy_value_network.py --configs pv128w01:128:3:0.1 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192 --output-dir src\tools\policy_value_network\trained\final_issue613_pv128w01
```

The C++ binary weights are written as:

C++ 用の binary weight は以下に出力されます:

```text
src/tools/policy_value_network/trained/final_issue613_pv128w01/best_policy_value_network_weights.bin
```

`trained/` is ignored by git because model artifacts are large/generated.

`trained/` は生成物が大きいため git 管理外です。

## C++ Sample / C++ サンプル

Build:

ビルド:

```powershell
g++ -std=c++17 -O3 src\tools\policy_value_network\policy_value_network_sample.cpp -o src\tools\policy_value_network\policy_value_network_sample.exe
```

Run from a board string:

盤面文字列から実行:

```powershell
src\tools\policy_value_network\policy_value_network_sample.exe src\tools\policy_value_network\trained\final_issue613_pv128w01\best_policy_value_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 8
```

Add `--side white` or `--side black` when the side to move should be explicit.

手番を明示したい場合は `--side white` または `--side black` を付けます。

Run from a transcript:

棋譜から実行:

```powershell
src\tools\policy_value_network\policy_value_network_sample.exe src\tools\policy_value_network\trained\final_issue613_pv128w01\best_policy_value_network_weights.bin --transcript f5d6c3 --top 8
```

The sample prints the side to move, value from that side's perspective, a disc
difference estimate (`value * 64`), and the top policy moves.

サンプルは手番、手番側目線の value、石差推定値 (`value * 64`)、policy 上位手を
表示します。
