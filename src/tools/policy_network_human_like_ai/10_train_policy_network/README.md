# Human-Like Policy Network Training

Related issue: #613

This directory trains a compact Othello/Reversi policy network from selected
Egaroucid Train Data v2 games. It was copied from `src/tools/policy_network`
and adapted for the human-like AI experiment.

関連 issue: #613

このディレクトリは、Egaroucid Train Data v2 から選択した棋譜を使って、
軽量な Othello/Reversi policy network を学習するためのものです。
`src/tools/policy_network` をコピーし、人間らしい AI 実験向けに調整しています。

## Input Format / 入力形式

The network input is side-relative:

- 64 player-to-move disc bits
- 64 opponent disc bits
- total 128 float inputs

ネットワーク入力は手番相対です。

- 手番側の石 64 bit
- 相手側の石 64 bit
- 合計 128 個の float 入力

The board-data records store these fields as `player` and `opponent`, and the
training code uses them directly. The input is not fixed black/white.

board data には `player` と `opponent` として保存されており、学習コードは
それをそのまま使います。入力は固定の黒白ではありません。

## Pipeline / 手順

1. Select random games from `train_data/transcript_release/0002`.
2. Write the selected games to
   `train_data/transcript/Egaroucid_Train_Data_v2_selected`.
3. Convert them to board data under
   `train_data/board_data/Egaroucid_Train_Data_v2_selected/records0`.
4. Train several compact TensorFlow/Keras policy networks.
5. Save raw training logs under `train_log`.

1. `train_data/transcript_release/0002` からランダムに棋譜を選びます。
2. 選んだ棋譜を
   `train_data/transcript/Egaroucid_Train_Data_v2_selected` に保存します。
3. それを
   `train_data/board_data/Egaroucid_Train_Data_v2_selected/records0` に変換します。
4. TensorFlow/Keras で複数の軽量 policy network を学習します。
5. 生の学習ログは `train_log` に保存します。

## Select Games / 棋譜選択

`--num-games` is required. The source files are assumed to contain 10,000 games
per `.txt`, following the dataset README.

`--num-games` は必須です。データセット README に従い、各 `.txt` には
10,000 局が入っている前提でサンプリングします。

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\select_transcripts.py --num-games 100000 --seed 613
```

## Convert To Board Data / board data への変換

Directory names in `transcript_release/0002` are random-opening depths. The
converter plays those random moves but does not write training records before
that depth.

`transcript_release/0002` のディレクトリ名はランダム初期手数です。変換時は
その手数までは盤面だけ進め、学習レコードとしては出力しません。

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\convert_selected_transcripts_to_board_data.py
```

## Training / 学習

Run several candidate model sizes and capture raw logs:

複数の候補モデルを学習し、生ログを保存します。

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_training_experiments.py `
  --configs 64x3,96x3,128x3,96x4 `
  --epochs 24 `
  --patience 6 `
  --max-train-samples 2000000 `
  --max-val-samples 200000 `
  --batch-size 8192
```

The trainer writes Keras `.h5` models and C++-loadable binary weights. Generated
model artifacts are ignored by git.

学習スクリプトは Keras `.h5` と C++ から読める binary weights を出力します。
生成されたモデル成果物は git 管理外です。

## Smoke Test / 煙突テスト

Use tiny sample sizes to check the whole pipeline:

全体の流れだけ確認する小規模テストです。

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\select_transcripts.py --num-games 20 --seed 613
python src\tools\policy_network_human_like_ai\10_train_policy_network\convert_selected_transcripts_to_board_data.py
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_training_experiments.py --configs 16x1 --epochs 1 --patience 1 --max-train-samples 128 --max-val-samples 32 --batch-size 32 --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\log_smoke
```

## C++ Sample / C++ サンプル

`policy_network_sample.cpp` loads the exported binary weights and prints a
policy distribution from a board or transcript. It is kept compatible with the
original `src/tools/policy_network` weight format.

`policy_network_sample.cpp` は出力された binary weights を読み、盤面または棋譜
から policy 分布を表示します。重み形式は元の `src/tools/policy_network` と
互換です。
