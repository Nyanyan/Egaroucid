# Human-Like Policy Network Training

Related issue: #613

## English

This directory trains compact Othello/Reversi policy networks for the
human-like AI experiment. It is based on `src/tools/policy_network`, with
dataset selection, resource logging, and WTHOR agreement evaluation added for
the experiment.

Terminology:

- `games`: complete game transcripts.
- `position_samples`: board positions written to the binary board-data files.
- Existing board-data directory names such as `records0` and `records1` are
  kept only because they are part of the current data layout.

Input features are side-relative:

- 64 player-to-move disc bits.
- 64 opponent disc bits.
- 128 float inputs in total.

The board-data binary samples store those bitboards as `player` and
`opponent`; the trainer uses them directly. The input is not fixed black/white.

Pipeline:

1. Select random games from `train_data/transcript_release/0002`.
2. Write selected games to `train_data/transcript/Egaroucid_Train_Data_v2_selected`.
3. Convert them to board data under `train_data/board_data/Egaroucid_Train_Data_v2_selected/records0`.
4. Train several compact TensorFlow/Keras policy networks.
5. Save raw logs and resource summaries under `train_log`.

Example:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\select_transcripts.py --num-games 1000000 --seed 613
python src\tools\policy_network_human_like_ai\10_train_policy_network\convert_selected_transcripts_to_board_data.py
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_training_experiments.py `
  --configs 64x3,96x3,128x3,96x4,160x3,192x3,128x4 `
  --epochs 20 `
  --patience 5 `
  --max-train-samples 5000000 `
  --max-val-samples 500000 `
  --batch-size 8192
```

The trainer writes Keras `.h5` models and C++-loadable binary weights.
Generated model artifacts and raw logs are intentionally ignored by git.

## 日本語

このディレクトリは、人間らしい AI 実験用の軽量な Othello/Reversi policy
network を学習するためのものです。`src/tools/policy_network` をもとに、
実験用データ選択、リソースログ、WTHOR 一致率評価を追加しています。

用語:

- `games`: 完全な棋譜の対局数です。
- `position_samples`: binary board-data に書き出した局面サンプル数です。
- `records0` や `records1` という既存ディレクトリ名は、現在のデータ配置名
  としてのみ残しています。

入力特徴は手番相対です。

- 手番側の石 64 bit。
- 相手側の石 64 bit。
- 合計 128 float 入力。

board-data の binary sample には `player` と `opponent` として保存されており、
学習コードはそれをそのまま使います。入力は固定の黒白ではありません。

流れ:

1. `train_data/transcript_release/0002` からランダムに対局を選びます。
2. 選んだ棋譜を `train_data/transcript/Egaroucid_Train_Data_v2_selected` に書きます。
3. `train_data/board_data/Egaroucid_Train_Data_v2_selected/records0` に変換します。
4. TensorFlow/Keras で複数の軽量 policy network を学習します。
5. 生ログとリソースサマリを `train_log` に保存します。

学習スクリプトは Keras `.h5` と C++ から読める binary weights を出力します。
生成物と生ログは git 管理外です。
