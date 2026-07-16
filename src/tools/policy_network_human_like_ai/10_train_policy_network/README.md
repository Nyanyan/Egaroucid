# Human-Like Policy Network Training

Related issue: #613

## English

This directory trains compact Othello/Reversi policy networks for the
human-like AI experiment. It is based on `src/tools/policy_network`, with
dataset selection, resource logging, WTHOR agreement evaluation, blending with
Egaroucid Console, and strength-test utilities added for the experiment.

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
5. Evaluate pure policy agreement on WTHOR position samples.
6. Evaluate blended policy/Egaroucid agreement with resumable shards and an
   optional SQLite hint-score cache.
7. Run strength tests with resumable JSONL output.

Training example:

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

WTHOR and blend helpers:

```powershell
python src\tools\policy_network_human_like_ai\20_test_with_wthor\analyze_wthor_position_duplicates.py
python src\tools\policy_network_human_like_ai\20_test_with_wthor\run_wthor_blend_shards.py `
  --output-dir src\tools\policy_network_human_like_ai\20_test_with_wthor\output\blend_wthor_full_chunked `
  --resume-from-completed-prefix `
  --positions-per-shard 20 `
  --jobs-per-shard 4 `
  --egaroucid-threads 8 `
  --time-limit-sec 3600 `
  --merge-completed `
  --blend-params '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0' `
  --top-n '1,2,3,4,5,8,10,16'
```

The sharded blend runner creates a shared SQLite hint-score cache by default at
`20_test_with_wthor/output/blend_wthor_full_chunked/hint_score_cache.sqlite3`
when the command above is used.
Use `--no-hint-cache` to disable it or `--hint-cache-db` to choose another path.
Use `--positions-per-shard`, `--range-start` / `--range-end`, and
`--time-limit-sec` to advance the full WTHOR run in resumable chunks. Use
`--merge-completed` to keep `partial_merged` updated even before the full WTHOR
run is complete. `manifest.json` stores only a first/last shard preview by
default; tune it with `--manifest-shard-preview`. Use
`--resume-from-completed-prefix` to continue from the end of the completed
contiguous shard prefix.

Strength-test helper:

```powershell
python src\tools\policy_network_human_like_ai\40_test_strength\run_strength_full.py `
  --resume `
  --time-limit-sec 3600
```

The strength runner writes completed games to `strength_games.jsonl` and can
resume the full 120,000-game schedule.

Generated model artifacts, raw logs, and evaluation outputs are intentionally
ignored by git.

## 日本語

このディレクトリは、人間らしいAI実験用の軽量な Othello/Reversi policy network を学習します。
`src/tools/policy_network` を土台にして、実験用のデータ選択、リソースログ、WTHOR一致率評価、
Egaroucid Console とのブレンド評価、強さ評価用のユーティリティを追加しています。

用語:

- `games`: 完全な棋譜、つまり対局数です。
- `position_samples`: binary board-data に書き出された局面サンプルです。
- `records0` や `records1` という既存ディレクトリ名は、現在のデータ配置の一部なのでパス名としてだけ残しています。

入力特徴は手番相対です。

- 手番側の石 64 bit。
- 相手側の石 64 bit。
- 合計 128 float 入力。

board-data の binary sample には `player` と `opponent` として保存されており、
学習コードはそれをそのまま使います。入力は固定の黒白ではありません。

流れ:

1. `train_data/transcript_release/0002` からランダムに対局を選びます。
2. 選んだ棋譜を `train_data/transcript/Egaroucid_Train_Data_v2_selected` に書き出します。
3. `train_data/board_data/Egaroucid_Train_Data_v2_selected/records0` に board-data を作ります。
4. TensorFlow/Keras で複数の軽量 policy network を学習します。
5. WTHOR 局面サンプルで policy 単体の一致率を評価します。
6. resumable shard と SQLite の hint-score cache を使って、policy と Egaroucid のブレンド一致率を評価します。
7. JSONL に途中結果を残しながら強さ評価を行います。

学習例:

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

WTHOR・ブレンド評価の補助コマンド:

```powershell
python src\tools\policy_network_human_like_ai\20_test_with_wthor\analyze_wthor_position_duplicates.py
python src\tools\policy_network_human_like_ai\20_test_with_wthor\run_wthor_blend_shards.py `
  --output-dir src\tools\policy_network_human_like_ai\20_test_with_wthor\output\blend_wthor_full_chunked `
  --resume-from-completed-prefix `
  --positions-per-shard 20 `
  --jobs-per-shard 4 `
  --egaroucid-threads 8 `
  --time-limit-sec 3600 `
  --merge-completed `
  --blend-params '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0' `
  --top-n '1,2,3,4,5,8,10,16'
```

shard 版のブレンド評価は、既定で
上のコマンドでは
`20_test_with_wthor/output/blend_wthor_full_chunked/hint_score_cache.sqlite3`
に共有 SQLite hint-score cache を作ります。無効化する場合は `--no-hint-cache`、保存先を変える場合は
`--hint-cache-db` を使います。
`--positions-per-shard`、`--range-start` / `--range-end`、`--time-limit-sec` を使うと、
全WTHOR実行を再開可能な小さい単位で進められます。
`--merge-completed` を使うと、全WTHORが完走する前でも `partial_merged` を更新できます。
`manifest.json` は既定で shard の先頭・末尾 preview だけを保存します。表示数は
`--manifest-shard-preview` で調整できます。
`--resume-from-completed-prefix` を使うと、完了済みの連続 shard prefix の末尾から自動で再開できます。

強さ評価の補助コマンド:

```powershell
python src\tools\policy_network_human_like_ai\40_test_strength\run_strength_full.py `
  --resume `
  --time-limit-sec 3600
```

強さ評価 runner は完了済み対局を `strength_games.jsonl` に保存し、120,000対局の full schedule を再開できます。

生成されたモデル、raw log、評価出力は git 管理外です。
