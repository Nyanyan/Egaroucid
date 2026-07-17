# Human-Like Policy Network Training

Related issue: #613

## English

This directory trains compact Othello/Reversi policy networks for predicting
human moves from WTHOR games.

The current commands and reports are for direct WTHOR training. Earlier report
snapshots were moved to the local archive
`src/tools/policy_network_human_like_ai/report/legacy`.

### Terminology

- `games` means complete game transcripts.
- `position samples` means the expanded board positions stored in the binary
  board-data files. This is the unit used by the trainer.
- Existing dataset directory names such as `records1` are kept only because
  they are part of the current file layout.

### Input

The input is side-relative, not fixed black/white:

- 64 player-to-move disc bits.
- 64 opponent disc bits.
- 128 float inputs in total.

The board-data binary samples already store the two bitboards as `player` and
`opponent`, and the trainer uses them directly.

### Dataset And Split

Source:

```text
$EGAROUCID_DATA/train_data/board_data/records1
```

Use `--wthor --split-mode shuffled` to train from all WTHOR board-data position
samples. The trainer shuffles the expanded WTHOR position-sample set with seed
`613`, then splits it into train, validation, and test subsets at the position
sample level.

| split | position samples |
| --- | ---: |
| train | 6,428,225 |
| validation | 803,528 |
| test | 803,529 |
| total | 8,035,282 |

### Epoch Checkpoint Rule

For shuffled WTHOR training, each epoch is evaluated on the validation split
after masking illegal moves. `best_model.h5` is the epoch with the highest
validation legal-masked top-1 human-move agreement, stored as
`val_legal_top1`. `EarlyStopping` and `ModelCheckpoint` both monitor this
metric. If legal masks are not available, the trainer falls back to Keras
`val_accuracy`, the unmasked exact top-1 accuracy over all 64 policy outputs.

Legal-masked top-N agreement means this: after the network predicts the 64-way
policy distribution, only legal moves in policy-output square order are ranked.
A position sample is a top-N hit when the WTHOR move is among the first N ranked
legal moves.

Across different training runs, the adopted model is selected by validation
legal-masked top-1 agreement. The test split is not used for model selection; it
is used only after the model is selected. If validation top-1 is exactly equal,
the tie-breakers are validation legal-masked top-3 agreement, fewer parameters,
and shorter training time, in that order.

### Resource Metric

The result tables use `max resident memory MiB`. This is the largest main-memory
working set observed by `resource_monitor.py` while the training command was
running. On Windows it is the process working set; when `psutil` is available it
is the summed resident memory of the training process and its child processes.
It is not GPU memory and it is not the model file size. The raw JSON key remains
`peak_rss_mib` for compatibility with earlier logs.

### Training Command

Example for the current adopted model family:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_large_768x4_dropout_lr_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_large_768x4_dropout_lr_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode shuffled `
    --configs w768d4_drop005_lr5e4:768:4:0.03:0.05:0.0:0.0005 `
    --epochs 50 `
    --patience 8 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_large_768x4_dropout_lr_e50
```

Generated models, raw logs, resource summaries, and archived local reports are
intentionally ignored by git.

## 日本語

このディレクトリでは、WTHOR棋譜から展開した局面を使い、人間の着手を予測する軽量な
Othello/Reversi policy networkを学習します。

現在のコマンドとreportは、WTHORを直接学習データにした実験だけを対象にしています。
以前のreport snapshotは、ローカルアーカイブ
`src/tools/policy_network_human_like_ai/report/legacy` に移動しました。

### 用語

- `games` は、1局分の完全な棋譜を意味します。
- `局面サンプル` は、binary board-dataに保存された展開済み局面を意味します。
  学習器はこの局面サンプルを単位として扱います。
- `records1` などの既存ディレクトリ名は、現在のデータ配置の一部であるため
  パス名としてのみ使います。

### 入力

入力は固定の黒白ではなく、手番相対です。

- 手番側の石がある64マス。
- 相手側の石がある64マス。
- 合計128個のfloat入力。

board-dataのbinary sampleには `player` と `opponent` として保存されているため、
学習コードはそれをそのまま使います。

### データセットと分割

入力データ:

```text
$EGAROUCID_DATA/train_data/board_data/records1
```

`--wthor --split-mode shuffled` を使うと、WTHOR由来の全局面サンプルを読み込みます。
seed `613` で局面サンプル全体をshuffleし、train、validation、testに分割します。
この分割の単位は局面サンプルです。

| split | 局面サンプル数 |
| --- | ---: |
| train | 6,428,225 |
| validation | 803,528 |
| test | 803,529 |
| total | 8,035,282 |

### epoch checkpointの基準

WTHOR shuffled trainingでは、各epoch終了時にvalidation splitで合法手mask後の
一致率を計算します。`best_model.h5` は、validation合法手mask後top-1人間着手一致率
`val_legal_top1` が最も高かったepochのモデルです。`EarlyStopping` と
`ModelCheckpoint` はどちらもこの値を監視します。合法手maskがない学習モードでは、
Kerasの `val_accuracy`、つまり64出力全体に対するmaskなしの厳密なtop-1 accuracyに
fallbackします。

合法手mask後top-N一致率とは、networkが出した64出力のpolicy分布に対して、
policy出力のマス順で合法手だけを残して順位付けし、WTHORで実際に打たれた手が
上位N手に含まれる局面サンプルの割合です。

複数の学習条件を比較するときは、validation合法手mask後top-1一致率が最も高い
モデルを採用します。test splitはモデル採用には使わず、採用後の最終確認だけに
使います。validation top-1が完全に同じ場合は、validation合法手mask後top-3一致率、
パラメータ数の少なさ、学習時間の短さの順で比較します。

### リソース指標

結果表の `max resident memory MiB` は、学習コマンド実行中に
`resource_monitor.py` が観測した最大のメインメモリ常駐量です。Windowsではprocess
working setを使います。`psutil` が使える場合は、学習processと子processのresident
memoryの合計です。GPUメモリ量ではなく、モデルファイルサイズでもありません。
過去ログとの互換性のため、raw JSONのkey名は `peak_rss_mib` のままです。

### 学習コマンド

現在の採用モデル系の例:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_large_768x4_dropout_lr_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_large_768x4_dropout_lr_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode shuffled `
    --configs w768d4_drop005_lr5e4:768:4:0.03:0.05:0.0:0.0005 `
    --epochs 50 `
    --patience 8 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_large_768x4_dropout_lr_e50
```

生成されたモデル、raw log、resource summary、archive済みlocal reportはgit管理外です。
