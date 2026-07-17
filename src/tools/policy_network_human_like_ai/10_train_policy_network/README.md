# Human-Like Policy Network Training

Related issue: #613

## English

This directory trains compact Othello/Reversi policy networks for predicting
human moves from WTHOR games.

The current report and commands are for direct WTHOR training. Earlier report
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

### Model Selection Rule

Inside one Keras run, the checkpoint named `best_model.h5` is selected by
validation `val_accuracy`, which is the unmasked exact top-1 accuracy over the
64 policy outputs.

Across different training runs, the adopted model is selected by validation
legal-masked top-1 human-move agreement. The test split is not used for model
selection; it is used only after the model is selected. If two runs have the
same validation legal-masked top-1 agreement, the tie-breakers are validation
legal-masked top-3 agreement, then fewer parameters, then shorter elapsed
training time.

Legal-masked top-N agreement means this: after the network predicts the 64-way
policy distribution, only legal moves in policy-output square order are ranked.
A position sample is a top-N hit when the WTHOR move is among the first N ranked
legal moves.

### Training Command

Example:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_sweep_512x4_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_sweep_512x4_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode shuffled `
    --configs 512x4 `
    --epochs 50 `
    --patience 8 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_sweep_512x4_e50
```

Generated models, raw logs, and resource summaries are intentionally ignored by
git.

## 日本語

このディレクトリでは、WTHOR棋譜から展開した局面を使い、人間の着手を予測する軽量な
Othello/Reversi policy networkを学習します。

現在のREADMEとRESULTSは、WTHORを直接学習データにした実験だけを対象にしています。
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

### 採用基準

1回のKeras学習の中では、`best_model.h5` は validation `val_accuracy` で選びます。
この `val_accuracy` は、合法手maskを使わない64出力全体の厳密なtop-1 accuracyです。

複数の学習条件を比較するときは、validation splitにおける合法手mask後top-1
人間着手一致率が最も高いモデルを採用します。test splitはモデル採用には使わず、
採用後の最終確認だけに使います。同率の場合は、validation合法手mask後top-3一致率、
パラメータ数の少なさ、学習時間の短さの順で比較します。

合法手mask後top-N一致率とは、networkが出した64出力のpolicy分布に対して、
policy出力のマス順で合法手だけを残して順位付けし、WTHORで実際に打たれた手が
上位N手に含まれる局面サンプルの割合です。

### 学習コマンド

例:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_sweep_512x4_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_sweep_512x4_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode shuffled `
    --configs 512x4 `
    --epochs 50 `
    --patience 8 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_sweep_512x4_e50
```

生成されたモデル、raw log、resource summaryはgit管理外です。
