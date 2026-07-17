# WTHOR Policy Network Training Results

Related issue: #613

Date: 2026-07-17

## English

### Dataset

The source data was the expanded WTHOR board data at:

```text
$EGAROUCID_DATA/train_data/board_data/records1
```

The trainer shuffled all WTHOR position samples with seed `613` and split them
at the position-sample level.

| split | position samples |
| --- | ---: |
| train | 6,428,225 |
| validation | 803,528 |
| test | 803,529 |
| total | 8,035,282 |

### Evaluation Definition

Human-move agreement is computed after masking illegal moves. For each position
sample, the 64 network outputs are ranked only among legal moves in
policy-output square order. Top-N agreement is the fraction of position samples
where the WTHOR move is included in the highest-ranked N legal moves.

The validation split is used for model selection. The test split is used only
for final reporting after selection.

### Model Selection Rule

The adopted model is the run with the highest validation legal-masked top-1
agreement. If validation legal-masked top-1 is equal, the tie-breakers are
validation legal-masked top-3 agreement, fewer parameters, and shorter elapsed
training time, in that order.

Within a single Keras run, `best_model.h5` is the checkpoint with the best
unmasked validation `val_accuracy`. The split agreement table below is computed
after reloading that checkpoint.

### Sweep Results

All runs used TensorFlow/Keras, batch size 8192, evaluation batch size 65536,
prediction batch size 8192, and seed `613`.

| config | params | requested epochs | epochs ran | best epoch | val exact top-1 | val legal top-1 | val legal top-3 | val legal top-5 | test legal top-1 | test legal top-3 | test legal top-5 | test legal top-10 | elapsed sec | peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 20 | 20 | 18 | 42.914% | 45.541% | 76.758% | 89.795% | 45.590% | 76.834% | 89.749% | 99.407% | 58.7 | 10,396.0 |
| 192x3 | 111,232 | 20 | 20 | 20 | 46.927% | 48.852% | 80.082% | 91.888% | 48.901% | 80.109% | 91.858% | 99.563% | 61.8 | 10,337.6 |
| 256x4 | 246,848 | 20 | 20 | 20 | 50.432% | 51.606% | 82.701% | 93.476% | 51.658% | 82.786% | 93.490% | 99.701% | 69.0 | 10,399.7 |
| 384x4 | 517,696 | 20 | 20 | 20 | 53.941% | 54.665% | 85.363% | 94.921% | 54.776% | 85.426% | 94.920% | 99.801% | 84.2 | 10,384.1 |
| 512x4 | 886,848 | 20 | 20 | 20 | 55.686% | 56.167% | 86.699% | 95.621% | 56.136% | 86.817% | 95.610% | 99.836% | 91.3 | 10,406.5 |
| 256x4 | 246,848 | 50 | 50 | 50 | 53.273% | 54.128% | 84.997% | 94.703% | 54.159% | 85.013% | 94.694% | 99.796% | 125.6 | 10,437.7 |
| 384x4 | 517,696 | 50 | 50 | 50 | 56.344% | 56.839% | 87.315% | 95.934% | 56.871% | 87.330% | 95.913% | 99.860% | 160.2 | 10,424.0 |
| 512x4 | 886,848 | 50 | 50 | 50 | 58.173% | 58.389% | 88.093% | 96.255% | 58.486% | 88.170% | 96.243% | 99.851% | 186.5 | 10,417.5 |

### Adopted Model

The adopted model is `512x4` trained for 50 epochs, because it has the highest
validation legal-masked top-1 agreement in this sweep: 58.389%.

Its held-out test agreement is:

| top N | test legal-masked agreement |
| ---: | ---: |
| 1 | 58.486% |
| 3 | 88.170% |
| 5 | 96.243% |
| 10 | 99.851% |

Artifacts:

- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/best_model.h5`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/best_policy_network_weights.bin`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/best_summary.json`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/w512_d4_a0.03/split_topn_accuracy.csv`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_sweep_512x4_e50.log`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_sweep_512x4_e50_resource.json`

Earlier report snapshots were moved to:

```text
src/tools/policy_network_human_like_ai/report/legacy
```

## 日本語

### データセット

入力データは、WTHORから展開した次のboard dataです。

```text
$EGAROUCID_DATA/train_data/board_data/records1
```

seed `613` でWTHOR全局面サンプルをshuffleし、局面サンプル単位でtrain、
validation、testに分割しました。

| split | 局面サンプル数 |
| --- | ---: |
| train | 6,428,225 |
| validation | 803,528 |
| test | 803,529 |
| total | 8,035,282 |

### 評価方法

人間着手一致率は、合法手mask後のpolicy network出力で計算しました。各局面サンプルで、
64個のnetwork出力のうち合法手だけをpolicy出力のマス順で残して順位付けします。
top-N一致率は、WTHORで実際に打たれた手が上位N手に含まれた局面サンプルの割合です。

validation splitはモデル採用に使います。test splitは、採用後の最終確認だけに使います。

### モデル採用基準

採用モデルは、validation splitの合法手mask後top-1一致率が最も高いrunです。
同率の場合は、validation合法手mask後top-3一致率、パラメータ数の少なさ、
学習時間の短さの順で比較します。

1回のKeras学習内では、`best_model.h5` は合法手maskなしのvalidation
`val_accuracy` が最良だったcheckpointです。下のsplit一致率は、そのcheckpointを
読み戻してから計算しています。

### 実験結果

全runでTensorFlow/Keras、batch size 8192、evaluation batch size 65536、
prediction batch size 8192、seed `613` を使いました。

| config | params | 指定epoch | 実行epoch | best epoch | val exact top-1 | val legal top-1 | val legal top-3 | val legal top-5 | test legal top-1 | test legal top-3 | test legal top-5 | test legal top-10 | elapsed sec | peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 20 | 20 | 18 | 42.914% | 45.541% | 76.758% | 89.795% | 45.590% | 76.834% | 89.749% | 99.407% | 58.7 | 10,396.0 |
| 192x3 | 111,232 | 20 | 20 | 20 | 46.927% | 48.852% | 80.082% | 91.888% | 48.901% | 80.109% | 91.858% | 99.563% | 61.8 | 10,337.6 |
| 256x4 | 246,848 | 20 | 20 | 20 | 50.432% | 51.606% | 82.701% | 93.476% | 51.658% | 82.786% | 93.490% | 99.701% | 69.0 | 10,399.7 |
| 384x4 | 517,696 | 20 | 20 | 20 | 53.941% | 54.665% | 85.363% | 94.921% | 54.776% | 85.426% | 94.920% | 99.801% | 84.2 | 10,384.1 |
| 512x4 | 886,848 | 20 | 20 | 20 | 55.686% | 56.167% | 86.699% | 95.621% | 56.136% | 86.817% | 95.610% | 99.836% | 91.3 | 10,406.5 |
| 256x4 | 246,848 | 50 | 50 | 50 | 53.273% | 54.128% | 84.997% | 94.703% | 54.159% | 85.013% | 94.694% | 99.796% | 125.6 | 10,437.7 |
| 384x4 | 517,696 | 50 | 50 | 50 | 56.344% | 56.839% | 87.315% | 95.934% | 56.871% | 87.330% | 95.913% | 99.860% | 160.2 | 10,424.0 |
| 512x4 | 886,848 | 50 | 50 | 50 | 58.173% | 58.389% | 88.093% | 96.255% | 58.486% | 88.170% | 96.243% | 99.851% | 186.5 | 10,417.5 |

### 採用モデル

採用モデルは、50epoch学習した `512x4` です。このsweepで最も高い
validation合法手mask後top-1一致率 58.389% を示したためです。

採用モデルのheld-out test splitでの一致率は次の通りです。

| top N | test合法手mask後一致率 |
| ---: | ---: |
| 1 | 58.486% |
| 3 | 88.170% |
| 5 | 96.243% |
| 10 | 99.851% |

生成物:

- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/best_model.h5`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/best_policy_network_weights.bin`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/best_summary.json`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_sweep_512x4_e50/w512_d4_a0.03/split_topn_accuracy.csv`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_sweep_512x4_e50.log`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_sweep_512x4_e50_resource.json`

以前のreport snapshotは次のローカルアーカイブに移動済みです。

```text
src/tools/policy_network_human_like_ai/report/legacy
```
