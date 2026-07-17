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

### Evaluation And Checkpoint Rule

Human-move agreement is computed after masking illegal moves. For each position
sample, the 64 network outputs are ranked only among legal moves in
policy-output square order. Top-N agreement is the fraction of position samples
where the WTHOR move is included in the highest-ranked N legal moves.

For current WTHOR shuffled runs, `best_model.h5` is selected by validation
legal-masked top-1 agreement, stored as `val_legal_top1`. The validation split
is used for both epoch checkpoint selection and model selection across runs.
The test split is used only for final reporting after selection.

If validation legal-masked top-1 is exactly equal across runs, the tie-breakers
are validation legal-masked top-3 agreement, fewer parameters, and shorter
elapsed training time, in that order.

### Resource Metric

`max resident memory MiB` is the largest main-memory working set observed by
`resource_monitor.py` while the training command was running. It is not GPU
memory and it is not the model file size. The raw resource JSON still uses the
key `peak_rss_mib` for backward compatibility.

### Initial WTHOR Sweep

These runs used the same data split. The `512x4` model from this table was
rerun under the current `val_legal_top1` checkpoint rule in the next table.

| config | params | requested epochs | epochs ran | best epoch | checkpoint monitor | val legal top-1 | test legal top-1 | test legal top-3 | test legal top-5 | elapsed sec | max resident memory MiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 20 | 20 | 18 | val_accuracy | 45.541% | 45.590% | 76.834% | 89.749% | 58.7 | 10,396.0 |
| 192x3 | 111,232 | 20 | 20 | 20 | val_accuracy | 48.852% | 48.901% | 80.109% | 91.858% | 61.8 | 10,337.6 |
| 256x4 | 246,848 | 20 | 20 | 20 | val_accuracy | 51.606% | 51.658% | 82.786% | 93.490% | 69.0 | 10,399.7 |
| 384x4 | 517,696 | 20 | 20 | 20 | val_accuracy | 54.665% | 54.776% | 85.426% | 94.920% | 84.2 | 10,384.1 |
| 512x4 | 886,848 | 20 | 20 | 20 | val_accuracy | 56.167% | 56.136% | 86.817% | 95.610% | 91.3 | 10,406.5 |
| 256x4 | 246,848 | 50 | 50 | 50 | val_accuracy | 54.128% | 54.159% | 85.013% | 94.694% | 125.6 | 10,437.7 |
| 384x4 | 517,696 | 50 | 50 | 50 | val_accuracy | 56.839% | 56.871% | 87.330% | 95.913% | 160.2 | 10,424.0 |
| 512x4 | 886,848 | 50 | 50 | 50 | val_accuracy | 58.389% | 58.486% | 88.170% | 96.243% | 186.5 | 10,417.5 |

### Larger-Model Sweep

All runs below used the current checkpoint rule: `checkpoint monitor =
val_legal_top1`.

| config | params | requested epochs | epochs ran | best epoch | val legal top-1 at best | val legal top-3 at best | test legal top-1 | test legal top-2 | test legal top-3 | test legal top-5 | test legal top-10 | elapsed sec | max resident memory MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x4 | 886,848 | 50 | 50 | 50 | 58.318% | 88.125% | 58.496% | 78.561% | 88.233% | 96.259% | 99.853% | 229.2 | 11,674.4 |
| 768x4 | 1,920,064 | 50 | 35 | 27 | 57.687% | 87.391% | 57.701% | 77.663% | 87.451% | 95.822% | 99.836% | 255.5 | 11,671.7 |
| 1024x4 | 3,346,496 | 50 | 25 | 17 | 57.229% | 87.148% | 57.287% | 77.379% | 87.240% | 95.736% | 99.834% | 229.1 | 11,473.7 |
| 512x6 | 1,412,160 | 50 | 50 | 43 | 57.888% | 87.460% | 57.871% | 77.833% | 87.534% | 95.736% | 99.802% | 291.1 | 11,707.6 |
| 768x6 | 3,101,248 | 50 | 25 | 17 | 57.279% | 87.064% | 57.411% | 77.392% | 87.185% | 95.588% | 99.801% | 256.6 | 11,347.0 |
| 1024x6 | 5,445,696 | 50 | 20 | 12 | 57.236% | 87.007% | 57.243% | 77.239% | 87.025% | 95.513% | 99.798% | 261.5 | 11,234.1 |
| 768x4, L2 1e-5, lr 0.0005 | 1,920,064 | 50 | 50 | 50 | 58.504% | 88.203% | 58.539% | 78.606% | 88.283% | 96.295% | 99.874% | 358.0 | 11,724.5 |
| 768x4, dropout 0.05, lr 0.0005 | 1,920,064 | 50 | 50 | 49 | 58.504% | 88.362% | 58.507% | 78.733% | 88.363% | 96.351% | 99.882% | 426.1 | 11,684.5 |

### Adopted Model

The adopted model is `768x4, dropout 0.05, learning rate 0.0005`. Its exact
validation legal-masked top-1 agreement was `0.5850412182276162`, which is two
validation hits higher than the L2 run
(`0.5850387292042094`). Its validation legal-masked top-3 agreement was also
higher. The test split was not used for selection.

Selected artifacts:

- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/best_model.h5`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/best_policy_network_weights.bin`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/best_summary.json`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/w768d4_drop005_lr5e4/split_topn_accuracy.csv`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_large_768x4_dropout_lr_e50.log`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_large_768x4_dropout_lr_e50_resource.json`

Earlier local report snapshots were moved to:

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

### 評価方法とcheckpoint基準

人間着手一致率は、合法手mask後のpolicy network出力で計算しました。各局面サンプルで、
64個のnetwork出力のうち合法手だけをpolicy出力のマス順で残して順位付けします。
top-N一致率は、WTHORで実際に打たれた手が上位N手に含まれた局面サンプルの割合です。

現在のWTHOR shuffled runでは、`best_model.h5` は validation合法手mask後top-1一致率
`val_legal_top1` が最も高いepochです。validation splitはepoch checkpoint選択と
run間のモデル採用に使います。test splitは、採用後の最終確認だけに使います。

validation合法手mask後top-1が完全に同じ場合は、validation合法手mask後top-3一致率、
パラメータ数の少なさ、学習時間の短さの順で比較します。

### リソース指標

`max resident memory MiB` は、学習コマンド実行中に `resource_monitor.py` が観測した
最大のメインメモリ常駐量です。GPUメモリ量ではなく、モデルファイルサイズでも
ありません。過去ログとの互換性のため、raw resource JSONではkey名を
`peak_rss_mib` のままにしています。

### 初期WTHOR sweep

これらは同じdata splitで実行したrunです。この表の `512x4` は、次の表で現在の
`val_legal_top1` checkpoint基準に揃えて再実行しています。

| config | params | 指定epoch | 実行epoch | best epoch | checkpoint monitor | val legal top-1 | test legal top-1 | test legal top-3 | test legal top-5 | elapsed sec | max resident memory MiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 20 | 20 | 18 | val_accuracy | 45.541% | 45.590% | 76.834% | 89.749% | 58.7 | 10,396.0 |
| 192x3 | 111,232 | 20 | 20 | 20 | val_accuracy | 48.852% | 48.901% | 80.109% | 91.858% | 61.8 | 10,337.6 |
| 256x4 | 246,848 | 20 | 20 | 20 | val_accuracy | 51.606% | 51.658% | 82.786% | 93.490% | 69.0 | 10,399.7 |
| 384x4 | 517,696 | 20 | 20 | 20 | val_accuracy | 54.665% | 54.776% | 85.426% | 94.920% | 84.2 | 10,384.1 |
| 512x4 | 886,848 | 20 | 20 | 20 | val_accuracy | 56.167% | 56.136% | 86.817% | 95.610% | 91.3 | 10,406.5 |
| 256x4 | 246,848 | 50 | 50 | 50 | val_accuracy | 54.128% | 54.159% | 85.013% | 94.694% | 125.6 | 10,437.7 |
| 384x4 | 517,696 | 50 | 50 | 50 | val_accuracy | 56.839% | 56.871% | 87.330% | 95.913% | 160.2 | 10,424.0 |
| 512x4 | 886,848 | 50 | 50 | 50 | val_accuracy | 58.389% | 58.486% | 88.170% | 96.243% | 186.5 | 10,417.5 |

### 大きなモデルのsweep

下のrunはすべて現在のcheckpoint基準、つまり `checkpoint monitor = val_legal_top1` で
実行しました。

| config | params | 指定epoch | 実行epoch | best epoch | val legal top-1 at best | val legal top-3 at best | test legal top-1 | test legal top-2 | test legal top-3 | test legal top-5 | test legal top-10 | elapsed sec | max resident memory MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x4 | 886,848 | 50 | 50 | 50 | 58.318% | 88.125% | 58.496% | 78.561% | 88.233% | 96.259% | 99.853% | 229.2 | 11,674.4 |
| 768x4 | 1,920,064 | 50 | 35 | 27 | 57.687% | 87.391% | 57.701% | 77.663% | 87.451% | 95.822% | 99.836% | 255.5 | 11,671.7 |
| 1024x4 | 3,346,496 | 50 | 25 | 17 | 57.229% | 87.148% | 57.287% | 77.379% | 87.240% | 95.736% | 99.834% | 229.1 | 11,473.7 |
| 512x6 | 1,412,160 | 50 | 50 | 43 | 57.888% | 87.460% | 57.871% | 77.833% | 87.534% | 95.736% | 99.802% | 291.1 | 11,707.6 |
| 768x6 | 3,101,248 | 50 | 25 | 17 | 57.279% | 87.064% | 57.411% | 77.392% | 87.185% | 95.588% | 99.801% | 256.6 | 11,347.0 |
| 1024x6 | 5,445,696 | 50 | 20 | 12 | 57.236% | 87.007% | 57.243% | 77.239% | 87.025% | 95.513% | 99.798% | 261.5 | 11,234.1 |
| 768x4, L2 1e-5, lr 0.0005 | 1,920,064 | 50 | 50 | 50 | 58.504% | 88.203% | 58.539% | 78.606% | 88.283% | 96.295% | 99.874% | 358.0 | 11,724.5 |
| 768x4, dropout 0.05, lr 0.0005 | 1,920,064 | 50 | 50 | 49 | 58.504% | 88.362% | 58.507% | 78.733% | 88.363% | 96.351% | 99.882% | 426.1 | 11,684.5 |

### 採用モデル

採用モデルは `768x4, dropout 0.05, learning rate 0.0005` です。validation合法手mask後
top-1一致率の実数値が `0.5850412182276162` で、L2 runの
`0.5850387292042094` よりvalidation hit数で2局面サンプル分だけ高いためです。
validation合法手mask後top-3一致率も、このdropout runのほうが高いです。
test splitは採用判定には使っていません。

採用モデルの生成物:

- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/best_model.h5`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/best_policy_network_weights.bin`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/best_summary.json`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_large_768x4_dropout_lr_e50/w768d4_drop005_lr5e4/split_topn_accuracy.csv`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_large_768x4_dropout_lr_e50.log`
- `src/tools/policy_network_human_like_ai/10_train_policy_network/train_log/wthor_large_768x4_dropout_lr_e50_resource.json`

以前のlocal report snapshotは次に移動済みです。

```text
src/tools/policy_network_human_like_ai/report/legacy
```
