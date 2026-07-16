# Training Results

Related issue: #613

## English

### Terminology

In this experiment, `games` means complete game transcripts. The generated
training positions are called `position_samples`. Existing directory names such
as `records0` and `records1` are kept only as dataset paths.

### Dataset

- Source: `train_data/transcript_release/0002`
- Selection: 1,000,000 games, seed `613`
- Converted games: 1,000,000 valid, 0 invalid
- Generated `position_samples`: 26,384,206
- Conversion resource: 179.371 sec, peak RSS 14.238 MiB

The number of `position_samples` is larger than the number of games because one
game contributes many move positions after the random opening segment.

### Training Search

All runs used 5,000,000 train `position_samples`, 500,000 validation
`position_samples`, batch size 8192, 20 epochs, patience 5, TensorFlow/Keras.

| Config | Params | Val top-1 | Val top-3 | Val top-5 | Time sec | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x3 | 20,736 | 0.297304 | 0.584340 | 0.742064 | 33.480 | 7751.410 |
| 96x3 | 37,216 | 0.308512 | 0.604394 | 0.763330 | 33.417 | 7787.094 |
| 128x3 | 57,792 | 0.326106 | 0.626642 | 0.783058 | 33.446 | 7782.047 |
| 96x4 | 46,528 | 0.313866 | 0.610754 | 0.767646 | 34.501 | 7784.199 |
| 160x3 | 82,464 | 0.341052 | 0.649038 | 0.804222 | 34.443 | 7794.395 |
| 192x3 | 111,232 | 0.353422 | 0.664336 | 0.816872 | 35.499 | 7738.395 |
| 128x4 | 74,304 | 0.329930 | 0.632604 | 0.787496 | 35.519 | 7652.391 |

Best validation config: `192x3`.

### Correct WTHOR Agreement

The WTHOR evaluator now uses a policy-output-order legal mask. Earlier
intermediate outputs used the feature-order bit mask and should be ignored.

| Config | Positions | Top-1 exact | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192x3 | 8,035,282 | 0.338097 | 0.360323 | 0.707869 | 0.870466 |
| 160x3 | 8,035,282 | 0.328847 | 0.351074 | 0.683399 | 0.852199 |
| 128x4 | 8,035,282 | 0.313147 | 0.335372 | 0.661174 | 0.841202 |
| 128x3 | 8,035,282 | 0.307269 | 0.329494 | 0.663007 | 0.839235 |
| 96x3 | 8,035,282 | 0.298108 | 0.314827 | 0.648127 | 0.824891 |
| 96x4 | 8,035,282 | 0.292177 | 0.308892 | 0.652110 | 0.829590 |
| 64x3 | 8,035,282 | 0.289575 | 0.306292 | 0.631556 | 0.810074 |

Best WTHOR config: `192x3`.

`192x3` by 10-move bucket:

| Moves | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: |
| 01-10 | 0.377334 | 0.812311 | 0.964305 |
| 11-20 | 0.292862 | 0.612274 | 0.799897 |
| 21-30 | 0.262999 | 0.577686 | 0.775109 |
| 31-40 | 0.280449 | 0.614852 | 0.811226 |
| 41-50 | 0.354634 | 0.715394 | 0.887165 |
| 51-60 | 0.595946 | 0.916727 | 0.986230 |

### Blend Benchmark

`hint 100` at Egaroucid Console 7.8.1 level 21 is expensive per position. A
30-position random WTHOR sample took 86.157 sec and peak RSS 1283.957 MiB.
That extrapolates to roughly 267 days for all 8,035,282 WTHOR positions on one
process, so full WTHOR blend evaluation is not practical in this session.

Small random sample, seed `613`, 30 positions:

| Blend param | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.566667 | 0.866667 | 1.000000 |
| 0.25 | 0.466667 | 0.733333 | 0.900000 |
| 0.50 | 0.466667 | 0.733333 | 0.866667 |
| 0.75 | 0.400000 | 0.700000 | 0.900000 |
| 1.00 | 0.300000 | 0.666667 | 0.900000 |

This sample is only for timing and a rough blend sanity check.

## 日本語

### 用語

この実験では `games` は完全な棋譜の対局数を意味します。生成された学習用局面は
`position_samples` と呼びます。`records0` や `records1` という既存ディレクトリ名は、
データセットのパス名としてのみ残しています。

### データセット

- 入力元: `train_data/transcript_release/0002`
- 選択: 1,000,000 対局、seed `613`
- 変換成功: 1,000,000 対局
- 変換失敗: 0 対局
- 生成 `position_samples`: 26,384,206
- 変換リソース: 179.371 秒、peak RSS 14.238 MiB

1対局から、ランダム序盤後の複数の着手局面を学習用に書き出すため、1,000,000 対局
から 26,384,206 局面サンプルが生成されます。

### 学習探索

全設定で、学習 5,000,000 局面サンプル、検証 500,000 局面サンプル、batch size
8192、20 epochs、patience 5、TensorFlow/Keras を使いました。

| Config | Params | Val top-1 | Val top-3 | Val top-5 | Time sec | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x3 | 20,736 | 0.297304 | 0.584340 | 0.742064 | 33.480 | 7751.410 |
| 96x3 | 37,216 | 0.308512 | 0.604394 | 0.763330 | 33.417 | 7787.094 |
| 128x3 | 57,792 | 0.326106 | 0.626642 | 0.783058 | 33.446 | 7782.047 |
| 96x4 | 46,528 | 0.313866 | 0.610754 | 0.767646 | 34.501 | 7784.199 |
| 160x3 | 82,464 | 0.341052 | 0.649038 | 0.804222 | 34.443 | 7794.395 |
| 192x3 | 111,232 | 0.353422 | 0.664336 | 0.816872 | 35.499 | 7738.395 |
| 128x4 | 74,304 | 0.329930 | 0.632604 | 0.787496 | 35.519 | 7652.391 |

検証で最良だった設定は `192x3` です。

### 修正後の WTHOR 一致率

WTHOR 評価では、policy 出力 index と同じ向きの合法手マスクを使うように修正しました。
途中で出力した feature-order bit mask の値は破棄してください。

| Config | Positions | Top-1 exact | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192x3 | 8,035,282 | 0.338097 | 0.360323 | 0.707869 | 0.870466 |
| 160x3 | 8,035,282 | 0.328847 | 0.351074 | 0.683399 | 0.852199 |
| 128x4 | 8,035,282 | 0.313147 | 0.335372 | 0.661174 | 0.841202 |
| 128x3 | 8,035,282 | 0.307269 | 0.329494 | 0.663007 | 0.839235 |
| 96x3 | 8,035,282 | 0.298108 | 0.314827 | 0.648127 | 0.824891 |
| 96x4 | 8,035,282 | 0.292177 | 0.308892 | 0.652110 | 0.829590 |
| 64x3 | 8,035,282 | 0.289575 | 0.306292 | 0.631556 | 0.810074 |

WTHOR 一致率でも最良は `192x3` でした。

`192x3` の10手刻み:

| Moves | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: |
| 01-10 | 0.377334 | 0.812311 | 0.964305 |
| 11-20 | 0.292862 | 0.612274 | 0.799897 |
| 21-30 | 0.262999 | 0.577686 | 0.775109 |
| 31-40 | 0.280449 | 0.614852 | 0.811226 |
| 41-50 | 0.354634 | 0.715394 | 0.887165 |
| 51-60 | 0.595946 | 0.916727 | 0.986230 |

### ブレンド評価ベンチマーク

Egaroucid Console 7.8.1 level 21 の `hint 100` は1局面あたりかなり重いです。
WTHOR からランダムに30局面を選んだ評価では 86.157 秒、peak RSS 1283.957 MiB
でした。単純換算では WTHOR 全 8,035,282 局面に対して1プロセスで約267日かかるため、
このセッション内で全局面のブレンド評価を完走するのは現実的ではありません。

seed `613`、30局面の小サンプル:

| Blend param | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.566667 | 0.866667 | 1.000000 |
| 0.25 | 0.466667 | 0.733333 | 0.900000 |
| 0.50 | 0.466667 | 0.733333 | 0.866667 |
| 0.75 | 0.400000 | 0.700000 | 0.900000 |
| 1.00 | 0.300000 | 0.666667 | 0.900000 |

この表は時間見積もりとブレンド処理の確認用で、最終的な係数決定にはより大きな
サンプルまたは並列化した評価が必要です。
