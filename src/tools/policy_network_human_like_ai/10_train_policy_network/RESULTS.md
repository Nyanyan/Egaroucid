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
process before caching, so full WTHOR blend evaluation is not practical in this
session as a direct run.

The WTHOR blend evaluator supports `--jobs`, `--range-start` / `--range-end`,
and `--hint-cache-db`. The sharded runner
`20_test_with_wthor/run_wthor_blend_shards.py` wraps those pieces for resumable
full runs, skips already-finished shards, and creates a shared SQLite
hint-score cache by default.

Small random sample, seed `613`, 30 positions:

| Blend param | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.566667 | 0.866667 | 1.000000 |
| 0.25 | 0.466667 | 0.733333 | 0.900000 |
| 0.50 | 0.466667 | 0.733333 | 0.866667 |
| 0.75 | 0.400000 | 0.700000 | 0.900000 |
| 1.00 | 0.300000 | 0.666667 | 0.900000 |

This sample is only for timing and a rough blend sanity check.

### WTHOR Hint-Cache Planning

`20_test_with_wthor/analyze_wthor_position_duplicates.py` counts exact
Egaroucid hint inputs `(black, white, side)`.

| Metric | Value |
| --- | ---: |
| WTHOR `position_samples` | 8,035,282 |
| Unique hint positions | 5,574,955 |
| Duplicate hint positions | 2,460,327 |
| Duplicate fraction | 0.306190 |
| Invalid side samples | 0 |
| Analysis elapsed | 6.745 sec |
| Analysis in-process peak RSS | 440.371 MiB |
| Resource-monitor elapsed | 7.104 sec |
| Resource-monitor peak RSS | 293.832 MiB |

The cache can remove about 30.6% of repeated hint calls within WTHOR and also
protects interrupted/resumed shard runs from recomputing finished hint scores.
However, the unique hint-position count is still 5.57 million, so a full blend
evaluation remains very large unless the Egaroucid hint cost is reduced or the
evaluation set is sampled.

Cache smoke test on the first two WTHOR position samples:

| Run | Elapsed sec | Peak RSS MiB | Lookups | Hits | Misses | Writes | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First run | 2.032 | 1325.613 | 2 | 0 | 2 | 2 | 2 |
| Second run, same DB | 1.030 | 8.523 | 2 | 2 | 0 | 0 | 2 |

A two-shard smoke run also passed and the merged output preserved cache stats:
2 lookups, 0 hits, 2 misses, 2 writes, 2 rows.

### Resumable Full-Run Controls

The WTHOR sharded runner now supports `--positions-per-shard` and
`--time-limit-sec`. This lets the full WTHOR blend run advance in smaller,
resumable chunks instead of requiring one very large shard to finish.

Smoke test with 5 WTHOR position samples, `positions_per_shard=2`,
`blend_param=1.0`, `top_n=1,3`:

- First invocation used `--time-limit-sec 0.001`: completed shard `0..2`, then
  stopped with `stop_reason=time_limit`.
- Second invocation reused the same output directory, skipped shard 0, ran
  shard `2..4`, then stopped with `stop_reason=max_shards_to_run`.
- Third invocation skipped shards 0 and 1, ran shard `4..5`, and merged all
  outputs with `stop_reason=finished`.

Merged smoke result:

| Blend param | Top-N | Exact accuracy | Symmetric accuracy | Positions |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1 | 0.200000 | 0.400000 | 5 |
| 1.0 | 3 | 0.800000 | 0.800000 | 5 |

### Strength Benchmark

The strength runner starts engine processes lazily instead of prestarting all
player slots. `blend_param=1.0` skips Egaroucid `hint 100`, and blended GTP
engines can cache hint output. Completed tasks are appended to
`strength_games.jsonl`; `--resume` reloads that file and runs only unfinished
tasks. `40_test_strength/run_strength_full.py` wraps the full schedule and
appends a raw run log to `run_strength_full.log`. The strength runner now also
supports `--time-limit-sec`, so the 120,000-game full schedule can be advanced
in bounded resumable chunks.

Smoke results:

- `egaroucid_l1` vs `blend_1.0`, 2 games: 2.034 sec after the policy-only skip.
- `egaroucid_l1` vs `blend_0.0`, 1 game: 37.499 sec, peak RSS 2567.539 MiB.
- Full-player short benchmark, `--max-games 2`: 152.064 sec, peak RSS 5109.445 MiB.
- Time-limit smoke, `egaroucid_l1` vs `blend_1.0`, target 4 games,
  `--time-limit-sec 1`: completed 2/4 games, stopped with
  `stop_reason=time_limit`, wrapper elapsed 3.039 sec, peak RSS 2608.930 MiB.

The full requested schedule is 120,000 games. The short full-player benchmark
suggests a multi-day run even with 32 parallel matches, and `hint 100` required
raising the strength-runner timeout from 60 sec to 300 sec.

## 日本語

### 用語

この実験では、`games` は完全な棋譜、つまり対局数を意味します。学習用に生成された局面は
`position_samples` と呼びます。`records0` や `records1` という既存ディレクトリ名は、
データセットのパス名としてだけ残しています。

### データセット

- 入力元: `train_data/transcript_release/0002`
- 選択: 1,000,000 対局、seed `613`
- 変換成功: 1,000,000 対局
- 変換失敗: 0 対局
- 生成された `position_samples`: 26,384,206
- 変換リソース: 179.371 秒、peak RSS 14.238 MiB

1対局から、ランダム序盤後の複数の着手局面を書き出すため、1,000,000 対局から
26,384,206 局面サンプルが生成されています。

### 学習探索

すべての実行で、学習 5,000,000 `position_samples`、検証 500,000 `position_samples`、
batch size 8192、20 epochs、patience 5、TensorFlow/Keras を使いました。

| Config | Params | Val top-1 | Val top-3 | Val top-5 | Time sec | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x3 | 20,736 | 0.297304 | 0.584340 | 0.742064 | 33.480 | 7751.410 |
| 96x3 | 37,216 | 0.308512 | 0.604394 | 0.763330 | 33.417 | 7787.094 |
| 128x3 | 57,792 | 0.326106 | 0.626642 | 0.783058 | 33.446 | 7782.047 |
| 96x4 | 46,528 | 0.313866 | 0.610754 | 0.767646 | 34.501 | 7784.199 |
| 160x3 | 82,464 | 0.341052 | 0.649038 | 0.804222 | 34.443 | 7794.395 |
| 192x3 | 111,232 | 0.353422 | 0.664336 | 0.816872 | 35.499 | 7738.395 |
| 128x4 | 74,304 | 0.329930 | 0.632604 | 0.787496 | 35.519 | 7652.391 |

検証で最も良かった設定は `192x3` です。

### 修正後の WTHOR 一致率

WTHOR 評価では、policy 出力 index と同じ向きの合法手マスクを使うように修正しました。
途中で出力していた feature-order bit mask の値は破棄してください。

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
WTHOR からランダムに30局面を選んだ評価では 86.157 秒、peak RSS 1283.957 MiB でした。
キャッシュなしで単純外挿すると、WTHOR 全 8,035,282 局面に対して1プロセスで約267日かかるため、
このセッション内で全局面のブレンド評価を直接実行するのは現実的ではありません。

WTHOR ブレンド評価は `--jobs`、`--range-start` / `--range-end`、`--hint-cache-db` に対応しています。
`20_test_with_wthor/run_wthor_blend_shards.py` はそれらをまとめ、完了済み shard を skip しながら
再開可能に実行し、既定で共有 SQLite hint-score cache を作成します。

seed `613`、30局面の小サンプル:

| Blend param | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.566667 | 0.866667 | 1.000000 |
| 0.25 | 0.466667 | 0.733333 | 0.900000 |
| 0.50 | 0.466667 | 0.733333 | 0.866667 |
| 0.75 | 0.400000 | 0.700000 | 0.900000 |
| 1.00 | 0.300000 | 0.666667 | 0.900000 |

この表は時間見積もりとブレンド処理の確認用で、最終的な係数決定にはもっと大きなサンプルか並列評価が必要です。

### WTHOR hint-cache 計画

`20_test_with_wthor/analyze_wthor_position_duplicates.py` で、Egaroucid の `hint 100` に渡す
実盤面キー `(black, white, side)` の重複を数えました。

| Metric | Value |
| --- | ---: |
| WTHOR `position_samples` | 8,035,282 |
| Unique hint positions | 5,574,955 |
| Duplicate hint positions | 2,460,327 |
| Duplicate fraction | 0.306190 |
| Invalid side samples | 0 |
| Analysis elapsed | 6.745 秒 |
| Analysis in-process peak RSS | 440.371 MiB |
| Resource-monitor elapsed | 7.104 秒 |
| Resource-monitor peak RSS | 293.832 MiB |

cache により、WTHOR 内で重複している約30.6%の `hint 100` 呼び出しを省けます。
また、中断・再開した shard 実行でも完了済みの hint score を再計算しなくて済みます。
ただし、unique hint positions はまだ 5.57 million あるため、Egaroucid 側の hint cost を下げるか、
評価対象をサンプリングしない限り、全局面ブレンド評価は依然としてかなり大きい実行になります。

WTHOR 先頭2局面での cache smoke test:

| Run | Elapsed sec | Peak RSS MiB | Lookups | Hits | Misses | Writes | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First run | 2.032 | 1325.613 | 2 | 0 | 2 | 2 | 2 |
| Second run, same DB | 1.030 | 8.523 | 2 | 2 | 0 | 0 | 2 |

2 shard の smoke run も通り、merge 後の出力にも cache stats が残ることを確認しました:
2 lookups、0 hits、2 misses、2 writes、2 rows。

### 再開可能な full 実行制御

WTHOR shard runner は `--positions-per-shard` と `--time-limit-sec` に対応しました。
これにより、全WTHORブレンド評価を巨大な shard 単位ではなく、再開可能な小さい単位で進められます。

WTHOR 5局面、`positions_per_shard=2`、`blend_param=1.0`、`top_n=1,3` の smoke test:

- 1回目は `--time-limit-sec 0.001` で shard `0..2` だけ完了し、
  `stop_reason=time_limit` で停止しました。
- 2回目は同じ出力先を使い、shard 0 を skip して shard `2..4` を実行し、
  `stop_reason=max_shards_to_run` で停止しました。
- 3回目は shard 0 と 1 を skip して shard `4..5` を実行し、全 shard を merge して
  `stop_reason=finished` になりました。

merge 後の smoke 結果:

| Blend param | Top-N | Exact accuracy | Symmetric accuracy | Positions |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1 | 0.200000 | 0.400000 | 5 |
| 1.0 | 3 | 0.800000 | 0.800000 | 5 |

### 強さ評価ベンチマーク

強さ評価 runner は、全 player slot を事前起動する方式から lazy start に変更しました。
`blend_param=1.0` では Egaroucid `hint 100` を呼ばず、ブレンド GTP engine では hint output を
cache できます。完了済み task は `strength_games.jsonl` に追記され、`--resume` で未完了 task だけを
実行できます。`40_test_strength/run_strength_full.py` は full schedule 用の wrapper で、raw run log を
`run_strength_full.log` に追記します。`--time-limit-sec` にも対応したため、120,000対局の full schedule を
時間で区切って再開可能に進められます。

smoke 結果:

- `egaroucid_l1` vs `blend_1.0`、2対局: policy-only skip 後 2.034 秒。
- `egaroucid_l1` vs `blend_0.0`、1対局: 37.499 秒、peak RSS 2567.539 MiB。
- 全プレイヤー構成の短縮ベンチ `--max-games 2`: 152.064 秒、peak RSS 5109.445 MiB。
- time-limit smoke、`egaroucid_l1` vs `blend_1.0`、目標4対局、
  `--time-limit-sec 1`: 2/4 対局を完了し、`stop_reason=time_limit` で停止。
  wrapper elapsed 3.039 秒、peak RSS 2608.930 MiB。

要求された full schedule は 120,000 対局です。短縮ベンチから見ても、32並列でも数日規模の実行になる見込みです。
また `hint 100` が60秒で戻らない局面があったため、strength runner の timeout 既定値を300秒に上げました。
